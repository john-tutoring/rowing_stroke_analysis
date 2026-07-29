# StrokeScore — how it all actually works

This is the longer version of the README. It goes file by file and explains what
each one does, why it does it that way, and the gotchas I ran into. I wrote most
of this for myself, so future-me remembers why things are the way they are.

---

## The big picture

Here's the whole trip a video takes:

```
your browser                          the server (app.py)
------------                          --------------------
pick a file  ──►  video starts playing from your own computer
hit Analyze  ──►  upload  ─────────►  save the file temporarily
                                      extraction.py  — find the body in each frame
                                      split_strokes.py — cut it into strokes
                                      feature_extraction.py — measure each stroke
                                      model.joblib   — score each stroke
                                      delete the file
             ◄──  JSON  ◄──────────   scores + body coordinates + per-frame numbers
draw the stick figure on a canvas sitting on top of the video
```

There are drawn-out versions of this and two other diagrams — the training pipeline
and the deployment steps — in [`diagrams.md`](diagrams.md).

The important design decision: **the server never sends the video back.** Your
browser already has the file, so it plays that copy, and the server only ever
returns numbers. That keeps the response tiny (about 200 KB for an 11-second clip
instead of tens of megabytes) and means I'm not storing anyone's video anywhere.

---

## The app files

### `app.py` — the server

This is a Flask app, which is basically the simplest way to make Python answer web
requests. It has four routes. `GET /` hands over the HTML page, `GET /static/...`
hands over the CSS and JavaScript, `GET /health` just says "yes I'm alive" (hosting
platforms like to ping something), and `POST /predict` is where all the real work
happens. When a video arrives it gets saved into `uploads/`, run through the whole
pipeline by a function called `video_to_predictions`, and then deleted — that
delete is in a `finally` block so it happens even if something crashes partway
through.

The other two functions here are about packaging things up for the browser.
`pose_payload()` pulls out just the X and Y of every body point (dropping Z, which
the drawing code doesn't need) and converts them back into plain 0-to-1
coordinates. `frame_metrics()` computes the numbers the "Frame analysis" tab shows
live: hip, knee, and elbow angles for every frame, plus how far the handle has
traveled and its speed and acceleration. The trained model gets loaded once when
the file is first imported, not on every request, because loading it every time
would be silly. There's a 200 MB cap on uploads.

### `index.html` — the page

Pure structure, no logic. It sets up the upload form, the video box (which is
actually three things stacked on top of each other: the `<video>`, a `<canvas>`
for the stick figure, and a little floating badge showing the current stroke and
its score), the three tab buttons, and the three panels those tabs switch between.

The one external thing it pulls in is Chart.js from a CDN, which draws the bar
chart on the stroke tab. Everything else is my own code. The panels for tabs that
aren't showing just have the `hidden` attribute on them.

### `static/style.css` — the look

Dark theme, set up with CSS variables at the top of the file so all the colors
(the hip pink, the knee blue, the elbow amber, etc.) are defined in one place. The
JavaScript reads those same variables when it draws the circles on the joints, so
the color of the ring on the hip and the color of the dot next to "Hip angle"
always match without me having to keep two lists in sync.

The part that took the most fiddling is `.pose-wrap`, the box holding the video and
the canvas. It's `position: relative` with the video at `width: 100%; height: auto`,
which means the video element's box is *exactly* the size of the video picture —
no black bars on the sides. That matters a lot, because the canvas sits right on
top at 100% by 100%, so drawing a body point is just "multiply its coordinate by
the canvas size." If there were black bars, all the math would be off. The canvas
also has `pointer-events: none` so clicks pass straight through it to the video's
play button.

### `static/app.js` — everything the page does

This is the biggest file in the project and it's all plain JavaScript, no React or
anything. It handles the upload, draws the stick figure, keeps that drawing in sync
with the video, switches tabs, builds the chart, and computes the summary stats.

**Drawing the figure.** There's a list called `POSE_EDGES` of the 16 pairs of
points that should be connected by a line — shoulders to each other, shoulder to
elbow to wrist on both arms, hip to knee to ankle to heel to toe on both legs, and
so on. I skip the ten face points because a face made of dots looks bad, and just
draw a circle for the head at the nose. Each line gets drawn twice: first a thick
dark see-through version, then the actual colored line on top. That dark outline is
so the figure is still visible when someone's wearing a white shirt.

**Staying in sync with the video.** The video's own `currentTime` is the clock for
everything. Every time the browser is ready to draw a frame, `frameAt()` finds
whichever row of pose data is closest to the current video time. Since the video
moves forward in tiny steps, it just walks a pointer forward through the list
instead of searching from scratch. When the video is paused, the loop stops
entirely and only redraws if you scrub or switch tabs. If the nearest pose row is
more than a quarter second away from where the video is — meaning MediaPipe just
couldn't find a body for a while — the figure isn't drawn at all and the numbers
show a dash, rather than freezing on stale data and looking broken.

**The iPhone problem.** iPhone `.MOV` files are HEVC, which Firefox and Chrome on
Linux can't play. When the `<video>` element fires an error, the whole panel flips
into a mode where the video is hidden, the box takes the aspect ratio the server
reported, and a timer loops through the pose data on its own so the stick figure
still animates. You lose the actual footage but keep the analysis.

**Coloring.** The stick figure is drawn the same way on every tab, only the color
changes. On the frame tab it's neutral gray with the colored rings on the joints;
on the stroke tab it takes the color of whichever stroke is currently playing; on
the summary tab it's just green. Figuring out "which stroke is this row in" happens
through a lookup array built once when the results come in, so it's instant rather
than searching every frame.

---

## The pipeline files

### `joint_info.py` — the list of body parts

The smallest and most boring file, and also the one everything else depends on.
MediaPipe returns 33 points in a fixed order, and this file writes down that order,
builds the CSV header row from it, and makes a dictionary called `COL_LOOKUP` that
maps a name like `right_knee_X` to a column number.

The reason this exists is that everything downstream is working with big NumPy
arrays where columns are just numbers. Without this file, the code would be full of
things like `arr[:, 78]` and it would be impossible to read or change. With it, I
write `COL_LOOKUP["right_knee_X"]` instead. There's also a separate
`CYCLE_HEADER` for the stroke CSV, which adds one extra column at the *end* for
which video a stroke came from — it goes at the end specifically so every existing
column number stays valid.

### `extraction.py` — video in, body points out

This opens a video with OpenCV, feeds frames to MediaPipe's `PoseLandmarker`, and
collects the results. It runs on **every other frame** (`FRAME_MODULUS = 2`),
which is plenty — a stroke takes a couple of seconds and nothing important happens
in 1/60th of a second. Frames where no body is found are thrown out completely,
which is why the row number is *not* a reliable clock and every row carries its own
timestamp instead.

Then it smooths everything. MediaPipe's output jitters a bit frame to frame, and
since the features involve derivatives — speed and acceleration — a little jitter
turns into a lot of noise. So each column gets a 5-frame moving average. The
timestamps get the exact same 5-frame average applied to them, so row `i` of the
coordinates still lines up with time `i`.

**The coordinate gotcha, which cost me an afternoon:** MediaPipe gives x as a
fraction of the frame's *width* and y as a fraction of its *height*. On a 16:9
video those are different scales, so a 45° angle in real life doesn't come out as
45° when you do the math. The fix is to store `y × (height / width)`, putting y
into the same units as x, and that's what's in the CSV. The consequence is that
stored y is not between 0 and 1 — it's between 0 and about 0.56 for a normal
landscape video. All the angle math uses these adjusted numbers, and `app.py`
converts back before sending coordinates to the browser. If the stick figure ever
looks vertically squashed, this conversion is the first thing to check.

### `split_strokes.py` — cutting the video into strokes

Every rowing stroke is one full out-and-back of the handle, so if I plot "how far
is the wrist in front of the ankle" over time, I get a clean wave with one bump per
stroke. `scipy.signal.find_peaks` finds the tops of those bumps, and a stroke is
everything between two consecutive peaks. Two settings control this:
`PEAK_DISTANCE` (peaks must be at least 10 sampled frames apart) and
`PEAK_PROMINENCE` (a bump has to be a real bump, not a wiggle). If there are N
peaks you get N−1 strokes, and whatever happens before the first peak or after the
last one doesn't belong to any stroke.

There are two versions of the splitting function because the app and the training
code need slightly different things. Training uses `split_cycles_with_videos()`,
which also tracks which video each stroke came from — that's needed so I can test
the model on a rower it has never seen. The web app uses
`split_cycles_with_ranges()`, which additionally gives back the start and end row
number of each stroke. Those row numbers are the key to the whole front end: since
the coordinates, the timestamps, and the per-frame numbers all share the same row
numbering, "stroke 3" can be turned into "the video from 4.2 s to 6.0 s."

This file also has `right_side_closer()`, which decides which side of the body is
facing the camera. MediaPipe's Z value is depth relative to the middle of the hips,
and more negative means closer to the camera, so I just average Z over the
shoulder, elbow, wrist, hip, knee, and ankle for each side and take whichever is
smaller. On the sample videos the two sides differ by so much that no threshold or
tie-breaking is needed. This matters because MediaPipe is basically guessing where
the far arm and leg are — they're hidden behind the body — so every measurement
reads the near side only.

### `feature_extraction.py` — measuring one stroke

This turns one stroke's worth of coordinates into six numbers. The building block
is `angle_abc()`, which takes three points and returns the angle at the middle one,
using the dot product formula. `get_angle_vector()` runs that on every frame of a
stroke to get an angle-over-time curve. All the angles are computed from X and Y
only — the rowers are filmed in profile so everything I care about is in the flat
plane of the image, and MediaPipe's depth channel is its least reliable output.

The six features and what they mean:

| Feature | How it's calculated | Units |
|---|---|---|
| `min_hip_angle` | Smallest knee–hip–shoulder angle in the stroke | degrees |
| `fastest_hip_velocity_timing` | Where in the stroke the hip angle is opening fastest | 0–1 |
| `knee_min_accel_timing` | Where the knee's angular acceleration is closest to zero | 0–1 |
| `body_angle_at_catch` | Torso angle at the catch, measured against the seat rail | degrees |
| `leg_back_lag` | Peak knee speed minus peak hip speed | −1 to 1 |
| `elbow_angle_range` | Biggest elbow angle minus smallest | degrees |

Two of these deserve extra explanation. `body_angle_at_catch` needs to know which
way is "flat," and I can't use the bottom of the image because nobody films with a
perfectly level phone. So `slide_axis()` looks at the path the hip traces over the
stroke — which is a straight line along the seat rail, because that's the only way
the seat can move — and uses that line as the reference direction. Camera tilt
tilts the rail and the rower equally, so the angle between them doesn't change.
And `catch_index()` finds the catch by looking for the frame where the knee is most
bent, which is the most compressed point of the stroke.

The list at the bottom, `DEFAULT_FEATURE_EXTRACTORS`, is the official feature list,
and **the order matters** — that's the column order the model was trained with, so
shuffling it would silently give wrong scores. There are a couple of extra feature
functions in the file that aren't in the list; those are ones I tried and didn't
keep.

---

## The training files

### `model_compare.py` — which model is least bad

This runs about ten different regression models on the same data and prints two
tables. Regression means the model predicts a number (a score) rather than a
category. The models range from a "dummy" that just guesses the average every time
— if a real model can't beat that, it's learned nothing — through linear models
like Ridge and Lasso, to tree-based ones like random forests.

The two tables are the important part, and they disagree with each other on
purpose. The **first** is normal 5-fold cross-validation: shuffle all the strokes,
train on 80%, test on 20%, five times. The problem is that strokes from the same
video look almost identical to each other, so the model gets tested on stroke 4 of
a video it already saw strokes 1, 2, 3, and 5 from. Of course it does well. The
**second** table is leave-one-video-out: hold out one entire video, train on the
rest, predict that video, repeat for all 15. That's the real test, because it's
exactly what the app faces — someone it's never seen. The numbers in the second
table are much worse and the ranking is different, and the second table is the one
to believe. Its `video_mae` column is the honest one: it averages a video's stroke
predictions and compares that to the coach's grade, which is the number the app
actually displays.

### `hyperparam_tune.py` — turning one knob at a time

A hyperparameter is a setting you choose before training rather than something the
model learns — how strongly to penalize big coefficients, how deep a tree can go,
that kind of thing. This is a little interactive terminal menu: pick a model, pick
one of its settings, and it sweeps through ten values of that setting and prints
the cross-validation score for each so you can see where the sweet spot is.

It's a scratchpad tool, not part of the app. It shares its data-loading and
feature-building code with `model_compare.py` so both are always measuring the same
thing. Honestly, with only 175 strokes there isn't much to gain from tuning — the
differences between settings are mostly smaller than the noise — but it was worth
building to check that.

### `export_model.py` — saving the finished model

Once you've decided on a model, this trains it on *all* the data (no holding
anything out, since we're done testing) and saves it to `artifacts/model.joblib`
using joblib, which is basically "save this Python object to a file." The app
loads that file at startup instead of retraining, which turns something that takes
several seconds into something that takes milliseconds.

The model is a two-step pipeline: a `MinMaxScaler` that squashes every feature into
0-to-1 so that a feature measured in degrees doesn't automatically dominate one
measured as a 0-to-1 fraction, then `Ridge(alpha=1.0)`, which is linear regression
with a penalty that keeps the coefficients from getting extreme. Ridge is a good
fit here mostly because the dataset is tiny — a fancier model would just memorize
15 videos. There's one line in the file marked as the line to change if you find
something better. It also writes `model_meta.json` next to it recording what was
used, how many strokes it was trained on, and the feature names in order. One
warning: Ridge doesn't know scores are supposed to be 0–100, so individual strokes
can come out at 104 or −3. The app only clamps that for picking a color, never for
the number it shows you.

### `plot_cycle_signals.py` — the debugging tool

Point this at a single video and it draws the wrist-minus-ankle wave with an ×
marking every peak the splitter found, then saves it as a PNG next to the video.
When a video gives a weird number of strokes, this shows you why in about two
seconds — usually it's a wobble in the pose data creating an extra peak, or a very
slow stroke whose bump wasn't prominent enough to count.

It deliberately repeats the peak-finding settings rather than importing them, so
you have to keep the two in sync manually — worth knowing if you ever change
`PEAK_DISTANCE` or `PEAK_PROMINENCE` in `split_strokes.py`.

---

## Data and setup files

### `pose_landmarker_lite.task`

Google's pre-trained pose model, about 5.5 MB. I didn't train this and I couldn't
— it took an enormous dataset and a lot of compute. "Lite" is the smallest of the
three sizes they offer; the bigger ones are more accurate but slower, and the lite
one is plenty for a clearly-lit side view of one person.

### `artifacts/model.joblib` and `artifacts/model_meta.json`

The joblib file is my trained model, saved so the app can load it instantly. The
JSON next to it is a plain-text note recording what's inside: Ridge, alpha 1.0,
MinMaxScaler, 175 strokes, 6 features, and the feature names in order. That's
purely for humans — nothing reads it — but it means I can tell what a saved model
is without having to load it up in Python.

One thing to watch: joblib files are a bit fussy about scikit-learn versions. If
the version that saved the file is very different from the version trying to load
it, you'll get warnings or errors. Keeping `requirements.txt` the same in both
places avoids this.

### `all_videos_all_joints.csv` and `cycle_data.csv`

`all_videos_all_joints.csv` (~13 MB) is the raw output of `extraction.py`: one row
per kept frame, 101 columns, all videos stacked on top of each other with a video
number in the first column so you can tell them apart. `cycle_data.csv` (~12 MB) is
the same data after `split_strokes.py` has cut it into strokes — one extra column
at the front for the stroke number and one at the end for the video number, and
frames that didn't fall inside a stroke are gone.

These exist so I don't have to re-run pose extraction every time I want to try a
different model. Extraction takes minutes; loading a CSV takes a second. They're
both in `.gitignore` because they're big and completely regenerable.

### `SampleVideos/` and `uploads/`

`SampleVideos/` holds the training videos, each with the coach's grade as digits
right before the extension (`ManRowing_95.mp4` → grade 95). `extraction.py` pulls
the number straight out of the filename with a regular expression, so getting the
naming wrong silently gives a stroke the wrong label. There are 15 videos, and
between them they produced 175 strokes.

`uploads/` is where an uploaded video sits for the few seconds it's being analyzed
before `app.py` deletes it. It contains a `.gitkeep` file, which is an empty file
whose only job is to make git keep an otherwise-empty folder.

### `requirements.txt`

The Python libraries: NumPy for arrays, SciPy for peak-finding, OpenCV for reading
video files, MediaPipe for pose detection, scikit-learn for the model, joblib for
saving it, matplotlib for the debug plot, and Flask plus gunicorn for the web
server. Flask is the development server and gunicorn is the production one — Flask
straight up warns you not to use it for real traffic.

### `Dockerfile` and `.dockerignore`

Docker packages the app together with a specific Python version and all its
libraries, so it runs the same on a server as it does on my laptop. The Dockerfile
starts from a slim Python 3.12 image, installs some Linux system libraries that
OpenCV and MediaPipe need, installs the Python packages, copies in only what the
app actually needs, and starts gunicorn.

Two details worth remembering. The copy step lists files individually, so **if you
add a new front-end file you have to add it to the Dockerfile too**, otherwise it
just won't exist in the container and you'll get a confusing 404. And gunicorn runs
with `-t 300`, a five-minute timeout, because a long video really can take minutes
to analyze and the default 30-second timeout would kill it mid-request. Analysis
time depends on how long the video is, not how high-resolution it is.

`.dockerignore` is the list of things to leave out: the videos, the CSVs, the
training scripts, the virtual environment. The container only needs to *run* the
model, not train it, so leaving all that out keeps the image much smaller.

### `.gitignore`

Keeps the big regenerable stuff off GitHub: `cycle_data.csv`, the `SampleVideos/`
folder, the debug PNGs, `__pycache__/`, the virtual environment, and the contents
of `uploads/`. Video files are big and GitHub isn't meant for them.

---

## What `/predict` sends back

The response is one JSON object. Everything in it that's per-frame is in the same
row order, which is what makes the front end work.

```jsonc
{
  "strokes_detected": 4,
  "near_side": "right",                   // which side faced the camera
  "predicted_grade_mean": 76.35,
  "strokes": [
    {
      "stroke": 1,                        // starts at 1, not 0
      "predicted_grade": 77.1,
      "features": {                       // the six measurements for this stroke
        "min_hip_angle": 41.2, "fastest_hip_velocity_timing": 0.3125,
        "knee_min_accel_timing": 0.5417, "body_angle_at_catch": 63.99,
        "leg_back_lag": -0.0833, "elbow_angle_range": 71.44
      }
    }
  ],
  "stroke_ranges": [[26, 86], [86, 171]], // first and last row of each stroke

  "pose": {                               // this draws the stick figure
    "fps": 60.0, "width": 1920, "height": 1080,
    "landmarks": ["nose", "...", "right_foot_index"],  // 33 names, fixed order
    "t":  [0.10, 0.13],                   // the time of each row, in seconds
    "xy": [[0.51, 0.33, "..."]]           // 66 numbers per row, both 0 to 1
  },

  "metrics": {                            // this drives the live numbers
    "hip_angle":   [],                    // degrees
    "knee_angle":  [],
    "elbow_angle": [],
    "wrist_x":     [],                    // how far the handle has traveled
    "wrist_v":     [],                    // its speed
    "wrist_a":     []                     // its acceleration
  }
}
```

The rule that ties it together: `pose.t`, `pose.xy`, and every array in `metrics`
have exactly the same length, and row `i` of all of them describes the same instant
`t[i]`. Every number in `stroke_ranges` is a valid row number in that same space.
Speed and acceleration are computed with `np.gradient(values, t)` rather than
assuming even spacing, because dropped frames mean the spacing *isn't* even.

Errors come back as `{"error": "..."}` — status 400 for things the user can fix
(no person found, no strokes found, no file attached) and 500 for anything
unexpected.

---

## A few front-end details worth writing down

**Score colors.** One function, `gradeColor()`, decides every color in the app, so
the bars, the stick figure, the badge, and the summary numbers always agree. The
basic idea is hue = 120 × score / 110 in HSL, which runs red at 0 through green at
110. The complication is that a decent video where every stroke lands between 70
and 80 would render as eight nearly identical greens, which is useless. So the
scale gets stretched per video: the best stroke keeps its true color, and the worst
stroke gets pushed toward red until the video's colors span at least 50° of hue.
If a video's scores are already spread out enough, it's left alone.

**The chart.** Chart.js, one bar per stroke. Picking a feature radio button adds a
dashed white line on a second y-axis on the right, since the features are in
completely different units from the scores. The chart is built the first time you
open that tab, not when the results arrive, because Chart.js can't measure a canvas
that's inside a hidden panel and ends up sizing it to zero.

**Summary math**, all done in the browser:
- Stroke rate = 60 × number of strokes ÷ (time of last stroke's end − time of
  first stroke's start).
- Best and worst are just the highest and lowest scoring strokes.
- Consistency = 100 minus the standard deviation of the scores, clamped to 0–100.
  Standard deviation measures spread, so low spread means every stroke came out
  about the same, which is what you want. The raw ± value is shown underneath.

**A CSS trap I hit:** HTML's `hidden` attribute stops working if you also give the
element an explicit `display: flex`. You need a `[hidden] { display: none }` rule
to override it. This is already handled but it's the kind of thing that wastes an
hour if you don't know it.

---

## Running it in dev vs production

Locally it's `python app.py` on port 5000. In Docker it's gunicorn on port 8080.
It runs with one worker, because MediaPipe and OpenCV are both memory-hungry and
one request already uses a lot of CPU — more workers on a small server just makes
everything slower.
