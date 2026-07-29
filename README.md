# StrokeScore — scoring rowing form from a phone video

Record yourself on an erg from the side, upload the video, and this app draws a
stick figure on top of you and gives every single stroke a score out of 100.

I row, and one thing that always bugged me is that the erg monitor tells you your
split and your stroke rate but says nothing about whether your form is any good.
A coach can watch you and tell instantly, but a coach can't be there for every
piece. So I tried to build something that watches the video and does a small part
of what a coach does — checking how far you compress at the catch, how far you
reach, whether your legs go before your back, and how far you draw the handle in.

There's a longer, nerdier writeup in [`details_technical.md`](details_technical.md)
that explains how each piece actually works, and [`diagrams.md`](diagrams.md) has
pictures of the training pipeline, the running app, and how to deploy it.

## How it works, in four steps

1. **Find the body.** Google's MediaPipe pose model looks at each frame of the
   video and gives back 33 dots — shoulders, elbows, wrists, hips, knees, ankles,
   and so on.
2. **Cut the video into strokes.** I track how far the wrist is in front of the
   ankle. On an erg that number goes up and down once per stroke, like a wave, so
   the peaks of that wave are the stroke boundaries.
3. **Measure each stroke.** For every stroke I calculate six numbers about the
   rower's body — angles and timings that coaches actually care about.
4. **Predict a score.** Those six numbers go into a small machine learning model
   that was trained on videos a coach already graded, and it guesses a score.

## What you see in the app

There's the video with an animated stick figure drawn over it, and three tabs
underneath:

- **Stroke analysis** — one bar per stroke, colored red (rough) to green (good).
  The stick figure changes color as the video plays so you can see which stroke
  you're watching. You can also lay any one of the six measurements over the chart
  to see how it lines up with the scores.
- **Frame analysis** — colored circles on the hip, knee, elbow, and wrist, and
  live numbers underneath that change as the video plays: joint angles, how far
  the handle has traveled, how fast it's moving, and its acceleration.
- **Session summary** — how many strokes, your stroke rate, average score, best
  and worst stroke, and a consistency percentage.

Your video never gets stored. The browser plays the copy that's already on your
computer, and the server deletes the uploaded file as soon as it's done doing the
math. Only numbers come back.

Two things worth knowing: videos can be up to 200 MB, but a long video takes a
long time to analyze (it has to look at every other frame). And iPhone `.MOV`
files use a codec that some browsers can't play — if that happens, the video stays
blank but the stick figure animates on its own so you still get the analysis.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python export_model.py      # only needed once, and only if artifacts/model.joblib is missing
python app.py               # then open http://127.0.0.1:5000
```

The app loads the already-trained model when it starts up. It never trains while
it's running — that would be way too slow.

### With Docker

```bash
docker build -t strokescore .
docker run --rm -p 8080:8080 strokescore   # open http://localhost:8080
```

I used this to put it on Railway: connect the repo, make sure
`artifacts/model.joblib` actually got committed, and point it at the Dockerfile.

## Training it on your own videos

Put graded videos in `SampleVideos/` and put the coach's grade in the filename
right before the extension, like `avery_90.mp4`. Then run these in order:

```bash
python extraction.py        # videos    → all_videos_all_joints.csv
python split_strokes.py     # that CSV  → cycle_data.csv (one stroke at a time)
python model_compare.py     # try a bunch of models, see which is least bad
python export_model.py      # save the chosen model to artifacts/model.joblib
```

If a video's strokes are being split in weird places, this draws a picture of the
wave I use for splitting so you can see what went wrong:

```bash
python plot_cycle_signals.py SampleVideos/whatever.mp4
```

## The six things it measures

| What it's called | What it means in rowing terms |
|---|---|
| `min_hip_angle` | How closed your hip angle gets — how compressed you are at the catch |
| `body_angle_at_catch` | How far your body is leaning forward at the catch (your reach) |
| `fastest_hip_velocity_timing` | When in the stroke your hips swing open fastest |
| `knee_min_accel_timing` | When your knees stop speeding up — roughly when the leg drive finishes |
| `leg_back_lag` | Whether your legs peak before your back does (legs → back → arms) |
| `elbow_angle_range` | How far you actually draw the handle in with your arms |

The timing ones are stored as a fraction of the stroke (0 = start, 1 = end) so
that a slow stroke and a fast stroke can still be compared. The angles are just
degrees.

## What each file does

There are diagrams of how these fit together in [`diagrams.md`](diagrams.md).

**The app itself**

| File | What it does |
|---|---|
| `app.py` | The server. Takes the uploaded video, runs the whole pipeline, sends back the scores and the stick-figure coordinates as JSON. |
| `index.html` | The page itself — the upload button, the video box, the three tabs. |
| `static/style.css` | All the styling. Dark theme, colors, layout. |
| `static/app.js` | Everything the page *does*: uploading, drawing the stick figure on the video, switching tabs, the chart, the live numbers. |

**The analysis pipeline (used by both the app and training)**

| File | What it does |
|---|---|
| `joint_info.py` | The list of the 33 body points and which column each one lives in. Everything else looks up columns through this. |
| `extraction.py` | Video in, body-point coordinates out. This is the slow part. |
| `split_strokes.py` | Chops the coordinates into individual strokes by finding the peaks of the wrist-minus-ankle wave. |
| `feature_extraction.py` | Turns one stroke's worth of coordinates into the six numbers in the table above. |

**Training and testing (not needed to run the app)**

| File | What it does |
|---|---|
| `model_compare.py` | Runs about ten different ML models on the data and prints how accurate each one was. |
| `hyperparam_tune.py` | A little menu program for tweaking one setting on one model and seeing if it helps. |
| `export_model.py` | Trains the final model and saves it to `artifacts/model.joblib`. |
| `plot_cycle_signals.py` | Debugging tool — draws the wave used for stroke splitting so you can see if it's cutting in the right places. |

**Data and setup files**

| File | What it does |
|---|---|
| `pose_landmarker_lite.task` | Google's pre-trained pose model file. I didn't make this, I just use it. |
| `artifacts/model.joblib` | My trained model, saved to a file so the app can just load it. |
| `artifacts/model_meta.json` | A tiny note-to-self about what's in that model file. |
| `all_videos_all_joints.csv` | Every body point from every frame of every training video. |
| `cycle_data.csv` | The same thing, but chopped into strokes. This is what training reads. |
| `SampleVideos/` | My training videos, with the coach's grade in each filename. |
| `uploads/` | Where an uploaded video sits for the few seconds it's being analyzed. |
| `requirements.txt` | The list of Python libraries to install. |
| `Dockerfile` | Instructions for building the container so it runs the same on a server. |
| `.dockerignore` | Stuff to leave out of the container (videos, training scripts — it doesn't need them). |
| `.gitignore` | Stuff not to put on GitHub (big video files, the giant CSVs). |
| `README.md` | This file. |
| `details_technical.md` | The longer explanation of how everything works. |

## Things I know could be better

- The model was trained on 175 strokes from 15 videos. That's not a lot. It's
  enough to tell a rough stroke from a clean one, but I wouldn't trust it to tell
  an 82 from an 85.
- Every stroke in a video gets the same grade during training, because the coach
  graded the whole video, not each stroke. So the model is really learning "what
  does a video graded 85 look like," and then applying that to single strokes.
- It only works on side-view video. If the camera is in front of you or behind
  you, the angles are meaningless.
- The pose model sometimes loses track of the far arm and leg, which is why I only
  ever measure the side facing the camera.
