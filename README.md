# UGV Autonomous Navigation — PS 26126

Vision-based autonomous navigation software prototype for an outdoor Unmanned
Ground Vehicle. Built for **Smart India Hackathon 2026**, Problem Statement
**26126** — *"Vision Based Autonomous Navigation for Unmanned Ground Vehicle
for Outdoor Environment"* (Bharat Electronics Limited, Category: Software).

Core requirements addressed: path/hazard detection from camera, GPS-denied
visual localization, dynamic collision avoidance, and reliable Point A → Point
B navigation in a simulated outdoor scenario — as a laptop-runnable software
prototype, not a production robotics stack (see **Scope & Limitations** below
for exactly what that means and why).

---

## Table of contents

- [Architecture](#architecture)
- [Data flow, end to end](#data-flow-end-to-end)
- [Install & run](#install--run)
- [Demo script](#demo-script)
- [Scope & limitations](#scope--limitations)
- [Future Industrial Version](#future-industrial-version)
- [Project layout](#project-layout)
- [Testing](#testing)

---

## Architecture

Single-process **FastAPI** backend (Python) driving a simulated vehicle and
world, with a **React** dashboard as the operator view. Internal data
channels are *named* after their eventual ROS 2 topic equivalents
(`slam/pose`, `navigation/risk_map`, `cmd_vel`, …) purely as a documentation
convention — there is no pub/sub runtime or ROS 2 dependency here (see
[Future Industrial Version](#future-industrial-version)).

```
Camera/Video Input  (webcam | uploaded video | synthetic demo feed)
        │
Perception   ── classical CV (MOG2 motion) fallback, or pretrained YOLOv8n
        │
Depth        ── ground-plane relative-depth heuristic
        │        + class-agnostic ground-plane-DEVIATION detector
        │          (catches static hazards neither perception path can see)
        │
Visual Localization ── ORB features + essential matrix (relative pose only)
        │
Terrain Risk Field  ── risk = w1·terrain + w2·obstacle + w3·depth + w4·localization_uncertainty
        │               (all weights configurable, not hardcoded)
        │
Safe Corridor Generator ── connected low-risk region reachable from the vehicle
        │
Global Planner (A*)  ── distance_cost + risk_cost + turning_cost (configurable)
        │
Local Planner  ── re-evaluated every cycle; invalidates blocked path segments
        │          and replans in real time
        │
Decision Engine ── SAFE/WARNING/DEGRADED/CRITICAL → RUN/SLOW/REPLAN/PAUSE
        │           (deterministic; consumes numbers only, calls no model)
        │           + independent raw-distance emergency backstop
        │
Vehicle Simulator ── unicycle/differential-drive kinematics, no real motor output
        │
        └──────────────► feeds back into Camera/Video Input (next frame)
```

Every perception/depth/localization module runs in two modes behind one
identical interface:

- **REAL MODE** — an actual algorithm (YOLOv8n, ORB feature tracking) runs
  on real input.
- **DEMO MODE** — synthetic/configurable data drives the exact same
  downstream interface, so the full pipeline runs identically end-to-end
  with no model loaded and no camera connected. This is what makes the
  live demo reliable regardless of the room's hardware.

## Data flow, end to end

1. **Camera** (`app/modules/camera/`) produces a BGR frame — from a webcam,
   an uploaded video file, or `DemoFrameSource`'s synthetic outdoor scene
   (sky/ground gradient, a drivable path corridor, colored hazard blobs, and
   small fixed "landmark" texture points so visual localization has genuine
   parallax to track).
2. **Perception** (`app/modules/perception/`) turns that frame into a
   `PerceptionResult`: obstacle boxes + confidence, a free-space mask, an
   annotated frame — via either `ClassicalPerception` (MOG2 background
   subtraction, zero downloads) or `YoloPerception` (pretrained YOLOv8n).
3. **Depth** (`app/modules/depth/`) adds a relative (never metric) depth map
   from a flat-ground-plane row→distance heuristic, plus **ground-plane
   deviation regions** — a single-frame, class-agnostic, motion-independent
   appearance-anomaly detector that catches un-classified static hazards
   (rocks, ditches, tree trunks) that neither MOG2 nor YOLO's COCO classes
   can see.
4. **Localization** (`app/modules/localization/`) runs ORB + BFMatcher +
   `cv2.findEssentialMat`/`recoverPose` frame-to-frame, producing a
   **relative pose** (never GPS-accurate, always reset-to-origin) and a
   confidence from tracked-feature/inlier statistics.
5. **Risk field** (`app/modules/risk/`) projects perception/depth detections
   from image space into a world-frame grid (via `app/modules/geometry.py`'s
   pinhole + ground-plane projection) and combines four weighted cost
   layers into `risk`.
6. **Corridor + planners** (`app/modules/corridor/`, `app/modules/planning/`)
   find the reachable low-risk region, A*-plan through it, and steer with a
   pure-pursuit local planner that replans immediately when the live risk
   grid invalidates the active path.
7. **Decision engine** (`app/modules/decision/`) gates the planner's
   proposed velocity through SAFE/WARNING/DEGRADED/CRITICAL →
   RUN/SLOW/REPLAN/PAUSE, using only numbers already computed above — plus
   an independent raw-distance emergency check (see
   [Scope & limitations](#scope--limitations)).
8. **Simulator** (`app/modules/simulator/`) applies the gated velocity to a
   unicycle kinematic model — no real motor output, ever.
9. **Telemetry** (`app/modules/telemetry/`, `app/api/telemetry.py`)
   broadcasts the full cycle over WebSocket at ~10Hz and logs every
   state-changing event (`PATH_REPLANNED`, `DEGRADED_ENTERED`,
   `EMERGENCY_STOP`, `GOAL_REACHED`, …) to SQLite with timestamp, severity,
   position, risk, and confidence.
10. The next simulated frame reflects the vehicle's new pose — closing the
    loop back to step 1.

## Install & run

Requires Python 3.11+ (this repo was developed on a **3.11 venv** —
newer Python versions may lack prebuilt wheels for OpenCV/PyTorch, see
`backend/requirements.txt`) and Node 18+.

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # core: FastAPI, OpenCV, NumPy, SQLAlchemy — fast, no torch

# optional: real YOLOv8n perception path (downloads yolov8n.pt on first use)
pip install -r requirements-real.txt

uvicorn app.main:app --reload --port 8000
```

Health check: `curl http://127.0.0.1:8000/health`. Interactive API docs at
`http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5183` (the dev port is pinned in `vite.config.ts`;
the backend's CORS allowlist in `backend/app/config.py` already includes
it). The dashboard needs the backend running first.

### Running the tests

```bash
cd backend
source .venv/bin/activate
pytest -q                 # ~130 tests, ~10s (one end-to-end demo test runs
                           # the real ~90s scripted scenario — see below)
```

## Demo script

1. Open the dashboard, confirm the header shows **backend connected**.
2. Click **Start SIH Demo**. This runs a deterministic, repeatable scripted
   scenario (`app/modules/demo/scenario.py`) — milestones trigger on the
   vehicle's *progress toward the goal*, never on wall-clock time, so the
   sequence is the same every run regardless of exact pipeline timing:
   1. **Point A → navigating.** The vehicle starts autonomous navigation
      toward Point B, camera/perception/depth/localization/risk/planning
      all live on the dashboard.
   2. **~25% progress: an obstacle appears** directly on the route. Watch
      the event log for detection and the map for the replanned path
      routing around it.
   3. **~45% progress: simulated confidence degradation.** A fault is
      injected (the same mechanism as the manual **Simulate Low
      Confidence** button) — watch the status badge enter
      DEGRADED/CRITICAL and then recover in *stages*, never jumping
      straight back to SAFE (a deliberate safety property, see below).
   4. **~70% progress: the emergency backstop fires.** A hazard is placed
      inside the raw-distance emergency clearance radius — watch the
      system PAUSE via the independent proximity check, then resume once
      the hazard clears.
   5. **Point B reached → SUCCESS**, shown on the status panel.
3. To narrate the safety rules directly: click **Add Obstacle** repeatedly
   to wall the vehicle in completely, and watch the decision engine report
   `NO SAFE PATH` and refuse to move rather than guessing.

Manual controls (Start/Pause/Stop/Reset, Add Obstacle, Simulate Low
Confidence, Enable/Disable Autonomous Mode) are always available alongside
the scripted demo for ad-hoc exploration.

## Scope & limitations

This is a laptop-runnable **software** prototype, verified end-to-end with
both unit tests and live, driven browser/API sessions — not a
production robotics stack. Deliberate, disclosed simplifications:

- **Relative pose and relative depth only.** Visual localization output is
  explicitly labeled "relative pose" everywhere it surfaces and is never
  presented as GPS-accurate; depth is a geometric heuristic, never metric.
- **No real motor output.** The vehicle is a simulated unicycle kinematic
  model end-to-end.
- **Demo camera scenery is only partially pose-consistent.** `DemoFrameSource`
  projects obstacles and localization landmarks from the vehicle's true pose
  (so risk/planning behave meaningfully), but the sky/ground gradient itself
  stays static — no full 3D scene renderer (that's real-time-graphics scope
  a hackathon MVP doesn't need to pay for).
- **Ground-plane deviation detector has a disclosed close-range blind spot.**
  It flags hazards via a residual against a locally-blurred "expected ground
  appearance" — for a large or very close object, the blur kernel partly
  averages into the object's own interior, weakening detection right when
  a vehicle is nearest to it. Two independent mitigations exist:
  1. **Risk-grid memory.** A hazard confirmed repeatedly from a safe
     distance (3–8m in testing) keeps elevated risk in the grid via decay,
     not instant reset — so the avoidance decision is usually already made
     before the vehicle gets close enough to lose the detection.
  2. **The raw-distance emergency backstop** (phase 1k, `DecisionEngine.
     check_emergency()`) is a deliberately *dumb*, always-on check that
     reads a raw clearance value directly — in this simulation,
     `Environment.nearest_obstacle_distance()`, standing in for a real
     UGV's ultrasonic/IR/bump sensor — and forces CRITICAL/PAUSE the
     instant anything is too close, **regardless of what perception, depth,
     or the risk field classified it as.** It shares no code path with the
     vision stack, so it cannot share the vision stack's blind spot. This
     is unit-tested explicitly: `test_emergency_fires_even_when_risk_
     field_says_everything_is_fine`.
- **Monocular VO has a real cold-start/noise sensitivity.** Two engineering
  fixes were needed and are documented in code where they live
  (`app/modules/pipeline/orchestrator.py`): confidence is EMA-smoothed
  before reaching the safety gate (single noisy frames shouldn't flip
  system state), and a stationary vehicle's confidence is floored rather
  than allowed to decay to zero (dead-reckoning error doesn't grow while
  genuinely stationary — without this, PAUSE-from-low-confidence and
  low-confidence-from-PAUSE form a real deadlock).
- **Classical fallback perception (MOG2) only sees motion**; YOLOv8n only
  sees its pretrained COCO classes (no rock/ditch/tree categories). This is
  exactly why the ground-plane deviation detector exists as the *primary*
  signal for terrain/obstacle risk, with MOG2/YOLO layered on top as
  *supplementary* signals — see `app/modules/risk/builder.py`.

## Future Industrial Version

Explicitly out of scope for this MVP, and why:

| Cut | One-line reason |
|---|---|
| **Training/fine-tuning any model** (e.g. SegFormer on RELLIS-3D/RUGD) | No dataset pipeline in scope; pretrained YOLOv8n is used as-is. A real deployment would fine-tune on outdoor-hazard classes RELLIS-3D/RUGD actually provide. |
| **ORB-SLAM3 / g2o / DBoW2** | A full C++ SLAM stack (global map, loop closure, bundle adjustment) is real-robotics-scope; this MVP intentionally implements only the frame-to-frame tracking *front-end* (ORB + essential matrix) in Python/OpenCV. |
| **ROS 2 / colcon / multi-node pub/sub** | A single FastAPI process with internal modules (named after their ROS 2 topic equivalents) keeps the timeline sane; porting to real ROS 2 nodes later is a wiring change, not a redesign. |
| **TensorRT / INT8 quantization** | An inference-speed optimization for real embedded hardware; irrelevant to a laptop-runnable software demo. |
| **True metric localization/depth scale** | Monocular vision alone cannot recover real-world scale without another sensor (stereo, IMU, wheel odometry) — out of scope; would be added for the industrial version. |
| **Real motor control / actuator safety certification** | Everything vehicle-side is simulated; a real UGV integration needs its own certified safety layer. |

## Project layout

```
ugv/
├── backend/
│   ├── app/
│   │   ├── main.py, config.py
│   │   ├── api/            # FastAPI routers: simulator, pipeline, telemetry, demo, health
│   │   ├── db/              # SQLAlchemy models + session (event log)
│   │   └── modules/
│   │       ├── simulator/   # vehicle kinematics, environment, top-down renderer
│   │       ├── camera/      # CameraSource: webcam | video file | synthetic demo
│   │       ├── perception/  # classical (MOG2) + real (YOLOv8n) — shared interface
│   │       ├── depth/       # relative depth + ground-plane deviation detector
│   │       ├── localization/# ORB + essential matrix visual odometry
│   │       ├── risk/        # world-frame risk grid + weighted cost builder
│   │       ├── corridor/    # safe corridor / reachability generator
│   │       ├── planning/    # A* global planner + local planner
│   │       ├── decision/    # deterministic decision engine (phase 1k)
│   │       ├── telemetry/   # SQLite event logger
│   │       ├── demo/        # scripted SIH demo scenario
│   │       ├── pipeline/    # orchestrator tying every module into one cycle
│   │       └── geometry.py  # shared camera <-> world projection math
│   ├── tests/                # ~130 tests across every module above
│   ├── requirements.txt      # core (fast, no torch)
│   └── requirements-real.txt # optional: YOLOv8n real perception path
└── frontend/
    └── src/
        ├── api.ts, types.ts, hooks/useTelemetry.ts
        └── components/        # CameraPanel, MapPanel, StatusPanel, ControlsPanel,
                                # EventLogPanel, DemoPanel
```

## Testing

Each module was built and verified independently before integration:
kinematics/geometry with closed-form unit tests, localization against
synthetic two-view scenes with known ground-truth motion, perception/depth
against both synthetic and real images, and the full pipeline against live,
driven simulation runs (including the ~90-second real scripted demo run as
an actual `pytest` test, not just a manual check). See each module's
docstring for what it does and does not claim, and `tests/` for the
executable proof.
