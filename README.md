# FleetSafe — AI-Enabled Public Transport Safety Monitoring System

A smartphone-based safety monitoring system for public transport buses. FleetSafe detects
illegal double-solid lane crossings in real time, captures video evidence, verifies
violations using Google Gemini AI, and monitors the driver for drowsiness.

**PUSL3190 Computing Project — Final Year Submission**
**Student:** Madarasinghe M A C H | **Index:** 10952717 | **Degree:** BSc (Hons) Software Engineering
**Supervisor:** Dr. Rasika Ranaweera | **University of Plymouth / NSBM Green University**

---

## System Overview

```
┌─────────────────────┐       JPEG frames         ┌────────────────────────────┐
│  Flutter Mobile App │  ────────────────────────► │   FastAPI Backend          │
│  (Android)          │                            │   Lane Detection Pipeline  │
│                     │  ◄────────────────────────  │   Violation Logic          │
│  • Driver login     │      overlay + labels       │   Evidence Capture         │
│  • Trip management  │                            │   Gemini AI Verification   │
│  • Live monitoring  │                            └──────────────┬─────────────┘
│  • Violation history│                                           │
└─────────────────────┘                             ┌────────────▼──────────────┐
                                                    │        Firebase            │
┌─────────────────────┐                            │  Authentication            │
│  Driver Drowsiness  │                            │  Firestore (violations)    │
│  Detection Module   │                            │  Cloud Storage (clips)     │
│  (standalone)       │                            └──────────────┬─────────────┘
│  • EAR / MAR        │                                           │
│  • Real-time alarm  │                             ┌────────────▼──────────────┐
└─────────────────────┘                            │   Gemini 2.5 Flash AI      │
                                                    │   Violation Verification   │
                                                    └───────────────────────────┘
```

---

## Repository Structure

```
FleetSafe-main/                    ← Python backend (this repo)
├── server.py                      ← FastAPI server — main entry point (918 lines)
├── LaneLines.py                   ← Lane detection + marking classification (774 lines)
├── Thresholding.py                ← CLAHE + HLS/HSV/Sobel binary pipeline
├── PerspectiveTransformation.py   ← Bird's-eye view homography transform
├── CameraCalibration.py           ← Chessboard lens distortion correction
├── frame_buffer.py                ← Circular pre-event buffer for evidence clips
├── firebase_upload.py             ← Firebase Storage upload + Firestore documents
├── firebase_setup.py              ← Firebase Admin SDK initialisation
├── gemini_verify.py               ← Gemini 2.5 Flash AI verification module
├── Experiment.ipynb               ← Thresholding parameter tuning notebook
├── camera_cal/                    ← Chessboard calibration images (22 images)
├── test_images/                   ← Test road frames for pipeline validation
├── clips/                         ← Saved evidence clips (auto-created, gitignored)
├── Driver-Drowsiness-Detection/   ← Standalone drowsiness detection module
│   └── Driver Drowsiness Detection.py
└── figure_scripts/                ← Scripts used to generate report figures
```

The **Flutter mobile app** is in a separate folder:

```
FleetSafe_app-main/
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── live_detection_screen.dart
│   │   ├── violations_screen.dart
│   │   ├── violation_detail.dart
│   │   └── driver_registration_screen.dart
│   ├── services/trip_service.dart
│   └── config/api_config.dart
└── assets/videos/test_drive.mp4   ← Test video used for frame streaming demo
```

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Backend | Python 3.9+ |
| Mobile app | Flutter SDK 3.19+, Android Studio |
| Device | Android 9.0+ device or emulator |
| Firebase | Project with Authentication, Firestore, and Storage enabled |
| AI | Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com)) |
| Drowsiness module | Python 3.9+, webcam, dlib shape predictor file |

---

## Setup and Installation

### Step 1 — Backend Server

**Install dependencies:**
```bash
cd FleetSafe-main
pip install fastapi uvicorn opencv-python numpy firebase-admin google-generativeai moviepy
```

**Add Firebase credentials:**
1. Open [Firebase Console](https://console.firebase.google.com) → Project Settings → Service Accounts
2. Click **Generate new private key** and download the JSON file
3. Rename it `firebase_key.json` and place it inside `FleetSafe-main/`

> `firebase_key.json` is in `.gitignore` and will never be committed to git.

**Set your Gemini API key:**
```bash
# macOS / Linux
export GEMINI_API_KEY=your_api_key_here

# Windows
set GEMINI_API_KEY=your_api_key_here
```

**Start the server:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required so Android devices and emulators can connect.

Confirm it is running — open `http://localhost:8000/docs` in your browser.
You should see the FastAPI interactive documentation page.

---

### Step 2 — Flutter Mobile App

**Install packages:**
```bash
cd FleetSafe_app-main
flutter pub get
```

**Connect the app to the server:**

The correct URL depends on how you run the app:

| How you run the app | URL to use |
|---------------------|-----------|
| Android emulator | `http://10.0.2.2:8000` (default, no change needed) |
| Physical device via USB | Use `adb reverse` (see below) |
| Physical device on Wi-Fi | Use your computer's LAN IP |

**For a physical device via USB (recommended):**
```bash
adb reverse tcp:8000 tcp:8000
flutter run
```

**For a physical device on Wi-Fi:**
```bash
# Find your computer's LAN IP
ifconfig | grep "inet " | grep -v 127.0.0.1   # macOS/Linux
ipconfig                                         # Windows

# Run with that IP
flutter run --dart-define=API_BASE_URL=http://YOUR_LAN_IP:8000
```

---

### Step 3 — Driver Drowsiness Detection Module

**Install dependencies:**
```bash
cd FleetSafe-main/Driver-Drowsiness-Detection
pip install opencv-python dlib imutils
```

**Download the dlib shape predictor:**
1. Download from [dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Extract:
   ```bash
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```
3. Move into the predictor folder:
   ```bash
   mkdir -p dlib_shape_predictor
   mv shape_predictor_68_face_landmarks.dat dlib_shape_predictor/
   ```

**Run:**
```bash
python "Driver Drowsiness Detection.py"
```

Press **Q** to quit.

**Optional flags:**
```bash
--no-sound                    # Disable audio alarm
--eye-ar-thresh 0.22          # Change EAR threshold (default: 0.25)
--eye-drowsy-frames 48        # Change frame count for alarm (default: 60, ~3 s at 20 FPS)
--camera-index 0              # Specify camera index if auto-detect fails
```

---

## How to Run a Demo (End-to-End)

1. Start the backend: `uvicorn server:app --host 0.0.0.0 --port 8000`
2. Start the Flutter app on your device
3. Register or log in with a Firebase email/password account
4. Tap **Start Monitoring** — the app creates a trip and begins streaming
5. The lane overlay appears in real time with LEFT/RIGHT marking labels
6. When a double-solid crossing is detected the system automatically:
   - Records an 8-second evidence clip (with 2-second pre-event buffer)
   - Uploads the clip to Firebase Storage
   - Sends the clip to Gemini AI for verification
   - Updates Firestore with the AI decision
7. Tap **Stop Trip** to end the session
8. Open **My Violations** to view the evidence clip, AI decision, and written explanation
9. Tap any violation to submit an appeal

To demo drowsiness detection, run the standalone module (Step 3) on a laptop with a forward-facing camera.

---

## API Endpoints

All endpoints except `GET /` require `Authorization: Bearer <firebase_id_token>`.

### Driver endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/me` | Get driver profile |
| POST | `/trip/start` | Start a trip session |
| POST | `/trip/end` | End the active trip |
| GET | `/trips/mine` | Get all trips for this driver |
| POST | `/detect` | Process a JPEG frame — returns overlay + lane state |
| GET | `/violations/mine` | Get all violations for this driver |
| POST | `/violations/{id}/appeal` | Submit an appeal |
| GET | `/me/stats` | Safety score + violation counts |

### Admin endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/init` | Bootstrap first admin account |
| GET | `/admin/stats` | System-wide statistics |
| GET | `/admin/drivers` | All drivers with violation counts |
| GET | `/admin/violations` | All violations (filterable by driver, status, date) |
| PATCH | `/admin/violations/{id}/appeal` | Approve or reject a driver appeal |
| PATCH | `/admin/drivers/{uid}/status` | Suspend or activate a driver |

Full documentation with request/response schemas: `http://localhost:8000/docs`

---

## How the Violation Pipeline Works

```
Frame received
      │
      ▼
Stage 1: Camera undistortion (chessboard calibration)
      │
      ▼
Stage 2: Perspective transform → bird's-eye view
      │
      ▼
Stage 3: Binary thresholding (CLAHE + HLS white + HSV yellow + Sobel edges)
      │
      ▼
Stage 4: Sliding window search → polynomial fitting (x = ay² + by + c)
      │
      ▼
Stage 5: Marking classification (connected-component analysis)
         dashed / solid / double_solid — 4-second temporal lock at ≥ 75% confidence
      │
      ▼
Stage 6: Violation check
         vehicle centre crosses double_solid boundary by > 25 px
         AND condition persists ≥ 0.5 s
               │
               ▼
         Violation emitted → FrameBuffer saves 8 s clip (background thread)
               │
               ▼
         Upload to Firebase Storage
               │
               ▼
         Create Firestore document  {status: "pending"}
               │
               ▼
         Gemini 2.5 Flash verifies clip
               │
               ▼
         Update Firestore  {status: "violation" / "not_violation" / "uncertain",
                            aiResult: {decision, confidence, reason}}
```

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| EAR drowsiness threshold | 0.25 | `Driver Drowsiness Detection.py` |
| Drowsiness alarm frame count | 60 (~3 s at 20 FPS) | `Driver Drowsiness Detection.py` |
| MAR yawning threshold | 0.79 | `Driver Drowsiness Detection.py` |
| Violation confirmation window | 0.5 seconds | `LaneLines.py` |
| Violation cooldown | 2 seconds | `LaneLines.py` |
| Temporal classification lock | 4 seconds at ≥ 75% confidence | `LaneLines.py` |
| Crossing margin | 25 pixels | `LaneLines.py` |
| Pre-event buffer | 40 frames (~2 s at 20 FPS) | `frame_buffer.py` |
| Firebase token cache TTL | 300 seconds | `server.py` |
| Safety score window | 30 days | `server.py` |

---

## Known Limitations

- The Flutter app streams frames from a pre-recorded test video (`test_drive.mp4`) rather than a live camera feed. Live camera integration is the immediate next development step.
- The admin dashboard has a complete backend API but no web frontend. All admin operations are available via the REST API.
- GPS-based speed monitoring is not yet implemented. The violation event schema includes location fields for future integration.
- NFC-based driver card authentication is planned as a future enhancement.
- Lane classification accuracy is approximately 84% overall and 95% on clear road markings. Accuracy reduces under heavy shadow or severely faded markings.

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Backend language | Python 3.9+ |
| API framework | FastAPI + Uvicorn |
| Computer vision | OpenCV 4.8+, NumPy |
| AI verification | Google Gemini 2.5 Flash |
| Facial landmarks | dlib 19.x, imutils |
| Mobile app | Flutter 3.19+ (Dart) |
| Authentication | Firebase Authentication |
| Database | Firebase Firestore |
| Video storage | Firebase Cloud Storage |
| Version control | Git + GitHub |

---

## Security

- `firebase_key.json` is in `.gitignore` and is never committed
- The Gemini API key is read from an environment variable only
- All protected endpoints require Firebase ID token verification
- Firestore security rules enforce driver data isolation (drivers see only their own records)
- Evidence clip URLs use signed expiry (7 days)

---

## Acknowledgements

The initial lane detection pipeline structure (perspective transform, thresholding, sliding window, polynomial fitting) draws on concepts from the Udacity Self-Driving Car Nanodegree Advanced Lane Finding project. All lane marking classification, violation logic, evidence capture, AI verification, mobile application, and cloud integration code was developed from scratch for this project.
