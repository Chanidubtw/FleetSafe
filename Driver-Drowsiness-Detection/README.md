![Driver Drowsiness Detection]
<h1 align="center">Driver Drowsiness Detector 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
    <img alt="Twitter: NeelanjanManna" src="https://img.shields.io/twitter/follow/NeelanjanManna.svg?style=social" />
  </a>
</p>

> A real-time drowsiness detection system for drivers, which alerts the driver if they fall asleep due to fatigue while still driving. The computer vision algorithm used for the implementation uses a trifold approach to detect drowsiness, including the measurement of forward head tilt angle, measurement of eye aspect ratio (to detect closure of eyes) and measurement of mouth aspect ratio (to detect yawning).

### 🏠 [Homepage](https://github.com/neelanjan00/Driver-Drowsiness-Detection)

## Install

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r Requirements.txt
```

## Usage

```sh
python "Driver Drowsiness Detection.py"
```

### Useful Options

```sh
# Auto camera detect (default), show FPS
python "Driver Drowsiness Detection.py" --show-fps

# Pick a specific camera index
python "Driver Drowsiness Detection.py" --camera-index 0

# Tune alert sensitivity
python "Driver Drowsiness Detection.py" --eye-ar-thresh 0.24 --eye-drowsy-frames 50

# Disable sound (visual alert only)
python "Driver Drowsiness Detection.py" --no-sound
```

Press `q` to exit the app window.

## Notes

- Ensure webcam permission is granted to your terminal/IDE.
- Landmark model is expected at `dlib_shape_predictor/shape_predictor_68_face_landmarks.dat`.
- On macOS, alarm audio uses `afplay`; on other platforms, a fallback alarm is used.


## Show your support

Give a ⭐️ if this project helped you!
