import cv2
import requests
import base64
import numpy as np
import time
import os

SERVER_URL = "http://127.0.0.1:8000/detect"
VIDEO_PATH = "test_video2.mp4"
ID_TOKEN = os.environ.get("ID_TOKEN", "").strip()
TRIP_ID = os.environ.get("TRIP_ID", "").strip()

if not ID_TOKEN:
    raise SystemExit("ID_TOKEN is not set. Export a fresh Firebase ID token before running this script.")

if not TRIP_ID:
    raise SystemExit("TRIP_ID is not set. Start a trip first, then export TRIP_ID before running this script.")

HEADERS = {
    "Authorization": f"Bearer {ID_TOKEN}",
    "X-Trip-Id": TRIP_ID,
}

SEND_FPS = 30          # try 4 if still slow
JPEG_QUALITY = 55     # 50-65 recommended
SIZE = (640, 360)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video")
    exit()

frame_interval = 1.0 / SEND_FPS
next_time = time.time()

while True:
    # Keep wall-clock pace
    now = time.time()
    if now < next_time:
        time.sleep(next_time - now)
    next_time += frame_interval

    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, SIZE)
    ok, img_encoded = cv2.imencode(
        ".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        continue

    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    start = time.time()
    try:
        print("AUTH HEADER SENT:", HEADERS.get("Authorization"))
        resp = requests.post(
            SERVER_URL,
            files=files,
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 401:
            print("Unauthorized (401): Firebase ID token is invalid/expired. Generate a new idToken and update ID_TOKEN.")
            break
        if resp.status_code >= 400:
            print(f"Server error {resp.status_code}:", resp.text)
        resp.raise_for_status()
        data = resp.json()

        overlay_bytes = base64.b64decode(data["overlay"])
        np_arr = np.frombuffer(overlay_bytes, np.uint8)
        overlay_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if overlay_img is None:
            continue

        latency_ms = (time.time() - start) * 1000
        cv2.putText(overlay_img, f"Latency: {latency_ms:.0f} ms",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Server Output", overlay_img)

    except Exception as e:
        print("Error:", e)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
