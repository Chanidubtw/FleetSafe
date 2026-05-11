# server.py
import base64
import os
import threading
import time
from typing import Dict, Tuple, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from firebase_admin import auth, firestore
from firebase_setup import init_firebase
from firebase_upload import upload_violation_clip, update_violation_ai_result
from gemini_verify import verify_video

from frame_buffer import FrameBuffer
from main import FindLaneLines

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# Firebase clients (init once)
# ------------------------------------------
db = None
bucket = None
try:
    db, bucket = init_firebase()
except Exception as e:
    print("Firebase init warning:", e)

# ------------------------------------------
# Lane system + buffer
# ------------------------------------------
lane_system = FindLaneLines()

CLIP_SECONDS = 8
CLIP_PRE_SECONDS = 2
frame_buffer = FrameBuffer(fps=20, seconds=CLIP_SECONDS, pre_seconds=CLIP_PRE_SECONDS)

CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

# ------------------------------------------
# Clip saving / latch
# ------------------------------------------
saving_clip_lock = threading.Lock()
saving_clip = False
last_violation_time = 0.0
VIOLATION_SAVE_COOLDOWN = 3.0  # seconds

violation_armed = True
no_violation_streak = 0
NO_VIOLATION_REARM_FRAMES = 20

last_frame_time = 0.0
STREAM_RESET_GAP_SECONDS = 1.5

# ------------------------------------------
# Token cache to reduce verify calls
# token -> (uid, exp_epoch)
# ------------------------------------------
token_cache_lock = threading.Lock()
token_cache: Dict[str, Tuple[str, int]] = {}

# ------------------------------------------
# Trip validation cache to avoid Firestore read every frame
# (uid, tripId) -> valid_until_epoch
# ------------------------------------------
trip_cache_lock = threading.Lock()
trip_cache: Dict[Tuple[str, str], int] = {}
TRIP_CACHE_TTL = 20  # seconds


def _unauthorized():
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


def _server_auth_error(message: str):
    return JSONResponse({"error": message}, status_code=500)


def require_user(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return _unauthorized()

    token = auth_header.replace("Bearer ", "", 1).strip()
    now_epoch = int(time.time())

    if not db:
        return _server_auth_error("Firebase Admin is not initialized on the server.")

    with token_cache_lock:
        cached = token_cache.get(token)
        if cached and cached[1] > now_epoch:
            return {"uid": cached[0]}

    try:
        decoded = auth.verify_id_token(token)
        uid = decoded["uid"]
        exp = int(decoded.get("exp", now_epoch + 300))

        with token_cache_lock:
            token_cache[token] = (uid, exp)

        return decoded
    except Exception as e:
        print("Auth verify failed:", e)
        return _unauthorized()


def _require_trip_for_user(uid: str, trip_id: str) -> Optional[JSONResponse]:
    """
    Validates that trip exists, belongs to uid, and is active.
    Uses short TTL cache to avoid constant reads during streaming.
    """
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    now_epoch = int(time.time())
    key = (uid, trip_id)

    with trip_cache_lock:
        exp = trip_cache.get(key)
        if exp and exp > now_epoch:
            return None

    try:
        ref = db.collection("trips").document(trip_id)
        snap = ref.get()
        if not snap.exists:
            return JSONResponse({"error": "trip not found"}, status_code=404)

        trip = snap.to_dict() or {}
        if trip.get("driverId") != uid:
            return JSONResponse({"error": "forbidden"}, status_code=403)

        if trip.get("status") != "active":
            return JSONResponse({"error": "trip not active"}, status_code=400)

        with trip_cache_lock:
            trip_cache[key] = now_epoch + TRIP_CACHE_TTL

        return None
    except Exception as e:
        print("Trip validate error:", e)
        return JSONResponse({"error": "trip validation failed"}, status_code=500)


# ------------------------------------------
# Background worker: save -> upload -> verify -> update
# ------------------------------------------
def save_clip_background(buffer_obj: FrameBuffer, recording, filename: str, driver_id: str, trip_id: str, location: Optional[Dict[str, Any]] = None):
    global saving_clip
    doc_id = None

    try:
        out = buffer_obj.save_recording(recording, filename)
        if out is None:
            print("No frames captured for clip.")
            return

        print("Violation clip saved:", out)

        # Upload + create Firestore doc
        try:
            doc_id, _ = upload_violation_clip(out, driver_id=driver_id, trip_id=trip_id, location=location)
            print("Firestore violation created:", doc_id)
        except Exception as e:
            print("Firebase upload/create failed:", e)

        # Gemini verify (async)
        print("Sending clip to Gemini...")
        result = verify_video(out)

        if doc_id:
            try:
                update_violation_ai_result(doc_id, result)
                print("Firestore updated aiResult/status:", doc_id)
            except Exception as e:
                print("Firestore AI update failed:", e)

    except Exception as e:
        print("Error in background pipeline:", e)
        if doc_id:
            fallback = {
                "decision": "Uncertain",
                "confidence": 0.0,
                "reason": f"Gemini/pipeline error: {e}",
            }
            try:
                update_violation_ai_result(doc_id, fallback, failed=True)
            except Exception as update_err:
                print("Failed to mark doc failed:", update_err)

    finally:
        with saving_clip_lock:
            saving_clip = False


# ------------------------------------------
# Basic endpoints
# ------------------------------------------
@app.get("/")
def root():
    return {"status": "lane server running"}


@app.post("/test/force-violation")
def force_violation(user=Depends(require_user)):
    """Dev/test endpoint: instantly fires a violation event on the next /detect frame."""
    if isinstance(user, JSONResponse):
        return user

    now = time.time()
    if hasattr(lane_system, "lanelines"):
        ll = lane_system.lanelines
        ll.violation_event = {
            "type": "TEST_FORCED_VIOLATION",
            "time": now,
        }
        ll.violation_display_until = now + ll.VIOLATION_DISPLAY_SECONDS

    return {"triggered": True, "message": "Violation will appear on next /detect response"}


@app.get("/me")
def me(user=Depends(require_user)):
    if isinstance(user, JSONResponse):
        return user
    return {"uid": user["uid"], "email": user.get("email")}


# ------------------------------------------
# Trips API
# ------------------------------------------
@app.post("/trip/start")
@app.post("/trips/start")
def start_trip(user=Depends(require_user), payload: Optional[Dict[str, Any]] = Body(default=None)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    payload = payload or {}
    uid = user["uid"]
    bus_no = payload.get("busNo")
    route_no = payload.get("routeNo")
    start_location = payload.get("startLocation")

    ref = db.collection("trips").document()
    ref.set({
        "driverId": uid,
        "busNo": bus_no,
        "routeNo": route_no,
        "startTime": firestore.SERVER_TIMESTAMP,
        "endTime": None,
        "status": "active",
        "startLocation": start_location,
        "endLocation": None,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    return {"tripId": ref.id, "status": "active"}


@app.post("/trip/end")
@app.post("/trips/end")
def end_trip(user=Depends(require_user), payload: Optional[Dict[str, Any]] = Body(default=None)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    payload = payload or {}
    uid = user["uid"]
    trip_id = payload.get("tripId")
    end_location = payload.get("endLocation")

    if not trip_id:
        return JSONResponse({"error": "tripId required"}, status_code=400)

    ref = db.collection("trips").document(trip_id)
    snap = ref.get()
    if not snap.exists:
        return JSONResponse({"error": "trip not found"}, status_code=404)

    trip = snap.to_dict() or {}
    if trip.get("driverId") != uid:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    ref.update({
        "endTime": firestore.SERVER_TIMESTAMP,
        "endLocation": end_location,
        "status": "ended",
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    # clear cache so next detect won't accept old trip
    with trip_cache_lock:
        trip_cache.pop((uid, trip_id), None)

    return {"tripId": trip_id, "status": "ended"}


# ------------------------------------------
# Driver stats & safety score
# ------------------------------------------
@app.get("/me/stats")
def me_stats(user=Depends(require_user)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    uid = user["uid"]
    cutoff_epoch = int(time.time()) - 30 * 24 * 3600  # last 30 days

    # Single-field query (no composite index needed).
    # Epoch filtering is done in Python to avoid requiring a Firestore index.
    v_docs = db.collection("violations").where("driverId", "==", uid).stream()

    confirmed = pending_v = cleared = appeals_approved = 0
    for d in v_docs:
        v = d.to_dict() or {}
        if (v.get("createdAtEpoch") or 0) < cutoff_epoch:
            continue
        s = v.get("status", "pending")
        if s == "violation":
            confirmed += 1
        elif s == "not_violation":
            cleared += 1
        else:
            pending_v += 1
        if v.get("appealStatus") == "approved":
            appeals_approved += 1

    # Single-field query for trips (no index needed)
    t_docs = db.collection("trips").where("driverId", "==", uid).stream()
    total_trips = sum(1 for _ in t_docs)

    # --- score formula ---
    # -10 per confirmed violation, -2 per pending (uncertain), +5 per approved appeal
    score = 100 - confirmed * 10 - pending_v * 2 + appeals_approved * 5
    score = max(0, min(100, score))

    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 60:
        grade = "Fair"
    else:
        grade = "Poor"

    return {
        "score": score,
        "grade": grade,
        "windowDays": 30,
        "totalTrips": total_trips,
        "violations": {
            "confirmed": confirmed,
            "pending": pending_v,
            "cleared": cleared,
        },
        "appealsApproved": appeals_approved,
    }


# ------------------------------------------
# Trip history
# ------------------------------------------
@app.get("/trips/mine")
def my_trips(user=Depends(require_user)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    uid = user["uid"]

    # Fetch all trips for this driver (no ordering to avoid index requirement)
    raw = list(db.collection("trips").where("driverId", "==", uid).stream())

    items = []
    for doc in raw:
        d = doc.to_dict() or {}
        d["id"] = doc.id

        # Keep raw timestamp for sorting before converting
        raw_start = d.get("startTime")
        sort_key = raw_start.timestamp() if raw_start and hasattr(raw_start, "timestamp") else 0

        for key in ["startTime", "endTime", "updatedAt"]:
            val = d.get(key)
            if val is not None:
                try:
                    d[key] = val.isoformat()
                except Exception:
                    d[key] = str(val)

        d["_sort"] = sort_key
        items.append(d)

    items.sort(key=lambda x: x.pop("_sort", 0), reverse=True)
    items = items[:50]

    # Attach per-trip violation counts in one query
    v_docs = db.collection("violations").where("driverId", "==", uid).stream()
    counts: Dict[str, Dict] = {}
    for vd in v_docs:
        v = vd.to_dict() or {}
        tid = v.get("tripId")
        if not tid:
            continue
        if tid not in counts:
            counts[tid] = {"total": 0, "confirmed": 0}
        counts[tid]["total"] += 1
        if v.get("status") == "violation":
            counts[tid]["confirmed"] += 1

    for item in items:
        c = counts.get(item["id"], {})
        item["violationCount"] = c.get("total", 0)
        item["confirmedViolations"] = c.get("confirmed", 0)

    return {"count": len(items), "items": items}


# ------------------------------------------
# Violations list (already useful for app)
# ------------------------------------------
@app.get("/violations/mine")
def my_violations(user=Depends(require_user)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    uid = user["uid"]
    docs = (
        db.collection("violations")
        .where("driverId", "==", uid)
        .order_by("createdAtEpoch", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )

    items = []
    for d in docs:
        data = d.to_dict() or {}
        data["id"] = d.id
        # timestamp is not JSON serializable sometimes
        if "timestamp" in data and data["timestamp"] is not None:
            data["timestamp"] = data["timestamp"].isoformat()
        if "updatedAt" in data and data["updatedAt"] is not None:
            data["updatedAt"] = data["updatedAt"].isoformat()
        items.append(data)

    return {"count": len(items), "items": items}


# ------------------------------------------
# Appeals
# ------------------------------------------
@app.post("/violations/{violation_id}/appeal")
def appeal_violation(
    violation_id: str,
    user=Depends(require_user),
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    uid = user["uid"]
    reason = ((payload or {}).get("reason") or "").strip()
    if not reason:
        return JSONResponse({"error": "reason is required"}, status_code=400)

    ref = db.collection("violations").document(violation_id)
    snap = ref.get()
    if not snap.exists:
        return JSONResponse({"error": "violation not found"}, status_code=404)

    v = snap.to_dict() or {}
    if v.get("driverId") != uid:
        return JSONResponse({"error": "forbidden"}, status_code=403)

    if v.get("appealStatus") is not None:
        return JSONResponse({"error": "already appealed"}, status_code=409)

    ref.update({
        "appealStatus": "pending",
        "appealReason": reason,
        "appealedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    return {"success": True, "appealStatus": "pending"}


# ------------------------------------------
# Admin helpers
# ------------------------------------------
def _is_admin(uid: str) -> bool:
    if db is None:
        return False
    try:
        return db.collection("admins").document(uid).get().exists
    except Exception:
        return False


def require_admin(user=Depends(require_user)):
    if isinstance(user, JSONResponse):
        return user
    if not _is_admin(user["uid"]):
        return JSONResponse({"error": "Admin access required"}, status_code=403)
    return user


def _serialize(d: dict, keys: list) -> dict:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                d[k] = v.isoformat()
            except Exception:
                d[k] = str(v)
    return d


# ------------------------------------------
# Admin endpoints
# ------------------------------------------
@app.post("/admin/init")
def admin_init(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Bootstrap the first admin. Requires ADMIN_SECRET env var to match."""
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    payload = payload or {}
    secret = (payload.get("secret") or "").strip()
    uid = (payload.get("uid") or "").strip()

    if not secret or not uid:
        return JSONResponse({"error": "secret and uid are required"}, status_code=400)

    expected = os.environ.get("ADMIN_SECRET", "")
    if not expected:
        return JSONResponse({"error": "ADMIN_SECRET env var not configured on server"}, status_code=500)
    if secret != expected:
        return JSONResponse({"error": "Invalid secret"}, status_code=403)

    db.collection("admins").document(uid).set({"createdAt": firestore.SERVER_TIMESTAMP})
    return {"success": True, "uid": uid}


@app.get("/admin/stats")
def admin_stats(user=Depends(require_admin)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    now_epoch = int(time.time())
    # Start of today UTC (midnight)
    import datetime
    today_utc = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_epoch = int(today_utc.timestamp())
    # Start of this week (Monday)
    week_start_utc = today_utc - datetime.timedelta(days=today_utc.weekday())
    week_start_epoch = int(week_start_utc.timestamp())

    # Stream ALL violations and filter in Python (avoids composite index)
    v_today = v_week = v_total = v_confirmed = v_pending_appeal = 0
    for vd in db.collection("violations").stream():
        v = vd.to_dict() or {}
        v_total += 1
        epoch = v.get("createdAtEpoch") or 0
        if epoch >= today_start_epoch:
            v_today += 1
        if epoch >= week_start_epoch:
            v_week += 1
        if v.get("status") == "violation":
            v_confirmed += 1
        if v.get("appealStatus") == "pending":
            v_pending_appeal += 1

    total_drivers = sum(1 for _ in db.collection("drivers").stream())
    active_trips = sum(
        1 for td in db.collection("trips").stream()
        if (td.to_dict() or {}).get("status") == "active"
    )

    return {
        "violations": {
            "today": v_today,
            "thisWeek": v_week,
            "total": v_total,
            "confirmed": v_confirmed,
        },
        "pendingAppeals": v_pending_appeal,
        "totalDrivers": total_drivers,
        "activeTrips": active_trips,
    }


@app.get("/admin/drivers")
def admin_drivers(user=Depends(require_admin)):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    # Build violation count per driver from a single collection stream
    violation_counts: Dict[str, int] = {}
    for vd in db.collection("violations").stream():
        v = vd.to_dict() or {}
        did = v.get("driverId")
        if did:
            violation_counts[did] = violation_counts.get(did, 0) + 1

    items = []
    for doc in db.collection("drivers").stream():
        d = doc.to_dict() or {}
        d["uid"] = doc.id
        _serialize(d, ["createdAt", "updatedAt", "lastLoginAt"])
        d["violationCount"] = violation_counts.get(doc.id, 0)
        items.append(d)

    items.sort(key=lambda x: (x.get("name") or "").lower())
    return {"count": len(items), "items": items}


@app.get("/admin/violations")
def admin_violations(
    user=Depends(require_admin),
    status: Optional[str] = None,
    appeal_status: Optional[str] = None,
    limit: int = 200,
):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    ref = db.collection("violations")
    if status is not None:
        query = ref.where("status", "==", status)
    elif appeal_status is not None:
        query = ref.where("appealStatus", "==", appeal_status)
    else:
        query = ref

    items = []
    for vd in query.stream():
        d = vd.to_dict() or {}
        d["id"] = vd.id
        _serialize(d, ["timestamp", "updatedAt", "appealedAt"])
        items.append(d)

    items.sort(key=lambda x: x.get("createdAtEpoch") or 0, reverse=True)
    return {"count": len(items), "items": items[:limit]}


@app.get("/admin/trips")
def admin_trips(user=Depends(require_admin), status: Optional[str] = None):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    ref = db.collection("trips")
    query = ref.where("status", "==", status) if status is not None else ref

    # Build per-trip violation counts from a single violations stream
    trip_violation_counts: Dict[str, int] = {}
    for vd in db.collection("violations").stream():
        v = vd.to_dict() or {}
        tid = v.get("tripId")
        if tid:
            trip_violation_counts[tid] = trip_violation_counts.get(tid, 0) + 1

    items = []
    for doc in query.stream():
        d = doc.to_dict() or {}
        d["id"] = doc.id

        # Capture raw startTime for sorting before serialising
        raw_start = d.get("startTime")
        d["_sort"] = raw_start.timestamp() if raw_start and hasattr(raw_start, "timestamp") else 0

        _serialize(d, ["startTime", "endTime", "updatedAt"])
        d["violationCount"] = trip_violation_counts.get(doc.id, 0)
        items.append(d)

    items.sort(key=lambda x: x.pop("_sort", 0), reverse=True)
    return {"count": len(items), "items": items[:100]}


@app.patch("/admin/violations/{violation_id}/appeal")
def admin_review_appeal(
    violation_id: str,
    user=Depends(require_admin),
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    payload = payload or {}
    decision = (payload.get("decision") or "").strip()
    note = (payload.get("note") or "").strip()

    if decision not in ("approved", "rejected"):
        return JSONResponse({"error": "decision must be 'approved' or 'rejected'"}, status_code=400)

    ref = db.collection("violations").document(violation_id)
    if not ref.get().exists:
        return JSONResponse({"error": "violation not found"}, status_code=404)

    ref.update({
        "appealStatus": decision,
        "appealNote": note,
        "appealReviewedBy": user["uid"],
        "appealReviewedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    return {"success": True, "appealStatus": decision}


@app.patch("/admin/drivers/{driver_uid}/status")
def admin_update_driver_status(
    driver_uid: str,
    user=Depends(require_admin),
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    if isinstance(user, JSONResponse):
        return user
    if db is None:
        return JSONResponse({"error": "Firebase not initialized"}, status_code=500)

    payload = payload or {}
    new_status = (payload.get("status") or "").strip()

    if new_status not in ("active", "suspended"):
        return JSONResponse({"error": "status must be 'active' or 'suspended'"}, status_code=400)

    ref = db.collection("drivers").document(driver_uid)
    if not ref.get().exists:
        return JSONResponse({"error": "driver not found"}, status_code=404)

    ref.update({
        "status": new_status,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

    return {"success": True, "status": new_status}


# ------------------------------------------
# MAIN: /detect endpoint
# ------------------------------------------
@app.post("/detect")
async def detect_lane(request: Request, user=Depends(require_user), file: UploadFile = File(...)):
    global saving_clip, last_violation_time, violation_armed, no_violation_streak, last_frame_time

    if isinstance(user, JSONResponse):
        return user

    uid = user["uid"]

    trip_id = request.headers.get("X-Trip-Id", "").strip()
    if not trip_id:
        return JSONResponse({"error": "tripId required (send X-Trip-Id header)"}, status_code=400)

    trip_err = _require_trip_for_user(uid, trip_id)
    if trip_err:
        return trip_err

    location = None
    try:
        lat = request.headers.get("X-Latitude")
        lng = request.headers.get("X-Longitude")
        if lat and lng:
            location = {"lat": float(lat), "lng": float(lng)}
    except (ValueError, TypeError):
        pass

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    now = time.time()
    if last_frame_time > 0 and (now - last_frame_time) > STREAM_RESET_GAP_SECONDS:
        violation_armed = True
        no_violation_streak = 0
        if hasattr(lane_system, "lanelines") and hasattr(lane_system.lanelines, "reset_stream_state"):
            lane_system.lanelines.reset_stream_state()
    last_frame_time = now

    frame_small = cv2.resize(frame_bgr, (640, 360))
    frame_buffer.add_frame(frame_small)

    frame_proc = cv2.resize(frame_bgr, (960, 540))
    frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    out_rgb = lane_system.forward(frame_rgb)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    overlay_small = cv2.resize(out_bgr, (640, 360))
    ok, jpg_buf = cv2.imencode(".jpg", overlay_small, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
    if not ok:
        return JSONResponse({"error": "Encoding failed"}, status_code=500)

    overlay_b64 = base64.b64encode(jpg_buf).decode("utf-8")

    left_type = left_conf = None
    right_type = right_conf = None
    violation_detected = False
    violation_event = None
    clip_path = None

    if hasattr(lane_system, "lanelines"):
        ll = lane_system.lanelines

        if hasattr(ll, "left_marking"):
            lt, lc = ll.left_marking
            left_type, left_conf = lt, float(lc)

        if hasattr(ll, "right_marking"):
            rt, rc = ll.right_marking
            right_type, right_conf = rt, float(rc)

        is_violation_now = hasattr(ll, "violation_event") and ll.violation_event is not None

        if is_violation_now:
            violation_detected = True
            violation_event = ll.violation_event
            no_violation_streak = 0

            with saving_clip_lock:
                can_save = (
                    violation_armed
                    and (not saving_clip)
                    and ((now - last_violation_time) >= VIOLATION_SAVE_COOLDOWN)
                )

                if can_save:
                    saving_clip = True
                    last_violation_time = now
                    violation_armed = False

                    ts = int(now)
                    clip_path = os.path.join(CLIPS_DIR, f"violation_{ts}.mp4")

                    recording = frame_buffer.start_recording()
                    threading.Thread(
                        target=save_clip_background,
                        args=(frame_buffer, recording, clip_path, uid, trip_id, location),
                        daemon=True
                    ).start()

                    ll.violation_event = None
        else:
            no_violation_streak += 1
            if no_violation_streak >= NO_VIOLATION_REARM_FRAMES:
                violation_armed = True

    return {
        "overlay": overlay_b64,
        "left_type": left_type,
        "left_conf": left_conf,
        "right_type": right_type,
        "right_conf": right_conf,
        "violation_detected": violation_detected,
        "violation_event": violation_event,
        "clip_path": clip_path,
        "saving_clip": saving_clip,
        "tripId": trip_id,
    }
