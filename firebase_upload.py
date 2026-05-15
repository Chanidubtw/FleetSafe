# firebase_upload.py
import os
import time
from typing import Any, Dict, Optional, Tuple

from firebase_admin import firestore
from firebase_setup import init_firebase

_db = None
_bucket = None


def _get_clients():
    global _db, _bucket
    if _db is None or _bucket is None:
        _db, _bucket = init_firebase()
    return _db, _bucket


def upload_violation_clip(
    local_path: str,
    driver_id: str,
    trip_id: str,
    location: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Uploads the local mp4 to Firebase Storage + creates Firestore doc in 'violations'.
    Returns (doc_id, payload_dict).
    """
    db, bucket = _get_clients()

    epoch = int(time.time())
    base = os.path.basename(local_path)
    storage_path = f"violations/{driver_id}/{epoch}_{base}"

    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_path, content_type="video/mp4")

    # DEMO: make public
    blob.make_public()
    video_url = blob.public_url

    # Denormalize busNo/routeNo from the trip for easier querying
    bus_no = None
    route_no = None
    try:
        trip_snap = db.collection("trips").document(trip_id).get()
        if trip_snap.exists:
            trip_data = trip_snap.to_dict() or {}
            bus_no = trip_data.get("busNo")
            route_no = trip_data.get("routeNo")
    except Exception as e:
        print("Warning: could not fetch trip data for violation:", e)

    doc = {
        "driverId": driver_id,
        "tripId": trip_id,
        "busNo": bus_no,
        "routeNo": route_no,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "createdAtEpoch": epoch,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "videoUrl": video_url,
        "storagePath": storage_path,
        "status": "pending",
        "aiResult": None,
        "location": location,
        "appealStatus": None,
        "appealReason": None,
        "appealedAt": None,
    }

    ref = db.collection("violations").document()
    ref.set(doc)

    doc["id"] = ref.id
    return ref.id, doc


def _status_from_ai(ai: Dict[str, Any]) -> str:
    decision = (ai.get("decision") or "").strip().lower()
    if decision in ["violation", "isviolation", "true"]:
        return "violation"
    if decision in ["notviolation", "not_violation", "false", "no_violation"]:
        return "not_violation"
    if decision in ["uncertain", "unknown"]:
        return "pending"
    return "pending"


def update_violation_ai_result(doc_id: str, ai_result: Dict[str, Any], failed: bool = False) -> None:
    db, _ = _get_clients()

    if failed:
        status = "failed"
    else:
        status = _status_from_ai(ai_result)

    db.collection("violations").document(doc_id).update({
        "aiResult": ai_result,
        "status": status,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })


