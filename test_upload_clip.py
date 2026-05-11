from firebase_upload import upload_violation_clip

clip = "clips/violation_1770875278.mp4"  # change if needed
doc_id, url = upload_violation_clip(clip, driver_id="demo_driver")

print("✅ Firestore doc:", doc_id)
print("✅ Video URL:", url)
