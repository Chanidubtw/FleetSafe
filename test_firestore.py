from firebase_setup import init_firebase
from firebase_admin import firestore

db, bucket = init_firebase()

doc = {
    "hello": "firebase works",
    "createdAt": firestore.SERVER_TIMESTAMP
}

db.collection("test").add(doc)
print("✅ inserted test document")
