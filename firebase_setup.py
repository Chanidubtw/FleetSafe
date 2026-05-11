import os

import firebase_admin
from firebase_admin import credentials, firestore, storage

def init_firebase():
    """
    Initializes Firebase Admin SDK once.
    Returns: (db, bucket)
    """
    if not firebase_admin._apps:
        cred_path = os.environ.get("FIREBASE_ADMIN_CREDENTIALS", "firebase_key.json")
        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                f"Firebase service account key not found: {cred_path}. "
                "Set FIREBASE_ADMIN_CREDENTIALS or place firebase_key.json in the project root."
            )

        cred = credentials.Certificate(cred_path)

        # Use your bucket name here:
        # You said: gs://laneviolation-71313.firebasestorage.app
        # So bucket name is: laneviolation-71313.firebasestorage.app
        firebase_admin.initialize_app(cred, {
            "storageBucket": "laneviolation-71313.firebasestorage.app"
        })

    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket
