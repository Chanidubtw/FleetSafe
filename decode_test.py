import base64
import json
import cv2
import numpy as np

with open("response.json") as f:
    data = json.load(f)

img_bytes = base64.b64decode(data["overlay"])
np_arr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

cv2.imshow("overlay", img)
cv2.waitKey(0)
