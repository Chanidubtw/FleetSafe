import requests

API_KEY = "AIzaSyBl6BY9OE_-m0gTVkNH1F_V5eQE8_VRXPw"

EMAIL = "driver@test.com"
PASSWORD = "12345678"

url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"

payload = {
    "email": EMAIL,
    "password": PASSWORD,
    "returnSecureToken": True
}

r = requests.post(url, json=payload)

data = r.json()

if "idToken" in data:
    print("\n✅ LOGIN SUCCESS\n")
    print("ID TOKEN:\n")
    print(data["idToken"])
else:
    print("❌ Login failed")
    print(data)


