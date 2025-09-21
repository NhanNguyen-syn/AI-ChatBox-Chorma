import requests, json
base = "http://127.0.0.1:8000/api"

login_data = {"username": "SG0510", "password": "admin123"}
print("Logging in as admin...")
r = requests.post(base + "/auth/login", json=login_data)
print("Login status:", r.status_code)
print("Login response:", r.text)

if r.status_code != 200:
    raise SystemExit("Login failed")

token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

print("Requesting /files/documents...")
r2 = requests.get(base + "/files/documents", headers=headers)
print("Documents status:", r2.status_code)
print("Documents response:", r2.text)

