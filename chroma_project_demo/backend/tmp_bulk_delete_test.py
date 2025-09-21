import requests, json
base = "http://127.0.0.1:8000/api"
login_data = {"username": "SG0510", "password": "admin123"}
r = requests.post(base + "/auth/login", json=login_data)
assert r.status_code == 200, r.text
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
# fetch docs
r2 = requests.get(base + "/files/documents", headers=headers)
print("GET status:", r2.status_code)
print("GET:", r2.text)
if r2.status_code == 200:
    data = r2.json()
    ids = [d["id"] for d in data.get("documents", [])]
    print("Found", len(ids), "doc ids")
    if ids:
        r3 = requests.delete(base + "/files/documents", headers=headers, data=json.dumps({"ids": ids[:3]}))
        print("DELETE status:", r3.status_code)
        print("DELETE resp:", r3.text)
        r4 = requests.get(base + "/files/documents", headers=headers)
        print("After GET:", r4.text)

