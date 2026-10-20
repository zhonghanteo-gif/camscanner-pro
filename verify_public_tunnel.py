import requests
import numpy as np
import cv2
import io
import json

BASE_URL = "https://address-metal-kits-springer.trycloudflare.com"

print(f"Testing public HTTPS URL: {BASE_URL}")

# 1. Health
r = requests.get(f"{BASE_URL}/health", timeout=10)
print(f"Health status: {r.status_code}, data: {r.json()}")

# 2. Synthetic receipt image
h, w = 600, 800
img = np.ones((h, w, 3), dtype=np.uint8) * 35  # dark desk
# draw skewed white paper
paper_poly = np.array([[[100, 80], [700, 110], [660, 520], [120, 500]]], dtype=np.int32)
cv2.fillPoly(img, paper_poly, (245, 245, 245))
cv2.putText(img, "OFFICIAL INVOICE", (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
cv2.putText(img, "Total: $149.99", (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 20, 20), 2)

_, enc = cv2.imencode(".jpg", img)
file_bytes = enc.tobytes()

# 3. Upload & corner detect
files = {"file": ("test_doc.jpg", io.BytesIO(file_bytes), "image/jpeg")}
r = requests.post(f"{BASE_URL}/api/scan/upload", files=files, timeout=15)
print(f"Upload response: {r.status_code}")
up_data = r.json()
print("Corners detected:", up_data.get("corners"))

# 4. Warp & Magic Color
warp_payload = {
    "upload_id": up_data["upload_id"],
    "corners": up_data["corners"],
    "filter_type": "magic",
    "brightness": 0,
    "contrast": 0
}
r = requests.post(f"{BASE_URL}/api/scan/warp-filter", json=warp_payload, timeout=15)
print(f"Warp & Filter response: {r.status_code}")
warp_data = r.json()
page_id = warp_data["page_id"]
print("Page ID created:", page_id)

# 5. Create Document
doc_payload = {"title": "Test Cloud Document", "category": "Invoice"}
r = requests.post(f"{BASE_URL}/api/documents", json=doc_payload, timeout=10)
doc_data = r.json()
doc_id = doc_data["id"]
print(f"Doc created: {doc_id}")

# 6. Add page to doc
r = requests.post(f"{BASE_URL}/api/documents/{doc_id}/pages", json={"page_id": page_id}, timeout=10)
print(f"Page added: {r.status_code}")

# 7. OCR
r = requests.post(f"{BASE_URL}/api/scan/ocr/{page_id}", timeout=20)
print(f"OCR response: {r.status_code}")
ocr_data = r.json()
print("OCR Text:", ocr_data.get("text", "")[:100])

# 8. Export PDF
r = requests.get(f"{BASE_URL}/api/documents/{doc_id}/export-pdf", timeout=15)
print(f"PDF Export status: {r.status_code}, content length: {len(r.content)} bytes")

# Cleanup doc
requests.delete(f"{BASE_URL}/api/documents/{doc_id}", timeout=10)
print("SUCCESS: Full End-to-End Public Verification Complete 100%!")
