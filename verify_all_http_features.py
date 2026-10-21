import io
import json
import base64
import urllib.request
import urllib.error
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://127.0.0.1:8000"

def make_test_image(text="TEST DOCUMENT"):
    img = Image.new("RGB", (800, 600), color=(240, 240, 235))
    d = ImageDraw.Draw(img)
    # Simulate a document page with header and lines
    d.rectangle([(80, 60), (720, 540)], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    d.text((120, 100), text, fill=(0, 0, 0))
    d.text((120, 150), "CamScanner Pro Automated Test Verification", fill=(50, 50, 50))
    d.text((120, 200), "Section 1: High Resolution Digital Capture", fill=(30, 30, 30))
    d.text((120, 240), "Section 2: Bilinear Homography Perspective Rectification", fill=(30, 30, 30))
    d.text((120, 280), "Section 3: Rapid Multilingual Optical Character Recognition", fill=(30, 30, 30))
    d.text((120, 320), "Section 4: Anti-Theft Watermarking & Electronic Signatures", fill=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def upload_multipart(url, fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    for name, filename, data, mime in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(url, data=bytes(body), headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    })
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read()

def post_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read()

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read(), resp.headers

def run_all_checks():
    print("==================================================")
    print("STARTING FULL END-TO-END SYSTEM VERIFICATION")
    print("==================================================")

    # 1. Test Batch Upload
    img1 = make_test_image("PAGE 1: CONTRACT AGREEMENT")
    img2 = make_test_image("PAGE 2: TERMS AND CONDITIONS")
    files = [
        ("files", "page1.jpg", img1, "image/jpeg"),
        ("files", "page2.jpg", img2, "image/jpeg")
    ]
    status, res = upload_multipart(f"{BASE_URL}/api/scan/batch-upload", {"title": "E2E Master Contract", "category": "Contract"}, files)
    assert status == 200, f"Batch upload failed: {res}"
    doc = json.loads(res.decode())
    doc_id = doc["id"]
    assert len(doc["pages"]) == 2
    print(f"[OK] 1. Batch Upload successful! Doc ID: {doc_id}, Pages: {len(doc['pages'])}")

    page1_id = doc["pages"][0]["id"]
    page2_id = doc["pages"][1]["id"]

    # 2. Test Electronic Signature Stamping on Page 1
    sig_canvas = Image.new("RGBA", (140, 60), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sig_canvas)
    sd.line([(10, 50), (45, 15), (75, 40), (125, 20)], fill=(30, 58, 138, 255), width=3)
    sbuf = io.BytesIO()
    sig_canvas.save(sbuf, format="PNG")
    sig_b64 = "data:image/png;base64," + base64.b64encode(sbuf.getvalue()).decode()

    sig_payload = {
        "page_id": page1_id,
        "signature_base64": sig_b64,
        "position": "br"
    }
    status, res = post_json(f"{BASE_URL}/api/documents/{doc_id}/signature", sig_payload)
    assert status == 200, f"Signature stamping failed: {res}"
    print(f"[OK] 2. Electronic Signature stamped on page 1 ({page1_id})")

    # 3. Test Anti-Theft Watermark across all pages
    wm_payload = {
        "watermark_text": "OFFICIAL COPY - VERIFIED",
        "opacity": 0.25,
        "color": "amber"
    }
    status, res = post_json(f"{BASE_URL}/api/documents/{doc_id}/watermark", wm_payload)
    assert status == 200, f"Watermarking failed: {res}"
    print(f"[OK] 3. Anti-Theft Watermark applied to all pages")

    # 4. Test Searchable PDF Export
    status, pdf_bytes, headers = get(f"{BASE_URL}/api/documents/{doc_id}/export-pdf")
    assert status == 200 and len(pdf_bytes) > 2000
    assert b"%PDF" in pdf_bytes[:10]
    print(f"[OK] 4. Searchable Dual-Layer PDF exported ({len(pdf_bytes)} bytes)")

    # 5. Test Microsoft Word DOCX Export
    status, docx_bytes, headers = get(f"{BASE_URL}/api/documents/{doc_id}/export-docx")
    assert status == 200 and len(docx_bytes) > 2000
    assert docx_bytes[:2] == b"PK"  # Valid ZIP/Office Open XML header
    print(f"[OK] 5. Microsoft Word DOCX exported ({len(docx_bytes)} bytes)")

    # 6. Test High-Res ZIP Export
    status, zip_bytes, headers = get(f"{BASE_URL}/api/documents/{doc_id}/export-zip")
    assert status == 200 and len(zip_bytes) > 1000
    assert zip_bytes[:2] == b"PK"
    print(f"[OK] 6. High-Res JPEG Scans ZIP archive exported ({len(zip_bytes)} bytes)")

    # 7. Test Plain Text OCR Export
    status, txt_bytes, headers = get(f"{BASE_URL}/api/documents/{doc_id}/export-txt")
    assert status == 200 and len(txt_bytes) > 20
    print(f"[OK] 7. Plain Text OCR exported ({len(txt_bytes)} bytes)")

    # 8. Test ID Card Front & Back Composite Endpoint
    id_front = make_test_image("ID CARD FRONT: CITIZEN ID")
    id_back = make_test_image("ID CARD BACK: ADDRESS DETAILS")
    id_files = [
        ("front", "front.jpg", id_front, "image/jpeg"),
        ("back", "back.jpg", id_back, "image/jpeg")
    ]
    status, res = upload_multipart(f"{BASE_URL}/api/scan/id-card-files", {"title": "Citizen ID Card 1:1"}, id_files)
    assert status == 200
    id_doc = json.loads(res.decode())
    print(f"[OK] 8. ID Card 1:1 Scale Composite generated (Doc ID: {id_doc['id']})")

    # 9. Clean up test documents
    req = urllib.request.Request(f"{BASE_URL}/api/documents/{doc_id}", method="DELETE")
    urllib.request.urlopen(req)
    req = urllib.request.Request(f"{BASE_URL}/api/documents/{id_doc['id']}", method="DELETE")
    urllib.request.urlopen(req)
    print(f"[OK] 9. Test documents cleaned up cleanly.")

    print("==================================================")
    print("ALL 9 CRITICAL HTTP END-TO-END PIPELINES VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    run_all_checks()
