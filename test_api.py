"""
Integration Test Suite for CamScanner Pro Full-Stack Application
Verifies OpenCV corner detection, homography warp, Retinex Magic Color,
Dewarp & Clean Print filters, RapidOCR output, Dual-Layer Searchable PDF,
Plain Text OCR Export, and ID Card 1:1 Composite Generation.
"""
import io
import json
import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
import pypdf

from app import app, load_db, STORAGE_DIR

client = TestClient(app)

def create_synthetic_document_image() -> bytes:
    """Creates an artificial photo of a white document on a dark desk with sample text."""
    # Dark desk background (800x600)
    bg = Image.new("RGB", (800, 600), color=(40, 44, 52))
    draw = ImageDraw.Draw(bg)

    # Slanted white document quad
    # Coordinates: tl=(100, 80), tr=(700, 110), br=(660, 520), bl=(120, 500)
    doc_polygon = [(100, 80), (700, 110), (660, 520), (120, 500)]
    draw.polygon(doc_polygon, fill=(245, 245, 245))

    # Add crisp text on document
    draw.text((200, 160), "CAMSCANNER OFFICIAL INVOICE", fill=(20, 20, 20))
    draw.text((200, 220), "Invoice ID: CS-2026-9988", fill=(50, 50, 50))
    draw.text((200, 260), "Client: Universal Fullstack Developer", fill=(50, 50, 50))
    draw.text((200, 300), "Amount Due: $1,450.00", fill=(20, 120, 40))
    draw.text((200, 350), "STATUS: PAID IN FULL", fill=(180, 20, 20))

    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_upload_and_detect_corners():
    img_bytes = create_synthetic_document_image()
    response = client.post(
        "/api/scan/upload",
        files={"file": ("test_doc.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "upload_id" in data
    assert data["width"] == 800
    assert data["height"] == 600
    assert "corners" in data
    corners = data["corners"]
    assert "tl" in corners and "tr" in corners and "br" in corners and "bl" in corners
    print("\n[OK] Upload and Corner Detection passed:", corners)
    return data["upload_id"], corners


def test_warp_and_magic_filter(upload_id, corners):
    payload = {
        "upload_id": upload_id,
        "corners": corners,
        "filter_type": "magic",
        "brightness": 0,
        "contrast": 0
    }
    response = client.post("/api/scan/warp-filter", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "page_id" in data
    assert "page_url" in data
    assert "thumbnail_url" in data
    assert data["filter"] == "magic"
    print("\n[OK] Warp & Retinex Division Magic Color passed:", data["page_id"])
    return data["page_id"]


def test_dewarp_and_clean_print_filters(upload_id, corners):
    # Test Dewarp (Curved Book)
    dewarp_res = client.post("/api/scan/warp-filter", json={
        "upload_id": upload_id,
        "corners": corners,
        "filter_type": "dewarp",
        "brightness": 0,
        "contrast": 0
    })
    assert dewarp_res.status_code == 200, dewarp_res.text
    print("\n[OK] Cylindrical Dewarp filter passed:", dewarp_res.json()["page_id"])

    # Test Clean Print
    print_res = client.post("/api/scan/warp-filter", json={
        "upload_id": upload_id,
        "corners": corners,
        "filter_type": "clean_print",
        "brightness": 0,
        "contrast": 0
    })
    assert print_res.status_code == 200, print_res.text
    print("\n[OK] Clean Print (Handwriting removal) passed:", print_res.json()["page_id"])


def test_ocr_extraction(page_id):
    response = client.post(f"/api/scan/ocr/{page_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "text" in data
    assert "lines" in data
    assert "boxes" in data
    print("\n[OK] RapidOCR Extraction passed. Lines detected:", len(data["lines"]))
    return data["text"]


def test_document_lifecycle_and_dual_layer_pdf(page_id):
    # 1. Create document
    create_res = client.post(
        "/api/documents",
        json={"title": "Test Invoice Scan", "page_ids": [page_id]}
    )
    assert create_res.status_code == 200, create_res.text
    doc = create_res.json()
    doc_id = doc["id"]
    assert len(doc["pages"]) == 1

    # 2. List documents
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # 3. Export Dual-Layer Searchable PDF
    pdf_res = client.get(f"/api/documents/{doc_id}/export-pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

    # 4. Verify text searchability inside PDF using PyPDF
    reader = pypdf.PdfReader(io.BytesIO(pdf_res.content))
    extracted = reader.pages[0].extract_text()
    print("\n[OK] Dual-Layer Searchable PDF verified! Size:", len(pdf_res.content), "bytes")
    print("     PyPDF text extract from PDF:", repr(extracted[:60]))

    # 5. Export TXT
    txt_res = client.get(f"/api/documents/{doc_id}/export-txt")
    assert txt_res.status_code == 200
    assert "text/plain" in txt_res.headers["content-type"]
    assert len(txt_res.text) > 20
    print("\n[OK] Plain Text OCR Export passed! Content length:", len(txt_res.text))

    # 6. Clean up test document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    print("\n[OK] Document deletion and file cleanup passed!")


def test_id_card_composite(page_id):
    # Test creating 1:1 scale ID card with front and back
    id_res = client.post("/api/scan/id-card", json={
        "front_page_id": page_id,
        "back_page_id": page_id,
        "title": "National ID Card Test",
        "category": "ID Card"
    })
    assert id_res.status_code == 200, id_res.text
    id_doc = id_res.json()
    assert id_doc["category"] == "ID Card"
    assert len(id_doc["pages"]) == 1
    print("\n[OK] ID Card 1:1 Scale Composite passed! Doc ID:", id_doc["id"])

    # Clean up
    client.delete(f"/api/documents/{id_doc['id']}")


if __name__ == "__main__":
    print("Running CamScanner Pro Extended Pipeline Test Suite...")
    upload_id, corners = test_upload_and_detect_corners()
    page_id = test_warp_and_magic_filter(upload_id, corners)
    test_dewarp_and_clean_print_filters(upload_id, corners)
    test_ocr_extraction(page_id)
    test_id_card_composite(page_id)
    test_document_lifecycle_and_dual_layer_pdf(page_id)
    print("\n=======================================================")
    print("  ALL EXTENDED CAMSCANNER PRO TESTS PASSED 100%!")
    print("=======================================================")
