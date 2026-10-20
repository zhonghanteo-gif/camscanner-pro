# CamScanner Pro (Full-Stack Document Scanner & OCR Engine)

An enterprise-grade, full-stack document scanning, perspective correction, and multi-page PDF engine built with **Python 3.12, FastAPI, OpenCV, and RapidOCR**.

Designed to deliver the authentic **CamScanner** mobile experience on Android phones, iPhones, and desktop browsers.

---

## Architecture Overview

```
camscanner-fullstack/
├── app.py                 # Core FastAPI full-stack server + embedded PWA mobile UI
├── requirements.txt       # Production Python dependencies
├── Dockerfile             # Containerized deployment spec
├── render.yaml            # 1-click cloud deployment blueprint
├── run.bat                # 1-click Windows launcher
├── test_api.py            # Comprehensive automated integration test suite
├── storage/               # Auto-initialized local persistence
│   ├── uploads/           # Raw high-resolution captured photos
│   ├── pages/             # Processed, perspective-warped, filtered document pages
│   ├── thumbnails/        # Compressed thumbnails for instant dashboard loading
│   ├── pdfs/              # Standardized multi-page A4 PDFs
│   └── data.json          # Document collection database
└── static/                # Static assets & icons
```

---

## Core Capabilities

1. **Auto Document Edge Detection (OpenCV)**
   - Scans camera photos on desks or tables.
   - Uses Canny edge detection, morphological dilation, and `cv2.approxPolyDP` to automatically detect the 4 paper corners.

2. **Homography Perspective Transform**
   - Implements `cv2.getPerspectiveTransform` and `cv2.warpPerspective` with Lanczos-4 anti-aliasing to flatten skewed photos into crisp, rectangular sheets.

3. **CamScanner "Magic Color" Enhancement**
   - Histogram percentile analysis on luminance channel to whiten paper backgrounds and erase desk shadows.
   - Text stroke deepening and unsharp masking for sharp legibility.
   - Color saturation preservation so official colored seals, signatures, and stamps pop against white paper.
   - Additional filters: **Clean B&W**, **Charcoal B&W 2**, **Grayscale**, **Lighten**, and **Original**.

4. **Offline Multilingual OCR (RapidOCR)**
   - Extracts printed and handwritten text using ONNX runtime without external dependencies.
   - 1-click text copying for notes, invoices, and documents.

5. **Multi-Page Collections & PDF Engine**
   - Add multiple scanned pages into one document.
   - One-tap export to standardized A4 PDFs with instant download or native Android share sheet (`navigator.share`).

6. **Mobile PWA Ready**
   - Real-time interactive 8-point boundary crop screen.
   - **Touch Magnifier Loupe:** Displays a 2.5x circular zoom lens above your thumb when dragging corner handles.
   - Emerald teal CamScanner branding (`#189a80`).

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```
Or simply double-click `run.bat` on Windows.

### 3. Open on Android Phone
Ensure your phone and computer are on the same Wi-Fi network, then open Chrome on your phone:
```
http://<YOUR_PC_LAN_IP>:8000
```
In Chrome's menu (`⋮`), tap **"Add to Home screen"** or **"Install app"** to install it as a standalone app!

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/scan/upload` | Upload image & detect 4 corners |
| `POST` | `/api/scan/warp-filter` | Perspective warp & apply filter |
| `POST` | `/api/scan/ocr/{page_id}` | Extract text with RapidOCR |
| `GET` | `/api/documents` | List all documents |
| `POST` | `/api/documents` | Create document collection |
| `GET` | `/api/documents/{id}` | Get document details & pages |
| `PUT` | `/api/documents/{id}` | Rename document |
| `DELETE` | `/api/documents/{id}` | Delete document and page files |
| `POST` | `/api/documents/{id}/pages` | Append page to document |
| `DELETE` | `/api/documents/{id}/pages/{pid}` | Remove page from document |
| `GET` | `/api/documents/{id}/export-pdf` | Download multi-page A4 PDF |
| `GET` | `/files/{folder}/{filename}` | Static file hosting URL |

---

## Automated Verification

Run the test suite:
```bash
python test_api.py
```
This tests synthetic document creation, corner detection, homography warp, filter processing, OCR extraction, and multi-page PDF generation.
