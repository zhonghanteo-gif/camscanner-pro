import os
import io
import json
import math
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import base64
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Initialize Directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PAGE_DIR = STORAGE_DIR / "pages"
THUMB_DIR = STORAGE_DIR / "thumbnails"
PDF_DIR = STORAGE_DIR / "pdfs"
STATIC_DIR = BASE_DIR / "static"
DB_FILE = STORAGE_DIR / "data.json"

for d in [UPLOAD_DIR, PAGE_DIR, THUMB_DIR, PDF_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize RapidOCR
try:
    from rapidocr import RapidOCR
    ocr_engine = RapidOCR()
    HAS_OCR = True
except Exception as e:
    ocr_engine = None
    HAS_OCR = False
    print(f"[Warning] RapidOCR not available: {e}")


# ==========================================
# Database Helper (JSON Store)
# ==========================================
def load_db() -> dict:
    if not DB_FILE.exists():
        initial = {"documents": []}
        save_db(initial)
        return initial
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"documents": []}


def save_db(data: dict):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==========================================
# Image Processing & Computer Vision Engine
# ==========================================
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Orders 4 coordinates: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    return rect


def detect_document_corners(image_bgr: np.ndarray) -> dict:
    """Detects 4 corners of a document on a desk, with fallback to balanced crop."""
    h, w = image_bgr.shape[:2]
    max_dim = 900
    scale = min(1.0, max_dim / max(h, w))
    small_w = int(w * scale)
    small_h = int(h * scale)
    small = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multi-pass edge detection
    edges = cv2.Canny(blurred, 35, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = small_w * small_h
    best_quad = None

    for c in contours[:10]:
        area = cv2.contourArea(c)
        if area < 0.12 * img_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype("float32")
            best_quad = pts / scale
            break

    if best_quad is not None:
        ordered = order_corners(best_quad)
        return {
            "tl": {"x": int(ordered[0][0]), "y": int(ordered[0][1])},
            "tr": {"x": int(ordered[1][0]), "y": int(ordered[1][1])},
            "br": {"x": int(ordered[2][0]), "y": int(ordered[2][1])},
            "bl": {"x": int(ordered[3][0]), "y": int(ordered[3][1])},
        }

    # Fallback: 8% margin inset
    inset_x = int(w * 0.08)
    inset_y = int(h * 0.08)
    return {
        "tl": {"x": inset_x, "y": inset_y},
        "tr": {"x": w - inset_x, "y": inset_y},
        "br": {"x": w - inset_x, "y": h - inset_y},
        "bl": {"x": inset_x, "y": h - inset_y},
    }


def warp_document(image_bgr: np.ndarray, corners: dict, max_dim: int = 2400) -> np.ndarray:
    """Warps an arbitrary quadrilateral image region into a flat rectangular document."""
    tl = np.array([corners["tl"]["x"], corners["tl"]["y"]], dtype="float32")
    tr = np.array([corners["tr"]["x"], corners["tr"]["y"]], dtype="float32")
    br = np.array([corners["br"]["x"], corners["br"]["y"]], dtype="float32")
    bl = np.array([corners["bl"]["x"], corners["bl"]["y"]], dtype="float32")

    w_top = np.linalg.norm(tr - tl)
    w_bottom = np.linalg.norm(br - bl)
    max_w = int(max(w_top, w_bottom))

    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    max_h = int(max(h_left, h_right))

    # Clamp dimensions for performance
    if max(max_w, max_h) > max_dim:
        s = max_dim / max(max_w, max_h)
        max_w = int(max_w * s)
        max_h = int(max_h * s)

    max_w = max(max_w, 200)
    max_h = max(max_h, 200)

    src = np.array([tl, tr, br, bl], dtype="float32")
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image_bgr, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def apply_camscanner_filter(
    image_bgr: np.ndarray,
    filter_type: str = "magic",
    brightness: int = 0,
    contrast: int = 0,
) -> np.ndarray:
    """Applies CamScanner filters: Magic Color, Clean B&W, Charcoal, Grayscale, Lighten, Original."""
    img = image_bgr.copy()

    if filter_type == "magic":
        # CamScanner Signature "Magic Color" via Retinex Division Normalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1. Background Illumination Estimation via large morphological dilation & Gaussian blur
        h, w = l.shape[:2]
        k_size = max(17, (min(h, w) // 35) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dilated = cv2.dilate(l, kernel)
        bg_illum = cv2.medianBlur(dilated, k_size)
        bg_illum = cv2.GaussianBlur(bg_illum, (k_size, k_size), 0)

        # 2. Division Normalization: I_norm = min(255, (I / I_bg) * 255)
        # Erases shadows and overhead lighting gradients across the paper
        norm_l = np.clip((l.astype(np.float32) / np.maximum(bg_illum.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)

        # 3. Dynamic range expansion: whiten near-whites, deepen dark text strokes
        norm_l[norm_l > 220] = 255
        norm_l[norm_l < 50] = (norm_l[norm_l < 50] * 0.8).astype(np.uint8)

        # 4. Unsharp masking kernel for razor-sharp text legibility
        gaussian = cv2.GaussianBlur(norm_l, (0, 0), 2.2)
        l_sharp = cv2.addWeighted(norm_l, 1.45, gaussian, -0.45, 0)

        # 5. Chromaticity restoration so official red stamps and blue pen ink pop
        a_boost = cv2.addWeighted(a, 1.25, np.full_like(a, 128), -0.25, 0)
        b_boost = cv2.addWeighted(b, 1.25, np.full_like(b, 128), -0.25, 0)

        merged = cv2.merge([l_sharp, a_boost, b_boost])
        res = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    elif filter_type == "no_shadow":
        # Official CamScanner "No Shadow" illumination flattening without harsh thresholding
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        h, w = l.shape[:2]
        k_size = max(21, (min(h, w) // 30) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dilated = cv2.dilate(l, kernel)
        bg = cv2.GaussianBlur(dilated, (k_size, k_size), 0)
        norm_l = np.clip((l.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 240.0, 0, 255).astype(np.uint8)
        merged = cv2.merge([norm_l, a, b])
        res = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    elif filter_type == "dewarp" or filter_type == "curved_book":
        # Cylindrical dewarping for curved pages / book spines
        h, w = img.shape[:2]
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, np.newaxis], (1, w))

        # Sine-based vertical baseline arc correction
        amp = max(10.0, h * 0.035)
        offset = amp * np.sin(np.pi * (map_x / w))
        map_y_shifted = np.clip(map_y - offset + (amp * 0.5), 0, h - 1).astype(np.float32)

        dewarped = cv2.remap(img, map_x, map_y_shifted, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        return apply_camscanner_filter(dewarped, "magic", brightness, contrast)

    elif filter_type == "clean_print":
        # High-contrast printed template filter (attenuates stray ballpoint/pencil artifacts)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        bw = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12)
        res = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    elif filter_type == "bw":
        # Clean B&W Document (Adaptive Gaussian binarization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply median blur to reduce salt-and-pepper noise
        smoothed = cv2.medianBlur(gray, 3)
        bw = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        res = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    elif filter_type == "bw2":
        # Charcoal B&W (Deep contrast with anti-aliased stroke edges)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p5 = np.percentile(gray, 5)
        p92 = np.percentile(gray, 92)
        val_range = max(30, p92 - p5)
        stretched = np.clip(((gray.astype(np.float32) - p5) / val_range) * 255.0, 0, 255)

        # S-curve
        stretched[stretched > 170] = 255
        stretched[stretched < 65] = 0
        mid = (stretched >= 65) & (stretched <= 170)
        stretched[mid] = ((stretched[mid] - 65) / 105.0) * 255.0

        res = cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    elif filter_type == "grayscale":
        # Clean balanced document grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        res = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    elif filter_type == "lighten":
        # Gamma compensation to lift dark shadows
        gamma = 0.65
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        res = cv2.LUT(img, table)

    else:
        # Original
        res = img

    # Manual brightness & contrast fine-tuning
    if brightness != 0 or contrast != 0:
        alpha = (contrast + 100.0) / 100.0
        beta = (brightness / 100.0) * 50.0
        res = cv2.convertScaleAbs(res, alpha=alpha, beta=beta)

    return res


# ==========================================
# FastAPI Application & Lifespan
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  CamScanner Full-Stack Server Starting...")
    print(f"  Storage root: {STORAGE_DIR}")
    print(f"  OCR Engine Status: {'Active (RapidOCR)' if HAS_OCR else 'Disabled'}")
    print("=" * 60)
    yield


app = FastAPI(
    title="CamScanner Pro Full-Stack Application",
    description="Full-stack Document Scanner & Multi-page PDF Engine with CamScanner Magic Color & RapidOCR.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount permanent hosted static file access
app.mount("/files/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/files/pages", StaticFiles(directory=str(PAGE_DIR)), name="pages")
app.mount("/files/thumbnails", StaticFiles(directory=str(THUMB_DIR)), name="thumbnails")
app.mount("/files/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==========================================
# Pydantic Schemas
# ==========================================
class CornerPoint(BaseModel):
    x: int
    y: int


class CornersInput(BaseModel):
    tl: CornerPoint
    tr: CornerPoint
    br: CornerPoint
    bl: CornerPoint


class WarpFilterRequest(BaseModel):
    upload_id: str
    corners: CornersInput
    filter_type: str = "magic"
    brightness: int = 0
    contrast: int = 0


class DocumentCreateRequest(BaseModel):
    title: Optional[str] = None
    page_ids: List[str] = []


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None


# ==========================================
# REST API Endpoints
# ==========================================
@app.post("/api/scan/upload")
async def upload_scan_image(file: UploadFile = File(...)):
    """Uploads a raw captured document photo and detects its 4 paper boundary corners."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    filename = f"{file_id}{ext}"
    dest_path = UPLOAD_DIR / filename

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # Read image with OpenCV
    img_bgr = cv2.imread(str(dest_path))
    if img_bgr is None:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not decode image.")

    h, w = img_bgr.shape[:2]
    detected_corners = detect_document_corners(img_bgr)

    return {
        "upload_id": filename,
        "width": w,
        "height": h,
        "corners": detected_corners,
        "url": f"/files/uploads/{filename}",
    }


@app.post("/api/scan/warp-filter")
async def warp_and_filter_document(payload: WarpFilterRequest):
    """Warps quadrilateral region into a flat document and applies CamScanner enhancement filters."""
    src_file = UPLOAD_DIR / payload.upload_id
    if not src_file.exists():
        raise HTTPException(status_code=404, detail="Upload not found.")

    img_bgr = cv2.imread(str(src_file))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Could not read source upload.")

    corners_dict = payload.corners.model_dump()
    warped = warp_document(img_bgr, corners_dict)
    enhanced = apply_camscanner_filter(
        warped,
        filter_type=payload.filter_type,
        brightness=payload.brightness,
        contrast=payload.contrast,
    )

    page_id = f"page_{uuid.uuid4().hex[:12]}"
    page_filename = f"{page_id}.jpg"
    thumb_filename = f"{page_id}_thumb.jpg"

    page_path = PAGE_DIR / page_filename
    thumb_path = THUMB_DIR / thumb_filename

    # Save high-res page
    cv2.imwrite(str(page_path), enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Save fast thumbnail (height 320)
    th_h, th_w = enhanced.shape[:2]
    scale = 320.0 / th_h
    thumb_img = cv2.resize(enhanced, (int(th_w * scale), 320), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(thumb_path), thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return {
        "page_id": page_id,
        "width": enhanced.shape[1],
        "height": enhanced.shape[0],
        "page_url": f"/files/pages/{page_filename}",
        "thumbnail_url": f"/files/thumbnails/{thumb_filename}",
        "filter": payload.filter_type,
    }


@app.post("/api/scan/ocr/{page_id}")
async def ocr_page(page_id: str):
    """Performs offline OCR on a scanned page and returns structured text lines."""
    page_path = PAGE_DIR / f"{page_id}.jpg"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found.")

    if not HAS_OCR or ocr_engine is None:
        return {
            "success": False,
            "text": "OCR engine is currently disabled or unavailable.",
            "lines": [],
        }

    try:
        out = ocr_engine(str(page_path))
        lines = []
        boxes = []
        scores = []

        if hasattr(out, "txts") and out.txts:
            lines = list(out.txts)
            if hasattr(out, "boxes") and out.boxes is not None:
                boxes = [b.tolist() if hasattr(b, "tolist") else b for b in out.boxes]
            if hasattr(out, "scores") and out.scores is not None:
                scores = [float(s) for s in out.scores]
        elif isinstance(out, tuple) and out[0]:
            lines = [r[1] for r in out[0]]
            boxes = [r[0] for r in out[0]]
            scores = [float(r[2]) for r in out[0]]
        elif isinstance(out, list):
            lines = [r[1] for r in out]
            boxes = [r[0] for r in out]
            scores = [float(r[2]) for r in out]

        full_text = "\n".join(lines)
        return {
            "success": True,
            "text": full_text,
            "lines": lines,
            "boxes": boxes,
            "scores": scores,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "text": "", "lines": [], "boxes": []}


@app.get("/api/documents")
async def list_documents(query: Optional[str] = None):
    """Lists all saved documents, sorted by most recently updated."""
    db = load_db()
    docs = db.get("documents", [])
    if query:
        q = query.lower()
        docs = [d for d in docs if q in d.get("title", "").lower()]
    docs.sort(key=lambda d: d.get("updated_at", 0), reverse=True)
    return {"documents": docs}


@app.post("/api/documents")
async def create_document(payload: DocumentCreateRequest):
    """Creates a new document collection with initial pages."""
    db = load_db()
    now_ts = int(datetime.now().timestamp() * 1000)
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"

    title = payload.title or f"Doc {datetime.now().strftime('%Y-%m-%d %H%M')}"

    pages = []
    for pid in payload.page_ids:
        page_file = f"{pid}.jpg"
        thumb_file = f"{pid}_thumb.jpg"
        pages.append(
            {
                "id": pid,
                "url": f"/files/pages/{page_file}",
                "thumbnail_url": f"/files/thumbnails/{thumb_file}",
            }
        )

    doc = {
        "id": doc_id,
        "title": title,
        "created_at": now_ts,
        "updated_at": now_ts,
        "pages": pages,
    }

    db.setdefault("documents", []).append(doc)
    save_db(db)
    return doc


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Gets details and pages of a document."""
    db = load_db()
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            return d
    raise HTTPException(status_code=404, detail="Document not found.")


@app.put("/api/documents/{doc_id}")
async def update_document(doc_id: str, payload: DocumentUpdateRequest):
    """Renames or updates document title."""
    db = load_db()
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            if payload.title:
                d["title"] = payload.title.strip()
            d["updated_at"] = int(datetime.now().timestamp() * 1000)
            save_db(db)
            return d
    raise HTTPException(status_code=404, detail="Document not found.")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes a document and its pages from disk."""
    db = load_db()
    docs = db.get("documents", [])
    target = None
    for d in docs:
        if d["id"] == doc_id:
            target = d
            break

    if not target:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove pages & thumbnails
    for page in target.get("pages", []):
        pid = page.get("id")
        if pid:
            (PAGE_DIR / f"{pid}.jpg").unlink(missing_ok=True)
            (THUMB_DIR / f"{pid}_thumb.jpg").unlink(missing_ok=True)

    # Remove PDF
    (PDF_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)

    db["documents"] = [d for d in docs if d["id"] != doc_id]
    save_db(db)
    return {"success": True, "deleted_id": doc_id}


@app.post("/api/documents/{doc_id}/pages")
async def add_page_to_document(doc_id: str, request: Request):
    """Appends an already processed page to a document."""
    pid = None
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        data = await request.json()
        pid = data.get("page_id")
    else:
        form = await request.form()
        pid = form.get("page_id")

    if not pid:
        raise HTTPException(status_code=422, detail="Missing page_id")

    db = load_db()
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            page_file = f"{pid}.jpg"
            thumb_file = f"{pid}_thumb.jpg"
            d.setdefault("pages", []).append(
                {
                    "id": pid,
                    "url": f"/files/pages/{page_file}",
                    "thumbnail_url": f"/files/thumbnails/{thumb_file}",
                }
            )
            d["updated_at"] = int(datetime.now().timestamp() * 1000)
            save_db(db)
            return d
    raise HTTPException(status_code=404, detail="Document not found.")


@app.delete("/api/documents/{doc_id}/pages/{page_id}")
async def delete_page_from_document(doc_id: str, page_id: str):
    """Removes a page from a document."""
    db = load_db()
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            pages = d.get("pages", [])
            d["pages"] = [p for p in pages if p["id"] != page_id]
            d["updated_at"] = int(datetime.now().timestamp() * 1000)

            (PAGE_DIR / f"{page_id}.jpg").unlink(missing_ok=True)
            (THUMB_DIR / f"{page_id}_thumb.jpg").unlink(missing_ok=True)

            save_db(db)
            return d
    raise HTTPException(status_code=404, detail="Document not found.")


def create_dual_layer_searchable_pdf(page_paths: List[Path], out_pdf_path: Path):
    """
    Creates a Dual-Layer Searchable PDF:
    - Visible Layer: High-contrast raster document image
    - Invisible Layer: Selectable text at matching OCR bounding boxes with text render mode 3
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        pil_images = [Image.open(p).convert("RGB") for p in page_paths if p.exists()]
        if pil_images:
            first = pil_images[0]
            rest = pil_images[1:] if len(pil_images) > 1 else []
            first.save(str(out_pdf_path), "PDF", resolution=150.0, save_all=True, append_images=rest)
            return

    c = canvas.Canvas(str(out_pdf_path))
    a4_w, a4_h = A4

    for p_path in page_paths:
        if not p_path.exists():
            continue
        try:
            with Image.open(p_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        scale = min(a4_w / img_w, a4_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        margin_x = (a4_w - draw_w) / 2.0
        margin_y = (a4_h - draw_h) / 2.0

        c.setPageSize(A4)
        c.drawImage(str(p_path), margin_x, margin_y, width=draw_w, height=draw_h)

        if HAS_OCR and ocr_engine is not None:
            try:
                out = ocr_engine(str(p_path))
                lines = []
                boxes = []
                if hasattr(out, "txts") and out.txts:
                    lines = list(out.txts)
                    if hasattr(out, "boxes") and out.boxes is not None:
                        boxes = [b.tolist() if hasattr(b, "tolist") else b for b in out.boxes]
                elif isinstance(out, tuple) and out[0]:
                    lines = [r[1] for r in out[0]]
                    boxes = [r[0] for r in out[0]]

                for line_str, box in zip(lines, boxes):
                    if not line_str or not box or len(box) < 4:
                        continue
                    bx0 = min(pt[0] for pt in box)
                    by0 = min(pt[1] for pt in box)
                    bx1 = max(pt[0] for pt in box)
                    by1 = max(pt[1] for pt in box)
                    box_h = by1 - by0

                    pdf_x = margin_x + (bx0 * scale)
                    pdf_y = margin_y + ((img_h - by1) * scale)
                    font_size = max(5.0, (box_h * scale) * 0.85)

                    text_obj = c.beginText()
                    text_obj.setTextRenderMode(3)  # Invisible searchable text
                    text_obj.setFont("Helvetica", font_size)
                    text_obj.setTextOrigin(pdf_x, pdf_y)
                    text_obj.textLine(line_str)
                    c.drawText(text_obj)
            except Exception as ocr_err:
                print(f"[Warning] Dual-layer OCR pass skipped for page {p_path.name}: {ocr_err}")

        c.showPage()

    c.save()


@app.get("/api/documents/{doc_id}/export-pdf")
async def export_document_pdf(doc_id: str, download: bool = False):
    """Compiles all pages into a crisp standardized A4 dual-layer searchable PDF."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target or not target.get("pages"):
        raise HTTPException(status_code=404, detail="Document or pages not found.")

    pdf_filename = f"{doc_id}.pdf"
    out_pdf_path = PDF_DIR / pdf_filename

    page_paths = [PAGE_DIR / f"{page['id']}.jpg" for page in target["pages"]]
    page_paths = [p for p in page_paths if p.exists()]

    if not page_paths:
        raise HTTPException(status_code=400, detail="No readable page images found on disk.")

    create_dual_layer_searchable_pdf(page_paths, out_pdf_path)

    clean_title = target["title"].replace(" ", "_")
    download_name = f"{clean_title}.pdf"

    return FileResponse(
        path=out_pdf_path,
        media_type="application/pdf",
        filename=download_name,
        headers={"Content-Disposition": f"{'attachment' if download else 'inline'}; filename={download_name}"},
    )


@app.get("/api/documents/{doc_id}/export-txt")
async def export_document_txt(doc_id: str):
    """Extracts OCR text from all pages and returns as a downloadable text document."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target or not target.get("pages"):
        raise HTTPException(status_code=404, detail="Document or pages not found.")

    text_sections = []
    text_sections.append(f"=== {target.get('title', 'Document')} ===")
    text_sections.append(f"Category: {target.get('category', 'General')}")
    text_sections.append(f"Pages: {len(target.get('pages', []))}\n")

    for idx, page in enumerate(target.get("pages", [])):
        p_path = PAGE_DIR / f"{page['id']}.jpg"
        text_sections.append(f"--- [Page {idx + 1}] ---")
        if p_path.exists() and HAS_OCR and ocr_engine is not None:
            try:
                out = ocr_engine(str(p_path))
                if hasattr(out, "txts") and out.txts:
                    text_sections.extend(list(out.txts))
                elif isinstance(out, tuple) and out[0]:
                    text_sections.extend([r[1] for r in out[0]])
                else:
                    text_sections.append("[No text detected on this page]")
            except Exception as e:
                text_sections.append(f"[OCR Error: {e}]")
        else:
            text_sections.append("[Page unavailable or OCR disabled]")
        text_sections.append("")

    full_body = "\n".join(text_sections)
    clean_title = target["title"].replace(" ", "_")
    return Response(
        content=full_body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={clean_title}_ocr.txt"},
    )


@app.get("/api/documents/{doc_id}/export-docx")
async def export_document_docx(doc_id: str):
    """Extracts OCR text from all pages and generates a formatted Microsoft Word (.docx) document."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target or not target.get("pages"):
        raise HTTPException(status_code=404, detail="Document or pages not found.")

    import docx
    from docx.shared import Pt, Inches, RGBColor

    doc = docx.Document()
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(target.get("title", "Scanned Document"))
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0, 183, 150)

    meta_p = doc.add_paragraph(f"Category: {target.get('category', 'General')} | Pages: {len(target.get('pages', []))} | Generated by CamScanner Pro")
    meta_p.runs[0].font.size = Pt(9.5)
    meta_p.runs[0].font.italic = True

    for idx, page in enumerate(target.get("pages", [])):
        p_path = PAGE_DIR / f"{page['id']}.jpg"
        doc.add_heading(f"Page {idx + 1}", level=2)

        if p_path.exists():
            try:
                doc.add_picture(str(p_path), width=Inches(3.5))
            except Exception:
                pass

        if p_path.exists() and HAS_OCR and ocr_engine is not None:
            try:
                out = ocr_engine(str(p_path))
                lines = []
                if hasattr(out, "txts") and out.txts:
                    lines = list(out.txts)
                elif isinstance(out, tuple) and out[0]:
                    lines = [r[1] for r in out[0]]

                if lines:
                    doc.add_paragraph()
                    for line in lines:
                        p = doc.add_paragraph(line)
                        p.paragraph_format.line_spacing = 1.15
                        p.paragraph_format.space_after = Pt(3)
                else:
                    doc.add_paragraph("[No text detected on this page]").runs[0].font.italic = True
            except Exception as e:
                doc.add_paragraph(f"[OCR extraction note: {e}]")
        if idx < len(target.get("pages", [])) - 1:
            doc.add_page_break()

    out_docx_path = PDF_DIR / f"{doc_id}_export.docx"
    doc.save(str(out_docx_path))
    clean_title = target["title"].replace(" ", "_")
    return FileResponse(
        path=out_docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{clean_title}.docx"
    )


@app.get("/api/documents/{doc_id}/export-zip")
async def export_document_zip(doc_id: str):
    """Packages all high-resolution page scans into a downloadable ZIP archive."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target or not target.get("pages"):
        raise HTTPException(status_code=404, detail="Document or pages not found.")

    out_zip_path = PDF_DIR / f"{doc_id}_pages.zip"
    with zipfile.ZipFile(str(out_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, page in enumerate(target.get("pages", [])):
            p_path = PAGE_DIR / f"{page['id']}.jpg"
            if p_path.exists():
                clean_name = f"page_{idx + 1:03d}_{page['id'][:8]}.jpg"
                zf.write(str(p_path), arcname=clean_name)

    clean_title = target["title"].replace(" ", "_")
    return FileResponse(
        path=out_zip_path,
        media_type="application/zip",
        filename=f"{clean_title}_scans.zip"
    )


class WatermarkPayload(BaseModel):
    watermark_text: str = "CONFIDENTIAL"
    opacity: float = 0.25
    color: str = "gray"


@app.post("/api/documents/{doc_id}/watermark")
async def apply_document_watermark(doc_id: str, payload: WatermarkPayload):
    """Applies a repeating 45-degree diagonal anti-theft watermark across all document pages."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target or not target.get("pages"):
        raise HTTPException(status_code=404, detail="Document not found.")

    alpha = int(np.clip(payload.opacity * 255, 20, 240))
    colors = {
        "gray": (128, 128, 128, alpha),
        "red": (220, 38, 38, alpha),
        "green": (0, 183, 150, alpha),
        "blue": (37, 99, 235, alpha)
    }
    fill_color = colors.get(payload.color.lower(), (128, 128, 128, alpha))

    for page in target["pages"]:
        p_path = PAGE_DIR / f"{page['id']}.jpg"
        if not p_path.exists():
            continue

        pil_img = Image.open(p_path).convert("RGBA")
        txt_overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
        w, h = pil_img.size
        font_size = max(24, int(min(w, h) * 0.05))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        stamp_w = int(font_size * max(4, len(payload.watermark_text)) * 1.2)
        stamp_h = int(font_size * 2.5)
        stamp_img = Image.new("RGBA", (stamp_w, stamp_h), (255, 255, 255, 0))
        s_draw = ImageDraw.Draw(stamp_img)
        s_draw.text((10, 10), payload.watermark_text, font=font, fill=fill_color)
        rotated_stamp = stamp_img.rotate(35, expand=1, resample=Image.BICUBIC)

        step_x = max(100, rotated_stamp.width + 60)
        step_y = max(100, rotated_stamp.height + 60)
        for y in range(0, h + step_y, step_y):
            for x in range(0, w + step_x, step_x):
                txt_overlay.paste(rotated_stamp, (x - 40, y - 40), rotated_stamp)

        watermarked = Image.alpha_composite(pil_img, txt_overlay).convert("RGB")
        watermarked.save(p_path, "JPEG", quality=92)

        thumb_path = THUMB_DIR / f"{page['id']}_thumb.jpg"
        thumb = watermarked.copy()
        thumb.thumbnail((240, 340))
        thumb.save(thumb_path, "JPEG", quality=85)

    target["updated_at"] = int(datetime.now().timestamp() * 1000)
    save_db(db)
    return {"status": "success", "message": f"Applied watermark '{payload.watermark_text}'", "doc": target}


class SignaturePayload(BaseModel):
    page_id: str
    signature_data: str
    pos_x_pct: float = 0.60
    pos_y_pct: float = 0.75
    scale_pct: float = 0.30


@app.post("/api/documents/{doc_id}/signature")
async def apply_document_signature(doc_id: str, payload: SignaturePayload):
    """Composites an electronic signature onto a specific page of the document."""
    db = load_db()
    target = None
    for d in db.get("documents", []):
        if d["id"] == doc_id:
            target = d
            break

    if not target:
        raise HTTPException(status_code=404, detail="Document not found.")

    p_path = PAGE_DIR / f"{payload.page_id}.jpg"
    if not p_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found.")

    sig_raw = payload.signature_data
    if "base64," in sig_raw:
        sig_raw = sig_raw.split("base64,")[1]

    sig_bytes = base64.b64decode(sig_raw)
    sig_img = Image.open(io.BytesIO(sig_bytes)).convert("RGBA")

    page_img = Image.open(p_path).convert("RGBA")
    pw, ph = page_img.size

    sig_w = max(100, int(pw * payload.scale_pct))
    aspect = sig_img.height / max(1, sig_img.width)
    sig_h = int(sig_w * aspect)
    sig_resized = sig_img.resize((sig_w, sig_h), Image.LANCZOS)

    pos_x = int(pw * payload.pos_x_pct)
    pos_y = int(ph * payload.pos_y_pct)
    pos_x = max(0, min(pos_x, pw - sig_w))
    pos_y = max(0, min(pos_y, ph - sig_h))

    page_img.paste(sig_resized, (pos_x, pos_y), sig_resized)
    final_page = page_img.convert("RGB")
    final_page.save(p_path, "JPEG", quality=94)

    thumb_path = THUMB_DIR / f"{payload.page_id}_thumb.jpg"
    thumb = final_page.copy()
    thumb.thumbnail((240, 340))
    thumb.save(thumb_path, "JPEG", quality=85)

    target["updated_at"] = int(datetime.now().timestamp() * 1000)
    save_db(db)
    return {"status": "success", "message": "Signature placed successfully!", "doc": target}


@app.post("/api/scan/batch-upload")
async def batch_upload_and_create_document(
    files: List[UploadFile] = File(...),
    title: Optional[str] = Form("Batch Scan"),
    category: Optional[str] = Form("General"),
    filter_type: Optional[str] = Form("magic")
):
    """Processes multiple uploaded images into a multi-page document automatically."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    created_pages = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        corners = detect_document_corners(img)
        warped = warp_document(img, corners)
        processed = apply_camscanner_filter(warped, filter_type=filter_type)

        page_id = f"page_{uuid.uuid4().hex[:12]}"
        page_filename = f"{page_id}.jpg"
        thumb_filename = f"{page_id}_thumb.jpg"

        cv2.imwrite(str(PAGE_DIR / page_filename), processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
        thumb = cv2.resize(processed, (200, int(200 * (processed.shape[0] / max(1, processed.shape[1])))))
        cv2.imwrite(str(THUMB_DIR / thumb_filename), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

        created_pages.append({
            "id": page_id,
            "url": f"/files/pages/{page_filename}",
            "thumbnail_url": f"/files/thumbnails/{thumb_filename}",
        })

    if not created_pages:
        raise HTTPException(status_code=400, detail="Failed to process any valid pages from uploaded files.")

    db = load_db()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    doc = {
        "id": doc_id,
        "title": title or f"Batch Scan ({len(created_pages)} pages)",
        "category": category or "General",
        "created_at": int(datetime.now().timestamp() * 1000),
        "updated_at": int(datetime.now().timestamp() * 1000),
        "pages": created_pages
    }
    db.setdefault("documents", []).insert(0, doc)
    save_db(db)
    return doc


class IDCardPayload(BaseModel):
    front_page_id: str
    back_page_id: str
    title: Optional[str] = "ID Card (Front & Back)"
    category: Optional[str] = "ID Card"


@app.post("/api/scan/id-card")
async def create_id_card_composite(payload: IDCardPayload):
    """Composes Front and Back ID cards scaled 1:1 onto a single 300 DPI A4 page."""
    front_path = PAGE_DIR / f"{payload.front_page_id}.jpg"
    back_path = PAGE_DIR / f"{payload.back_page_id}.jpg"

    if not front_path.exists() or not back_path.exists():
        raise HTTPException(status_code=404, detail="Front or back card page not found.")

    front_img = cv2.imread(str(front_path))
    back_img = cv2.imread(str(back_path))

    a4_w, a4_h = 2480, 3508
    composite = np.full((a4_h, a4_w, 3), 255, dtype=np.uint8)

    card_w, card_h = 1011, 638
    front_scaled = cv2.resize(front_img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)
    back_scaled = cv2.resize(back_img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)

    x_offset = (a4_w - card_w) // 2
    y_front = 550
    y_back = 1950

    composite[y_front:y_front + card_h, x_offset:x_offset + card_w] = front_scaled
    composite[y_back:y_back + card_h, x_offset:x_offset + card_w] = back_scaled

    cv2.rectangle(composite, (x_offset - 6, y_front - 6), (x_offset + card_w + 6, y_front + card_h + 6), (220, 220, 220), 2)
    cv2.rectangle(composite, (x_offset - 6, y_back - 6), (x_offset + card_w + 6, y_back + card_h + 6), (220, 220, 220), 2)
    cv2.putText(composite, "CARD FRONT", (x_offset, y_front - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    cv2.putText(composite, "CARD BACK", (x_offset, y_back - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    cv2.putText(composite, "CamScanner Pro - ID Card 1:1 Scale", (x_offset, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (24, 154, 128), 3)

    composite_id = f"id_card_{uuid.uuid4().hex[:10]}"
    comp_page_file = f"{composite_id}.jpg"
    comp_thumb_file = f"{composite_id}_thumb.jpg"

    cv2.imwrite(str(PAGE_DIR / comp_page_file), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    thumb = cv2.resize(composite, (240, 340), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(THUMB_DIR / comp_thumb_file), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    db = load_db()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    doc = {
        "id": doc_id,
        "title": payload.title,
        "category": payload.category or "ID Card",
        "created_at": int(datetime.now().timestamp() * 1000),
        "updated_at": int(datetime.now().timestamp() * 1000),
        "pages": [
            {
                "id": composite_id,
                "url": f"/files/pages/{comp_page_file}",
                "thumbnail_url": f"/files/thumbnails/{comp_thumb_file}",
            }
        ],
    }
    db.setdefault("documents", []).insert(0, doc)
    save_db(db)
    return doc


@app.post("/api/scan/id-card-files")
async def create_id_card_from_files(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    title: Optional[str] = Form("ID Card (Front & Back)"),
):
    """Composes uploaded Front and Back ID photos directly into a 1:1 scale A4 composite document."""
    front_bytes = await front.read()
    back_bytes = await back.read()

    front_np = np.frombuffer(front_bytes, np.uint8)
    back_np = np.frombuffer(back_bytes, np.uint8)
    front_img = cv2.imdecode(front_np, cv2.IMREAD_COLOR)
    back_img = cv2.imdecode(back_np, cv2.IMREAD_COLOR)

    if front_img is None or back_img is None:
        raise HTTPException(status_code=400, detail="Invalid image file(s) for front or back ID.")

    a4_w, a4_h = 2480, 3508
    composite = np.full((a4_h, a4_w, 3), 255, dtype=np.uint8)

    card_w, card_h = 1011, 638
    front_scaled = cv2.resize(front_img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)
    back_scaled = cv2.resize(back_img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)

    x_offset = (a4_w - card_w) // 2
    y_front = 550
    y_back = 1950

    composite[y_front:y_front + card_h, x_offset:x_offset + card_w] = front_scaled
    composite[y_back:y_back + card_h, x_offset:x_offset + card_w] = back_scaled

    cv2.rectangle(composite, (x_offset - 6, y_front - 6), (x_offset + card_w + 6, y_front + card_h + 6), (220, 220, 220), 2)
    cv2.rectangle(composite, (x_offset - 6, y_back - 6), (x_offset + card_w + 6, y_back + card_h + 6), (220, 220, 220), 2)
    cv2.putText(composite, "CARD FRONT", (x_offset, y_front - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    cv2.putText(composite, "CARD BACK", (x_offset, y_back - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    cv2.putText(composite, "CamScanner Pro - ID Card 1:1 Scale", (x_offset, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (24, 154, 128), 3)

    composite_id = f"id_card_{uuid.uuid4().hex[:10]}"
    comp_page_file = f"{composite_id}.jpg"
    comp_thumb_file = f"{composite_id}_thumb.jpg"

    cv2.imwrite(str(PAGE_DIR / comp_page_file), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    thumb = cv2.resize(composite, (240, 340), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(THUMB_DIR / comp_thumb_file), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    db = load_db()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    doc = {
        "id": doc_id,
        "title": title or "ID Card (Front & Back)",
        "category": "ID Card",
        "created_at": int(datetime.now().timestamp() * 1000),
        "updated_at": int(datetime.now().timestamp() * 1000),
        "pages": [
            {
                "id": composite_id,
                "url": f"/files/pages/{comp_page_file}",
                "thumbnail_url": f"/files/thumbnails/{comp_thumb_file}",
            }
        ],
    }
    db.setdefault("documents", []).insert(0, doc)
    save_db(db)
    return doc


# ==========================================
# Frontend Dashboard (CamScanner Mobile & Web UI)
# ==========================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100 select-none">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <meta name="theme-color" content="#00b796">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" type="image/png" href="/static/favicon.png">
  <link rel="apple-touch-icon" href="/static/logo_192.png">
  <title>CamScanner Pro</title>

  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            brand: {
              DEFAULT: '#00b796',
              dark: '#00977c',
              light: '#19bcaa',
              surface: '#15171c',
              card: '#1e222a',
              bg: '#0f1115'
            }
          }
        }
      }
    }
  </script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>

  <style>
    :root {
      --brand: #00b796;
    }
    body {
      -webkit-tap-highlight-color: transparent;
      user-select: none;
      touch-action: manipulation;
    }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    #crop-workspace {
      background: radial-gradient(circle at center, #1e222c 0%, #0a0b0e 100%);
    }
    #crop-canvas {
      touch-action: none;
      cursor: crosshair;
    }
    #magnifier-loupe {
      box-shadow: 0 10px 30px -5px rgba(0,0,0,0.8), 0 0 0 3px rgba(0, 183, 150, 0.7);
      pointer-events: none;
      transform: translate(-50%, -100%);
      transition: opacity 0.15s ease-out;
    }
    .filter-pill.active {
      border-color: #00b796 !important;
      background-color: rgba(0, 183, 150, 0.15) !important;
    }
    .filter-pill.active span {
      color: #19bcaa !important;
      font-weight: 700;
    }
    input[type="range"] {
      height: 4px; border-radius: 2px; background: #334155; outline: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none;
      width: 16px; height: 16px; border-radius: 50%;
      background: #00b796; cursor: pointer; border: 2px solid white;
    }
  </style>
</head>
<body class="h-full flex flex-col font-sans overflow-hidden bg-brand-bg text-slate-100">

  <!-- ============================================== -->
  <!-- 1. HOME VIEW: DOCUMENT LIST -->
  <!-- ============================================== -->
  <div id="view-home" class="view flex flex-col h-full w-full">
    <!-- Header -->
    <header class="bg-brand-surface border-b border-slate-800 px-4 py-3 flex items-center justify-between shadow-md">
      <div class="flex items-center space-x-2.5">
        <img src="/static/logo_192.png" alt="CamScanner Logo" class="w-9 h-9 rounded-xl shadow-md border border-slate-700/60 object-contain p-0.5 bg-slate-900">
        <div>
          <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
            CamScanner <span class="text-[10px] px-1.5 py-0.5 rounded bg-brand/20 text-brand-light font-semibold border border-brand/30">PRO</span>
          </h1>
          <p id="doc-counter" class="text-xs text-slate-400">Loading documents...</p>
        </div>
      </div>
      <div class="flex items-center space-x-1">
        <button id="pwa-install-btn" onclick="installPWA()" class="hidden px-2.5 py-1 text-xs bg-brand hover:bg-brand-dark text-white font-medium rounded-full shadow flex items-center gap-1 transition mr-1">
          <i data-lucide="download" class="w-3.5 h-3.5"></i>
          <span>Install</span>
        </button>
        <button id="btn-search-toggle" class="p-2 text-slate-300 hover:text-white rounded-full hover:bg-slate-800 transition">
          <i data-lucide="search" class="w-5 h-5"></i>
        </button>
        <button id="btn-refresh" class="p-2 text-slate-300 hover:text-white rounded-full hover:bg-slate-800 transition">
          <i data-lucide="refresh-cw" class="w-5 h-5"></i>
        </button>
      </div>
    </header>

    <!-- Search Input -->
    <div id="search-container" class="hidden px-4 py-2 bg-brand-surface border-b border-slate-800">
      <div class="relative">
        <i data-lucide="search" class="w-4 h-4 absolute left-3 top-2.5 text-slate-400"></i>
        <input type="text" id="input-search" placeholder="Search scanned documents..." class="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-brand">
      </div>
    </div>

    <!-- Category Pills -->
    <div class="px-4 py-2.5 flex space-x-2 overflow-x-auto no-scrollbar border-b border-slate-900 bg-brand-bg/50" id="category-pills">
      <button class="cat-pill active px-3 py-1 rounded-full text-xs font-semibold bg-brand text-white shadow-sm transition" onclick="filterCategory('All', this)">All Docs</button>
      <button class="cat-pill px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 text-slate-300 hover:bg-slate-800 transition" onclick="filterCategory('Invoice', this)">Invoices</button>
      <button class="cat-pill px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 text-brand-light hover:bg-slate-800 flex items-center gap-1 transition" onclick="openIDCardModal()">
        <i data-lucide="credit-card" class="w-3.5 h-3.5"></i>
        <span>ID Card</span>
      </button>
      <button class="cat-pill px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 text-slate-300 hover:bg-slate-800 transition" onclick="filterCategory('Book', this)">Books</button>
      <button class="cat-pill px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 text-slate-300 hover:bg-slate-800 transition" onclick="filterCategory('Note', this)">Notes</button>
    </div>

    <!-- Document List Container -->
    <main class="flex-1 overflow-y-auto px-4 py-3 space-y-3" id="doc-list-container">
      <div id="empty-state" class="hidden flex flex-col items-center justify-center h-full text-center py-16 px-4">
        <img src="/static/official/ic_scanner_empty_doc-BWsAcaxT.png" alt="No Documents" class="w-36 h-36 object-contain mb-3 opacity-90 drop-shadow-md">
        <h3 class="text-base font-semibold text-slate-200">No Scanned Documents</h3>
        <p class="text-xs text-slate-400 mt-1 max-w-xs">
          Tap the emerald camera button below to take a photo or select an image from your gallery.
        </p>
      </div>
    </main>

    <!-- Floating Action Button (CamScanner Signature Speed-Dial) -->
    <div class="fixed bottom-6 right-6 z-20 flex flex-col items-end gap-3">
      <button id="btn-batch" onclick="triggerBatchScan()" class="w-12 h-12 rounded-full bg-slate-800 border border-slate-700 text-purple-400 shadow-xl flex items-center justify-center hover:bg-slate-700 active:scale-95 transition" title="Batch Multi-Page Scan">
        <i data-lucide="layers" class="w-5 h-5"></i>
      </button>

      <button id="btn-idcard" onclick="openIDCardModal()" class="w-12 h-12 rounded-full bg-slate-800 border border-slate-700 text-brand-light shadow-xl flex items-center justify-center hover:bg-slate-700 active:scale-95 transition" title="ID Card Front & Back (1:1 A4)">
        <i data-lucide="credit-card" class="w-5 h-5"></i>
      </button>

      <button id="btn-gallery" class="w-12 h-12 rounded-full bg-slate-800 border border-slate-700 text-slate-200 shadow-xl flex items-center justify-center hover:bg-slate-700 active:scale-95 transition" title="Upload from Gallery">
        <i data-lucide="image" class="w-5 h-5"></i>
      </button>

      <button id="btn-camera" class="w-16 h-16 rounded-full bg-brand hover:bg-brand-light text-white shadow-2xl shadow-brand/40 flex items-center justify-center active:scale-95 transition" title="Start Scan">
        <i data-lucide="camera" class="w-8 h-8"></i>
      </button>
    </div>

    <!-- Hidden Native File Inputs -->
    <input type="file" id="camera-file-input" accept="image/*" capture="environment" class="hidden">
    <input type="file" id="gallery-file-input" accept="image/*" class="hidden">
    <input type="file" id="batch-file-input" accept="image/*" multiple class="hidden" onchange="handleBatchSelect(event)">
  </div>

  <!-- ============================================== -->
  <!-- 2. CROP & PERSPECTIVE ADJUSTMENT VIEW -->
  <!-- ============================================== -->
  <div id="view-crop" class="view hidden flex flex-col h-full w-full bg-black z-30">
    <header class="px-4 py-3 flex items-center justify-between bg-slate-900/90 backdrop-blur border-b border-slate-800">
      <button id="btn-crop-cancel" class="p-2 text-slate-300 hover:text-white">
        <i data-lucide="x" class="w-6 h-6"></i>
      </button>
      <h2 class="text-sm font-semibold text-white tracking-wide">Adjust Boundaries</h2>
      <button id="btn-crop-next" class="p-2 text-brand-light hover:text-brand font-bold flex items-center gap-1">
        <span>Next</span>
        <i data-lucide="arrow-right" class="w-5 h-5"></i>
      </button>
    </header>

    <div class="relative flex-1 flex items-center justify-center overflow-hidden touch-none" id="crop-workspace">
      <canvas id="crop-canvas" class="max-w-full max-h-full block shadow-2xl"></canvas>

      <!-- Magnifier Loupe Lens -->
      <div id="magnifier-loupe" class="hidden pointer-events-none absolute w-28 h-28 rounded-full border-2 border-brand bg-slate-950 shadow-2xl overflow-hidden z-40">
        <canvas id="loupe-canvas" class="w-full h-full"></canvas>
        <div class="absolute inset-0 flex items-center justify-center">
          <div class="w-full h-[1px] bg-brand/80"></div>
          <div class="h-full w-[1px] bg-brand/80 absolute"></div>
        </div>
      </div>
    </div>

    <footer class="bg-slate-900 border-t border-slate-800 px-4 py-3 flex items-center justify-around text-xs text-slate-300">
      <button id="btn-crop-auto" class="flex flex-col items-center gap-1 hover:text-brand-light transition">
        <i data-lucide="sparkles" class="w-5 h-5 text-brand"></i>
        <span>Auto Detect</span>
      </button>
      <button id="btn-crop-all" class="flex flex-col items-center gap-1 hover:text-white transition">
        <i data-lucide="maximize" class="w-5 h-5"></i>
        <span>Full Page</span>
      </button>
      <button id="btn-rotate-left" class="flex flex-col items-center gap-1 hover:text-white transition">
        <i data-lucide="rotate-ccw" class="w-5 h-5"></i>
        <span>Rotate Left</span>
      </button>
      <button id="btn-rotate-right" class="flex flex-col items-center gap-1 hover:text-white transition">
        <i data-lucide="rotate-cw" class="w-5 h-5"></i>
        <span>Rotate Right</span>
      </button>
    </footer>
  </div>

  <!-- ============================================== -->
  <!-- 3. FILTER & ENHANCEMENT VIEW -->
  <!-- ============================================== -->
  <div id="view-filter" class="view hidden flex flex-col h-full w-full bg-black z-30">
    <header class="px-4 py-3 flex items-center justify-between bg-slate-900/90 backdrop-blur border-b border-slate-800">
      <button id="btn-filter-back" class="p-2 text-slate-300 hover:text-white">
        <i data-lucide="arrow-left" class="w-6 h-6"></i>
      </button>
      <h2 class="text-sm font-semibold text-white tracking-wide">Enhancement Filters</h2>
      <button id="btn-filter-save" class="p-2 text-brand-light hover:text-brand font-bold flex items-center gap-1">
        <i data-lucide="check" class="w-6 h-6 text-brand"></i>
      </button>
    </header>

    <div class="relative flex-1 flex items-center justify-center p-3 overflow-hidden bg-slate-950">
      <img id="filter-preview-img" class="max-w-full max-h-full object-contain rounded shadow-2xl border border-slate-800">
      <div id="filter-loading" class="hidden absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center gap-2 text-brand-light text-sm font-medium">
        <i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i>
        <span>Processing with OpenCV...</span>
      </div>
    </div>

    <!-- Brightness & Contrast fine tuning -->
    <div class="bg-slate-900/95 border-t border-slate-800/80 px-6 py-2 flex items-center justify-between gap-4 text-xs text-slate-300">
      <div class="flex-1 flex items-center gap-2">
        <i data-lucide="sun" class="w-4 h-4 text-slate-400"></i>
        <input type="range" id="slider-brightness" min="-50" max="50" value="0" class="w-full accent-brand">
      </div>
      <div class="flex-1 flex items-center gap-2">
        <i data-lucide="contrast" class="w-4 h-4 text-slate-400"></i>
        <input type="range" id="slider-contrast" min="-50" max="50" value="0" class="w-full accent-brand">
      </div>
    </div>

    <!-- Filter Pills Scroller -->
    <footer class="bg-brand-surface border-t border-slate-800 p-3 pb-6 overflow-x-auto no-scrollbar">
      <div class="flex items-center space-x-3 justify-center min-w-max px-2" id="filter-pills-list">
        <!-- Magic Color -->
        <button data-filter="magic" class="filter-pill active flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-brand bg-brand/10 transition">
          <div class="w-12 h-12 rounded-lg bg-emerald-600 flex items-center justify-center text-white shadow">
            <i data-lucide="wand-2" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] font-semibold text-brand-light">Magic Color</span>
        </button>

        <!-- No Shadow (CamScanner Illumination Flattening) -->
        <button data-filter="no_shadow" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-teal-800 flex items-center justify-center text-white shadow">
            <i data-lucide="sun-medium" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">No Shadow</span>
        </button>

        <!-- B&W Clean -->
        <button data-filter="bw" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-slate-900 border border-white flex items-center justify-center text-white shadow">
            <span class="font-extrabold text-xs">B&W</span>
          </div>
          <span class="text-[11px] text-slate-300">B&W Clean</span>
        </button>

        <!-- Charcoal -->
        <button data-filter="bw2" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-neutral-800 flex items-center justify-center text-white shadow">
            <span class="font-extrabold text-xs text-neutral-300">B&W 2</span>
          </div>
          <span class="text-[11px] text-slate-300">Charcoal</span>
        </button>

        <!-- Grayscale -->
        <button data-filter="grayscale" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-gray-500 flex items-center justify-center text-white shadow">
            <i data-lucide="droplet" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">Grayscale</span>
        </button>

        <!-- Lighten -->
        <button data-filter="lighten" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-amber-500/80 flex items-center justify-center text-white shadow">
            <i data-lucide="sun-medium" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">Lighten</span>
        </button>

        <!-- Dewarp (Curved Book) -->
        <button data-filter="dewarp" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-teal-700 flex items-center justify-center text-white shadow">
            <i data-lucide="book-open" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">Dewarp Book</span>
        </button>

        <!-- Clean Print (Handwriting Removal) -->
        <button data-filter="clean_print" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-indigo-700 flex items-center justify-center text-white shadow">
            <i data-lucide="printer" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">Clean Print</span>
        </button>

        <!-- Original -->
        <button data-filter="original" class="filter-pill flex flex-col items-center gap-1.5 px-3 py-2 rounded-xl border border-transparent hover:border-slate-700 transition">
          <div class="w-12 h-12 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow">
            <i data-lucide="image" class="w-6 h-6"></i>
          </div>
          <span class="text-[11px] text-slate-300">Original</span>
        </button>
      </div>
    </footer>
  </div>

  <!-- ============================================== -->
  <!-- 4. DOCUMENT DETAIL & MULTI-PAGE VIEWER -->
  <!-- ============================================== -->
  <div id="view-doc" class="view hidden flex flex-col h-full w-full bg-brand-bg z-30">
    <header class="bg-brand-surface border-b border-slate-800 px-4 py-3 flex items-center justify-between shadow-md">
      <div class="flex items-center space-x-2">
        <button id="btn-doc-back" class="p-1.5 -ml-1 text-slate-300 hover:text-white">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <div class="flex flex-col">
          <span id="doc-title-text" class="text-sm font-bold text-white truncate max-w-[180px]">Document</span>
          <span id="doc-meta-text" class="text-[11px] text-slate-400">1 page</span>
        </div>
      </div>
      <div class="flex items-center space-x-1">
        <button id="btn-rename-doc" class="p-2 text-slate-300 hover:text-white" title="Rename Document">
          <i data-lucide="edit-3" class="w-5 h-5"></i>
        </button>
        <button id="btn-delete-doc" class="p-2 text-red-400 hover:text-red-300" title="Delete Document">
          <i data-lucide="trash-2" class="w-5 h-5"></i>
        </button>
      </div>
    </header>

    <!-- Document Quick Action Toolbar -->
    <div class="bg-slate-900 border-b border-slate-800 px-3 py-2 flex items-center justify-around gap-1 text-[11px] shadow-sm">
      <button onclick="openSignatureModal()" class="flex flex-col items-center gap-1 text-slate-300 hover:text-brand-light transition py-1 px-2 rounded-lg hover:bg-slate-800/60">
        <i data-lucide="pen-tool" class="w-4 h-4 text-brand"></i>
        <span>Sign</span>
      </button>
      <button onclick="openWatermarkModal()" class="flex flex-col items-center gap-1 text-slate-300 hover:text-amber-400 transition py-1 px-2 rounded-lg hover:bg-slate-800/60">
        <i data-lucide="shield-alert" class="w-4 h-4 text-amber-400"></i>
        <span>Watermark</span>
      </button>
      <button id="btn-export-docx" class="flex flex-col items-center gap-1 text-slate-300 hover:text-blue-400 transition py-1 px-2 rounded-lg hover:bg-slate-800/60">
        <i data-lucide="file-type-2" class="w-4 h-4 text-blue-400"></i>
        <span>Word</span>
      </button>
      <button id="btn-export-zip" class="flex flex-col items-center gap-1 text-slate-300 hover:text-purple-400 transition py-1 px-2 rounded-lg hover:bg-slate-800/60">
        <i data-lucide="archive" class="w-4 h-4 text-purple-400"></i>
        <span>ZIP</span>
      </button>
      <button id="btn-export-txt" class="flex flex-col items-center gap-1 text-slate-300 hover:text-emerald-400 transition py-1 px-2 rounded-lg hover:bg-slate-800/60">
        <i data-lucide="file-text" class="w-4 h-4 text-emerald-400"></i>
        <span>OCR</span>
      </button>
    </div>

    <main class="flex-1 overflow-y-auto p-4">
      <div id="doc-pages-grid" class="grid grid-cols-2 gap-4">
        <!-- Rendered dynamically -->
      </div>
    </main>

    <footer class="bg-brand-surface border-t border-slate-800 p-3 pb-6 flex items-center justify-between gap-3 shadow-lg">
      <button id="btn-add-page" class="flex-1 py-3 px-4 rounded-xl border border-brand/50 bg-brand/10 text-brand-light font-semibold text-xs flex items-center justify-center gap-2 hover:bg-brand/20 active:scale-98 transition">
        <i data-lucide="plus-circle" class="w-5 h-5"></i>
        <span>Add Page</span>
      </button>

      <button id="btn-export-pdf" class="flex-1 py-3 px-4 rounded-xl bg-brand text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-brand/30 hover:bg-brand-light active:scale-98 transition">
        <i data-lucide="file-down" class="w-5 h-5"></i>
        <span>Share Searchable PDF</span>
      </button>
    </footer>
  </div>

  <!-- OCR Result Modal -->
  <div id="modal-ocr" class="hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-brand-surface border border-slate-700 w-full max-w-lg rounded-2xl shadow-2xl flex flex-col max-h-[80vh] overflow-hidden">
      <div class="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <i data-lucide="file-text" class="w-4 h-4 text-brand"></i>
          <span>Extracted OCR Text</span>
        </h3>
        <button id="btn-close-ocr" class="text-slate-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>
      <div class="p-4 overflow-y-auto flex-1 font-mono text-xs text-slate-200 whitespace-pre-wrap select-text leading-relaxed bg-slate-900/60" id="ocr-text-content">
        Running OCR...
      </div>
      <div class="p-3 border-t border-slate-800 flex justify-end gap-2">
        <button id="btn-copy-ocr" class="px-4 py-2 rounded-xl bg-brand text-white text-xs font-semibold flex items-center gap-1.5 hover:bg-brand-light active:scale-95 transition">
          <i data-lucide="copy" class="w-4 h-4"></i>
          <span>Copy Text</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ID Card Scanner Modal -->
  <div id="modal-idcard" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-brand-surface border border-slate-700 w-full max-w-md rounded-2xl shadow-2xl flex flex-col overflow-hidden">
      <div class="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <i data-lucide="credit-card" class="w-4 h-4 text-brand-light"></i>
          <span>ID Card Scanner (1:1 Scale A4)</span>
        </h3>
        <button onclick="closeIDCardModal()" class="text-slate-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>

      <div class="p-4 space-y-3.5">
        <p class="text-xs text-slate-300">
          Combines Front and Back of your ID card into a standard ISO-7810 1:1 scale composite sheet, ready for printing.
        </p>

        <!-- Front Side -->
        <div>
          <span class="text-xs font-semibold text-slate-300 block mb-1">1. Card Front</span>
          <div id="idcard-front-box" onclick="document.getElementById('idcard-front-input').click()" class="h-28 border-2 border-dashed border-slate-700 hover:border-brand/70 rounded-xl bg-slate-900/60 flex flex-col items-center justify-center cursor-pointer overflow-hidden transition relative">
            <img id="idcard-front-preview" class="hidden w-full h-full object-contain">
            <div id="idcard-front-placeholder" class="flex flex-col items-center text-slate-400">
              <i data-lucide="camera" class="w-6 h-6 text-brand-light mb-1"></i>
              <span class="text-xs">Tap to upload / capture Front</span>
            </div>
          </div>
          <input type="file" id="idcard-front-input" accept="image/*" class="hidden" onchange="previewIDCardSide(event, 'front')">
        </div>

        <!-- Back Side -->
        <div>
          <span class="text-xs font-semibold text-slate-300 block mb-1">2. Card Back</span>
          <div id="idcard-back-box" onclick="document.getElementById('idcard-back-input').click()" class="h-28 border-2 border-dashed border-slate-700 hover:border-brand/70 rounded-xl bg-slate-900/60 flex flex-col items-center justify-center cursor-pointer overflow-hidden transition relative">
            <img id="idcard-back-preview" class="hidden w-full h-full object-contain">
            <div id="idcard-back-placeholder" class="flex flex-col items-center text-slate-400">
              <i data-lucide="camera" class="w-6 h-6 text-brand-light mb-1"></i>
              <span class="text-xs">Tap to upload / capture Back</span>
            </div>
          </div>
          <input type="file" id="idcard-back-input" accept="image/*" class="hidden" onchange="previewIDCardSide(event, 'back')">
        </div>

        <div>
          <label class="text-xs font-semibold text-slate-300 block mb-1">Document Title</label>
          <input type="text" id="idcard-title-input" value="ID Card (Front & Back)" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-brand">
        </div>
      </div>

      <div class="p-3 border-t border-slate-800 flex justify-end gap-2 bg-slate-900/40">
        <button onclick="closeIDCardModal()" class="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white">Cancel</button>
        <button id="btn-submit-idcard" onclick="submitIDCard()" class="px-4 py-2 rounded-xl bg-brand text-white text-xs font-semibold flex items-center gap-1.5 hover:bg-brand-light active:scale-95 transition disabled:opacity-50">
          <i data-lucide="layers" class="w-4 h-4"></i>
          <span>Create 1:1 Composite</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Watermark Modal -->
  <div id="modal-watermark" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-brand-surface border border-slate-700 w-full max-w-md rounded-2xl shadow-2xl flex flex-col overflow-hidden">
      <div class="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <i data-lucide="shield-alert" class="w-4 h-4 text-amber-400"></i>
          <span>Anti-Theft Watermark Protection</span>
        </h3>
        <button onclick="closeWatermarkModal()" class="text-slate-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>

      <div class="p-4 space-y-3.5">
        <p class="text-xs text-slate-300">
          Stamps diagonal repeating security watermarks across all document pages to prevent unauthorized duplication or reuse.
        </p>

        <div>
          <label class="text-xs font-semibold text-slate-300 block mb-1">Watermark Text</label>
          <input type="text" id="watermark-text-input" value="FOR VERIFICATION ONLY" placeholder="e.g. CONFIDENTIAL, VISA ONLY" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-brand">
        </div>

        <div>
          <div class="flex justify-between text-xs text-slate-300 mb-1">
            <span>Watermark Opacity</span>
            <span id="wm-opacity-label">25%</span>
          </div>
          <input type="range" id="watermark-opacity-slider" min="10" max="80" value="25" oninput="document.getElementById('wm-opacity-label').textContent = this.value + '%'" class="w-full accent-brand">
        </div>

        <div>
          <label class="text-xs font-semibold text-slate-300 block mb-1.5">Watermark Color</label>
          <div class="flex items-center gap-3">
            <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input type="radio" name="wm-color" value="gray" checked class="accent-brand"> Gray
            </label>
            <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input type="radio" name="wm-color" value="red" class="accent-red-500"> Red
            </label>
            <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input type="radio" name="wm-color" value="green" class="accent-emerald-500"> Green
            </label>
            <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input type="radio" name="wm-color" value="blue" class="accent-blue-500"> Blue
            </label>
          </div>
        </div>
      </div>

      <div class="p-3 border-t border-slate-800 flex justify-end gap-2 bg-slate-900/40">
        <button onclick="closeWatermarkModal()" class="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white">Cancel</button>
        <button id="btn-submit-watermark" onclick="submitWatermark()" class="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition active:scale-95">
          <i data-lucide="shield-check" class="w-4 h-4"></i>
          <span>Apply to All Pages</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Signature Modal -->
  <div id="modal-signature" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-brand-surface border border-slate-700 w-full max-w-md rounded-2xl shadow-2xl flex flex-col overflow-hidden">
      <div class="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <i data-lucide="pen-tool" class="w-4 h-4 text-brand-light"></i>
          <span>Electronic Signature Studio</span>
        </h3>
        <button onclick="closeSignatureModal()" class="text-slate-400 hover:text-white">
          <i data-lucide="x" class="w-5 h-5"></i>
        </button>
      </div>

      <div class="p-4 space-y-3">
        <p class="text-xs text-slate-300">
          Sign inside the box with your finger or stylus. It will be stamped onto the selected page with transparent background.
        </p>

        <!-- Ink Canvas -->
        <div class="border-2 border-slate-700 rounded-xl bg-white overflow-hidden relative touch-none">
          <canvas id="sig-pad-canvas" width="380" height="160" class="w-full h-40 block cursor-crosshair"></canvas>
          <span class="absolute bottom-1 right-2 text-[10px] text-slate-400 pointer-events-none select-none">Draw signature here</span>
        </div>

        <div class="flex items-center justify-between text-xs">
          <div class="flex items-center gap-2">
            <span class="text-slate-400">Ink:</span>
            <button onclick="setSigColor('#1e3a8a', this)" class="sig-color-btn w-5 h-5 rounded-full bg-blue-900 border-2 border-white"></button>
            <button onclick="setSigColor('#000000', this)" class="sig-color-btn w-5 h-5 rounded-full bg-black border-2 border-transparent"></button>
            <button onclick="setSigColor('#b91c1c', this)" class="sig-color-btn w-5 h-5 rounded-full bg-red-700 border-2 border-transparent"></button>
          </div>
          <button onclick="clearSignaturePad()" class="px-2.5 py-1 text-xs text-slate-400 hover:text-white border border-slate-700 rounded-lg">Clear</button>
        </div>

        <div class="flex items-center justify-between gap-3 text-xs">
          <div class="flex-1">
            <label class="text-slate-300 block mb-1">Target Page</label>
            <select id="sig-page-select" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-brand"></select>
          </div>
          <div class="flex-1">
            <label class="text-slate-300 block mb-1">Signature Position</label>
            <select id="sig-pos-select" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-brand">
              <option value="br">Bottom Right</option>
              <option value="bl">Bottom Left</option>
              <option value="bc">Bottom Center</option>
              <option value="tr">Top Right</option>
            </select>
          </div>
        </div>
      </div>

      <div class="p-3 border-t border-slate-800 flex justify-end gap-2 bg-slate-900/40">
        <button onclick="closeSignatureModal()" class="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white">Cancel</button>
        <button id="btn-submit-sig" onclick="submitSignature()" class="px-4 py-2 rounded-xl bg-brand text-white text-xs font-semibold flex items-center gap-1.5 hover:bg-brand-light active:scale-95 transition">
          <i data-lucide="check" class="w-4 h-4"></i>
          <span>Stamp Signature</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Frontend Logic -->
  <script>
    const state = {
      currentView: 'home',
      documents: [],
      activeDoc: null,
      isAddingPage: false,
      selectedCategory: 'All',
      idCardFiles: { front: null, back: null },

      // Current scan upload session
      uploadId: null,
      sourceWidth: 0,
      sourceHeight: 0,
      corners: { tl: {x:0, y:0}, tr: {x:0, y:0}, br: {x:0, y:0}, bl: {x:0, y:0} },
      displayScale: 1,
      sourceImg: new Image(),

      // Filter settings
      filterType: 'magic',
      brightness: 0,
      contrast: 0,
      processedPageId: null,

      // Interaction
      activeHandle: null
    };

    const views = {
      home: document.getElementById('view-home'),
      crop: document.getElementById('view-crop'),
      filter: document.getElementById('view-filter'),
      doc: document.getElementById('view-doc')
    };

    function switchView(viewName) {
      state.currentView = viewName;
      Object.keys(views).forEach(k => {
        views[k].classList.toggle('hidden', k !== viewName);
      });
      setTimeout(() => lucide.createIcons(), 50);
    }

    function filterCategory(category, elem) {
      state.selectedCategory = category;
      document.querySelectorAll('#category-pills .cat-pill').forEach(btn => {
        btn.classList.remove('active', 'bg-brand', 'text-white', 'shadow-sm');
        btn.classList.add('bg-slate-800/80', 'text-slate-300');
      });
      if (elem) {
        elem.classList.add('active', 'bg-brand', 'text-white', 'shadow-sm');
        elem.classList.remove('bg-slate-800/80', 'text-slate-300');
      }
      renderDocuments();
    }

    function openIDCardModal() {
      state.idCardFiles = { front: null, back: null };
      document.getElementById('idcard-front-input').value = '';
      document.getElementById('idcard-back-input').value = '';
      document.getElementById('idcard-front-preview').classList.add('hidden');
      document.getElementById('idcard-front-placeholder').classList.remove('hidden');
      document.getElementById('idcard-back-preview').classList.add('hidden');
      document.getElementById('idcard-back-placeholder').classList.remove('hidden');
      document.getElementById('modal-idcard').classList.remove('hidden');
      lucide.createIcons();
    }

    function closeIDCardModal() {
      document.getElementById('modal-idcard').classList.add('hidden');
    }

    function previewIDCardSide(event, side) {
      const file = event.target.files[0];
      if (!file) return;
      state.idCardFiles[side] = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        const preview = document.getElementById(`idcard-${side}-preview`);
        const placeholder = document.getElementById(`idcard-${side}-placeholder`);
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        placeholder.classList.add('hidden');
      };
      reader.readAsDataURL(file);
    }

    async function submitIDCard() {
      if (!state.idCardFiles.front || !state.idCardFiles.back) {
        alert('Please provide photos or scans for both Front and Back sides of the card.');
        return;
      }
      const title = document.getElementById('idcard-title-input').value.trim() || 'ID Card (Front & Back)';
      const btn = document.getElementById('btn-submit-idcard');
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Composing 1:1 A4...</span>';
      lucide.createIcons();

      const formData = new FormData();
      formData.append('front', state.idCardFiles.front);
      formData.append('back', state.idCardFiles.back);
      formData.append('title', title);

      try {
        const res = await fetch('/api/scan/id-card-files', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error(await res.text());
        const doc = await res.json();
        closeIDCardModal();
        await fetchDocuments();
        openDocumentDetail(doc);
      } catch (err) {
        alert('Failed to generate ID card composite: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="layers" class="w-4 h-4"></i><span>Create 1:1 Composite</span>';
        lucide.createIcons();
      }
    }

    // ==============================================
    // BATCH MULTI-PAGE SCAN
    // ==============================================
    function triggerBatchScan() {
      document.getElementById('batch-file-input').click();
    }

    async function handleBatchSelect(event) {
      const files = event.target.files;
      if (!files || files.length === 0) return;

      const btn = document.getElementById('btn-batch');
      const origHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i>';
      lucide.createIcons();

      const title = prompt('Document Title for Batch Scan:', `Batch Scan (${files.length} pages)`);
      if (!title) {
        btn.disabled = false;
        btn.innerHTML = origHtml;
        event.target.value = '';
        lucide.createIcons();
        return;
      }

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      formData.append('title', title.trim());

      try {
        const res = await fetch('/api/scan/batch-upload', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error(await res.text());
        const doc = await res.json();
        await fetchDocuments();
        openDocumentDetail(doc);
      } catch (err) {
        alert('Batch scan failed: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = origHtml;
        event.target.value = '';
        lucide.createIcons();
      }
    }

    // ==============================================
    // ANTI-THEFT WATERMARK STUDIO
    // ==============================================
    function openWatermarkModal() {
      if (!state.activeDoc) {
        alert('Please open a document first.');
        return;
      }
      document.getElementById('modal-watermark').classList.remove('hidden');
      lucide.createIcons();
    }

    function closeWatermarkModal() {
      document.getElementById('modal-watermark').classList.add('hidden');
    }

    async function submitWatermark() {
      if (!state.activeDoc) return;
      const textInput = document.getElementById('wm-text-input').value.trim() || 'CONFIDENTIAL';
      const opacityVal = parseFloat(document.getElementById('wm-opacity-slider').value) / 100.0;
      const colorRadio = document.querySelector('input[name="wm-color"]:checked')?.value || 'gray';

      const btn = document.getElementById('btn-submit-watermark');
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Applying...</span>';
      lucide.createIcons();

      try {
        const res = await fetch(`/api/documents/${state.activeDoc.id}/watermark`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            watermark_text: textInput,
            opacity: opacityVal,
            color: colorRadio
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const updated = await res.json();
        closeWatermarkModal();
        await fetchDocuments();
        openDocumentDetail(updated);
      } catch (err) {
        alert('Failed to apply watermark: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="shield-check" class="w-4 h-4"></i><span>Apply to All Pages</span>';
        lucide.createIcons();
      }
    }

    // ==============================================
    // ELECTRONIC SIGNATURE PAD STUDIO
    // ==============================================
    let sigCanvas = null;
    let sigCtx = null;
    let isDrawing = false;
    let hasDrawn = false;
    let sigStrokeColor = '#1e3a8a';

    function initSignaturePad() {
      sigCanvas = document.getElementById('sig-pad-canvas');
      if (!sigCanvas) return;
      sigCtx = sigCanvas.getContext('2d');

      function getPos(e) {
        const rect = sigCanvas.getBoundingClientRect();
        const scaleX = sigCanvas.width / rect.width;
        const scaleY = sigCanvas.height / rect.height;
        let clientX = e.clientX;
        let clientY = e.clientY;
        if (e.touches && e.touches[0]) {
          clientX = e.touches[0].clientX;
          clientY = e.touches[0].clientY;
        }
        return {
          x: (clientX - rect.left) * scaleX,
          y: (clientY - rect.top) * scaleY
        };
      }

      function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        const p = getPos(e);
        sigCtx.beginPath();
        sigCtx.moveTo(p.x, p.y);
      }

      function moveDraw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const p = getPos(e);
        sigCtx.strokeStyle = sigStrokeColor;
        sigCtx.lineWidth = 3;
        sigCtx.lineCap = 'round';
        sigCtx.lineJoin = 'round';
        sigCtx.lineTo(p.x, p.y);
        sigCtx.stroke();
      }

      function endDraw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        isDrawing = false;
        sigCtx.closePath();
      }

      sigCanvas.addEventListener('pointerdown', startDraw);
      sigCanvas.addEventListener('pointermove', moveDraw);
      sigCanvas.addEventListener('pointerup', endDraw);
      sigCanvas.addEventListener('pointercancel', endDraw);
      sigCanvas.addEventListener('touchstart', startDraw, { passive: false });
      sigCanvas.addEventListener('touchmove', moveDraw, { passive: false });
      sigCanvas.addEventListener('touchend', endDraw, { passive: false });
    }

    function clearSignaturePad() {
      if (sigCanvas && sigCtx) {
        sigCtx.clearRect(0, 0, sigCanvas.width, sigCanvas.height);
        hasDrawn = false;
      }
    }

    function setSigColor(hex, btnEl) {
      sigStrokeColor = hex;
      document.querySelectorAll('.sig-color-btn').forEach(b => {
        b.classList.remove('border-white');
        b.classList.add('border-transparent');
      });
      if (btnEl) {
        btnEl.classList.remove('border-transparent');
        btnEl.classList.add('border-white');
      }
    }

    function openSignatureModal() {
      if (!state.activeDoc || !state.activeDoc.pages || state.activeDoc.pages.length === 0) {
        alert('Please open a document with at least one page first.');
        return;
      }
      const select = document.getElementById('sig-page-select');
      select.innerHTML = '';
      state.activeDoc.pages.forEach((p, idx) => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = `Page ${idx + 1} (${p.id.substring(0, 8)})`;
        select.appendChild(opt);
      });

      clearSignaturePad();
      document.getElementById('modal-signature').classList.remove('hidden');
      lucide.createIcons();
    }

    function closeSignatureModal() {
      document.getElementById('modal-signature').classList.add('hidden');
    }

    async function submitSignature() {
      if (!state.activeDoc) return;
      if (!hasDrawn) {
        alert('Please sign inside the signature box before stamping.');
        return;
      }
      const pageId = document.getElementById('sig-page-select').value;
      const pos = document.getElementById('sig-pos-select').value;
      const dataUrl = sigCanvas.toDataURL('image/png');

      const btn = document.getElementById('btn-submit-sig');
      btn.disabled = true;
      btn.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Stamping...</span>';
      lucide.createIcons();

      try {
        const res = await fetch(`/api/documents/${state.activeDoc.id}/signature`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            page_id: pageId,
            signature_base64: dataUrl,
            position: pos
          })
        });
        if (!res.ok) throw new Error(await res.text());
        const updated = await res.json();
        closeSignatureModal();
        await fetchDocuments();
        openDocumentDetail(updated);
      } catch (err) {
        alert('Failed to stamp signature: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="check" class="w-4 h-4"></i><span>Stamp Signature</span>';
        lucide.createIcons();
      }
    }

    async function fetchDocuments() {
      try {
        const res = await fetch('/api/documents');
        const data = await res.json();
        state.documents = data.documents || [];
        renderDocuments();
      } catch (err) {
        console.error('Failed to load docs', err);
      }
    }

    function renderDocuments(query = '') {
      const container = document.getElementById('doc-list-container');
      const emptyState = document.getElementById('empty-state');
      const counter = document.getElementById('doc-counter');

      let docs = state.documents;
      if (state.selectedCategory && state.selectedCategory.toLowerCase() !== 'all') {
        docs = docs.filter(d => (d.category || '').toLowerCase() === state.selectedCategory.toLowerCase());
      }
      if (query) {
        docs = docs.filter(d => d.title.toLowerCase().includes(query.toLowerCase()));
      }

      counter.textContent = `${docs.length} ${docs.length === 1 ? 'document' : 'documents'}`;

      const cards = container.querySelectorAll('.doc-card');
      cards.forEach(c => c.remove());

      if (docs.length === 0) {
        emptyState.classList.remove('hidden');
        return;
      }
      emptyState.classList.add('hidden');

      docs.forEach(doc => {
        const card = document.createElement('div');
        card.className = 'doc-card bg-brand-surface border border-slate-800 rounded-xl p-3 flex items-center gap-3.5 shadow-sm hover:border-slate-700 cursor-pointer transition';
        const thumb = doc.pages?.[0]?.thumbnail_url || '';
        const dateStr = new Date(doc.updated_at).toLocaleDateString(undefined, {
          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const pCount = doc.pages?.length || 1;

        card.innerHTML = `
          <div class="w-16 h-20 rounded-lg bg-slate-900 overflow-hidden flex-shrink-0 border border-slate-800 flex items-center justify-center">
            ${thumb ? `<img src="${thumb}" class="w-full h-full object-cover">` : `<i data-lucide="file-text" class="w-8 h-8 text-slate-500"></i>`}
          </div>
          <div class="flex-1 min-w-0">
            <h3 class="text-sm font-semibold text-white truncate">${escapeHtml(doc.title)}</h3>
            <p class="text-xs text-slate-400 mt-1">${dateStr}</p>
            <div class="mt-2 flex items-center gap-2">
              <span class="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-medium">
                ${pCount} ${pCount === 1 ? 'page' : 'pages'}
              </span>
              ${doc.category ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-brand/15 text-brand-light font-medium border border-brand/20">${escapeHtml(doc.category)}</span>` : ''}
            </div>
          </div>
          <button class="btn-menu p-2 text-slate-400 hover:text-white rounded-full hover:bg-slate-800" data-id="${doc.id}">
            <i data-lucide="more-vertical" class="w-5 h-5"></i>
          </button>
        `;

        card.addEventListener('click', (e) => {
          if (e.target.closest('.btn-menu')) {
            e.stopPropagation();
            showDocMenu(doc);
            return;
          }
          openDocumentDetail(doc);
        });

        container.appendChild(card);
      });
      lucide.createIcons();
    }

    function escapeHtml(s) {
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function openDocumentDetail(doc) {
      state.activeDoc = doc;
      document.getElementById('doc-title-text').textContent = doc.title;
      const count = doc.pages?.length || 1;
      document.getElementById('doc-meta-text').textContent = `${count} ${count === 1 ? 'page' : 'pages'} • ${new Date(doc.updated_at).toLocaleDateString()}`;

      const grid = document.getElementById('doc-pages-grid');
      grid.innerHTML = '';

      (doc.pages || []).forEach((p, idx) => {
        const item = document.createElement('div');
        item.className = 'bg-brand-surface rounded-xl border border-slate-800 overflow-hidden shadow-md flex flex-col';
        item.innerHTML = `
          <div class="aspect-[3/4] bg-slate-900 overflow-hidden flex items-center justify-center p-1.5 cursor-pointer">
            <img src="${p.thumbnail_url || p.url}" class="w-full h-full object-contain rounded">
          </div>
          <div class="px-3 py-2 flex items-center justify-between border-t border-slate-800/80 bg-slate-900/40 text-xs">
            <span class="font-medium text-slate-300">Page ${idx + 1}</span>
            <div class="flex items-center gap-1">
              <button class="btn-ocr-page p-1 text-slate-400 hover:text-brand-light" title="Extract Text (OCR)" data-pid="${p.id}">
                <i data-lucide="file-text" class="w-3.5 h-3.5"></i>
              </button>
              <button class="btn-del-page p-1 text-slate-400 hover:text-red-400" title="Delete Page" data-pid="${p.id}">
                <i data-lucide="trash" class="w-3.5 h-3.5"></i>
              </button>
            </div>
          </div>
        `;

        item.querySelector('.btn-ocr-page').addEventListener('click', (e) => {
          e.stopPropagation();
          runPageOcr(p.id);
        });

        item.querySelector('.btn-del-page').addEventListener('click', async (e) => {
          e.stopPropagation();
          if (confirm(`Delete Page ${idx + 1}?`)) {
            await fetch(`/api/documents/${doc.id}/pages/${p.id}`, { method: 'DELETE' });
            const fresh = await (await fetch(`/api/documents/${doc.id}`)).json();
            openDocumentDetail(fresh);
            fetchDocuments();
          }
        });

        grid.appendChild(item);
      });

      switchView('doc');
      lucide.createIcons();
    }

    async function runPageOcr(pageId) {
      const modal = document.getElementById('modal-ocr');
      const content = document.getElementById('ocr-text-content');
      content.textContent = 'Extracting text with RapidOCR...';
      modal.classList.remove('hidden');
      lucide.createIcons();

      try {
        const res = await fetch(`/api/scan/ocr/${pageId}`, { method: 'POST' });
        const data = await res.json();
        if (data.text) {
          content.textContent = data.text;
        } else {
          content.textContent = 'No text detected in this page.';
        }
      } catch (err) {
        content.textContent = 'Failed to extract text: ' + err.message;
      }
    }

    document.getElementById('btn-close-ocr').addEventListener('click', () => {
      document.getElementById('modal-ocr').classList.add('hidden');
    });

    document.getElementById('btn-copy-ocr').addEventListener('click', () => {
      const text = document.getElementById('ocr-text-content').textContent;
      navigator.clipboard.writeText(text);
      alert('OCR text copied to clipboard!');
    });

    function showDocMenu(doc) {
      const choice = prompt(`Document: "${doc.title}"\\n1. Export / Share PDF\\n2. Rename Document\\n3. Delete Document\\nType 1, 2, or 3:`, '1');
      if (choice === '1') {
        window.open(`/api/documents/${doc.id}/export-pdf`, '_blank');
      } else if (choice === '2') {
        const newTitle = prompt('New title:', doc.title);
        if (newTitle && newTitle.trim()) {
          fetch(`/api/documents/${doc.id}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ title: newTitle.trim() })
          }).then(() => fetchDocuments());
        }
      } else if (choice === '3') {
        if (confirm(`Delete "${doc.title}"?`)) {
          fetch(`/api/documents/${doc.id}`, { method: 'DELETE' }).then(() => fetchDocuments());
        }
      }
    }

    // Upload & Scan Handling
    async function handleFileSelect(file) {
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/scan/upload', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();

        state.uploadId = data.upload_id;
        state.sourceWidth = data.width;
        state.sourceHeight = data.height;
        state.corners = data.corners;

        state.sourceImg.onload = () => {
          setupCropWorkspace();
        };
        state.sourceImg.src = data.url;
      } catch (err) {
        alert('Upload failed: ' + err.message);
      }
    }

    // Setup Crop Canvas
    const cropCanvas = document.getElementById('crop-canvas');
    const cropCtx = cropCanvas.getContext('2d');
    const cropWorkspace = document.getElementById('crop-workspace');
    const loupe = document.getElementById('magnifier-loupe');
    const loupeCanvas = document.getElementById('loupe-canvas');
    const loupeCtx = loupeCanvas.getContext('2d');

    function setupCropWorkspace() {
      updateCropDisplaySize();
      switchView('crop');
      drawCropCanvas();
    }

    function updateCropDisplaySize() {
      const containerW = cropWorkspace.clientWidth || window.innerWidth;
      const containerH = cropWorkspace.clientHeight || (window.innerHeight - 120);

      const imgW = state.sourceWidth;
      const imgH = state.sourceHeight;

      state.displayScale = Math.min((containerW - 32) / imgW, (containerH - 32) / imgH, 1);
      cropCanvas.width = Math.round(imgW * state.displayScale);
      cropCanvas.height = Math.round(imgH * state.displayScale);
    }

    function drawCropCanvas() {
      const cw = cropCanvas.width;
      const ch = cropCanvas.height;
      const s = state.displayScale;

      cropCtx.clearRect(0, 0, cw, ch);
      cropCtx.drawImage(state.sourceImg, 0, 0, cw, ch);

      const pts = {
        tl: { x: state.corners.tl.x * s, y: state.corners.tl.y * s },
        tr: { x: state.corners.tr.x * s, y: state.corners.tr.y * s },
        br: { x: state.corners.br.x * s, y: state.corners.br.y * s },
        bl: { x: state.corners.bl.x * s, y: state.corners.bl.y * s }
      };

      const mid = {
        top: { x: (pts.tl.x + pts.tr.x) / 2, y: (pts.tl.y + pts.tr.y) / 2 },
        right: { x: (pts.tr.x + pts.br.x) / 2, y: (pts.tr.y + pts.br.y) / 2 },
        bottom: { x: (pts.bl.x + pts.br.x) / 2, y: (pts.bl.y + pts.br.y) / 2 },
        left: { x: (pts.tl.x + pts.bl.x) / 2, y: (pts.tl.y + pts.bl.y) / 2 }
      };

      // Shaded overlay outside polygon
      cropCtx.save();
      cropCtx.fillStyle = 'rgba(0,0,0,0.55)';
      cropCtx.fillRect(0, 0, cw, ch);

      cropCtx.globalCompositeOperation = 'destination-out';
      cropCtx.beginPath();
      cropCtx.moveTo(pts.tl.x, pts.tl.y);
      cropCtx.lineTo(pts.tr.x, pts.tr.y);
      cropCtx.lineTo(pts.br.x, pts.br.y);
      cropCtx.lineTo(pts.bl.x, pts.bl.y);
      cropCtx.closePath();
      cropCtx.fill();
      cropCtx.restore();

      // Boundary strokes
      cropCtx.save();
      cropCtx.strokeStyle = '#189a80';
      cropCtx.lineWidth = 2.5;
      cropCtx.beginPath();
      cropCtx.moveTo(pts.tl.x, pts.tl.y);
      cropCtx.lineTo(pts.tr.x, pts.tr.y);
      cropCtx.lineTo(pts.br.x, pts.br.y);
      cropCtx.lineTo(pts.bl.x, pts.bl.y);
      cropCtx.closePath();
      cropCtx.stroke();

      // Corner handles
      const drawCorner = (p) => {
        cropCtx.beginPath();
        cropCtx.arc(p.x, p.y, 14, 0, Math.PI * 2);
        cropCtx.fillStyle = 'rgba(24, 154, 128, 0.4)';
        cropCtx.fill();

        cropCtx.beginPath();
        cropCtx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        cropCtx.fillStyle = '#ffffff';
        cropCtx.fill();
        cropCtx.strokeStyle = '#189a80';
        cropCtx.lineWidth = 2.5;
        cropCtx.stroke();
      };

      const drawMid = (p) => {
        cropCtx.beginPath();
        cropCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        cropCtx.fillStyle = '#ffffff';
        cropCtx.fill();
        cropCtx.strokeStyle = '#189a80';
        cropCtx.lineWidth = 2;
        cropCtx.stroke();
      };

      drawCorner(pts.tl);
      drawCorner(pts.tr);
      drawCorner(pts.br);
      drawCorner(pts.bl);

      drawMid(mid.top);
      drawMid(mid.right);
      drawMid(mid.bottom);
      drawMid(mid.left);

      cropCtx.restore();
    }

    function getPointerPos(e) {
      const rect = cropCanvas.getBoundingClientRect();
      const cx = e.clientX ?? e.touches?.[0]?.clientX;
      const cy = e.clientY ?? e.touches?.[0]?.clientY;
      return {
        x: cx - rect.left,
        y: cy - rect.top,
        screenX: cx,
        screenY: cy
      };
    }

    function findHandle(pos) {
      const s = state.displayScale;
      const hit = 28;
      const corners = {
        tl: { x: state.corners.tl.x * s, y: state.corners.tl.y * s },
        tr: { x: state.corners.tr.x * s, y: state.corners.tr.y * s },
        br: { x: state.corners.br.x * s, y: state.corners.br.y * s },
        bl: { x: state.corners.bl.x * s, y: state.corners.bl.y * s }
      };

      for (const [k, p] of Object.entries(corners)) {
        if (Math.hypot(p.x - pos.x, p.y - pos.y) <= hit) return k;
      }

      const mids = {
        top: { x: (corners.tl.x + corners.tr.x) / 2, y: (corners.tl.y + corners.tr.y) / 2 },
        right: { x: (corners.tr.x + corners.br.x) / 2, y: (corners.tr.y + corners.br.y) / 2 },
        bottom: { x: (corners.bl.x + corners.br.x) / 2, y: (corners.bl.y + corners.br.y) / 2 },
        left: { x: (corners.tl.x + corners.bl.x) / 2, y: (corners.tl.y + corners.bl.y) / 2 }
      };

      for (const [k, p] of Object.entries(mids)) {
        if (Math.hypot(p.x - pos.x, p.y - pos.y) <= hit) return k;
      }
      return null;
    }

    cropCanvas.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      const pos = getPointerPos(e);
      const h = findHandle(pos);
      if (!h) return;
      state.activeHandle = h;
      updateLoupe(pos);
      loupe.classList.remove('hidden');
    });

    window.addEventListener('pointermove', (e) => {
      if (!state.activeHandle) return;
      e.preventDefault();
      const pos = getPointerPos(e);
      const s = state.displayScale;
      const rx = Math.max(0, Math.min(state.sourceWidth, pos.x / s));
      const ry = Math.max(0, Math.min(state.sourceHeight, pos.y / s));

      if (['tl', 'tr', 'br', 'bl'].includes(state.activeHandle)) {
        state.corners[state.activeHandle] = { x: Math.round(rx), y: Math.round(ry) };
      } else if (state.activeHandle === 'top') {
        const dy = ry - (state.corners.tl.y + state.corners.tr.y) / 2;
        state.corners.tl.y += dy; state.corners.tr.y += dy;
      } else if (state.activeHandle === 'bottom') {
        const dy = ry - (state.corners.bl.y + state.corners.br.y) / 2;
        state.corners.bl.y += dy; state.corners.br.y += dy;
      } else if (state.activeHandle === 'left') {
        const dx = rx - (state.corners.tl.x + state.corners.bl.x) / 2;
        state.corners.tl.x += dx; state.corners.bl.x += dx;
      } else if (state.activeHandle === 'right') {
        const dx = rx - (state.corners.tr.x + state.corners.br.x) / 2;
        state.corners.tr.x += dx; state.corners.br.x += dx;
      }

      drawCropCanvas();
      updateLoupe(pos);
    });

    window.addEventListener('pointerup', () => {
      state.activeHandle = null;
      loupe.classList.add('hidden');
    });

    function updateLoupe(pos) {
      const s = state.displayScale;
      const rx = pos.x / s;
      const ry = pos.y / s;

      loupe.style.left = `${pos.screenX}px`;
      loupe.style.top = `${pos.screenY - 30}px`;

      loupeCanvas.width = 112;
      loupeCanvas.height = 112;
      loupeCtx.clearRect(0, 0, 112, 112);
      loupeCtx.drawImage(
        state.sourceImg,
        rx - 25, ry - 25, 50, 50,
        0, 0, 112, 112
      );
    }

    // Process & Filter Warp
    async function requestWarpAndFilter() {
      const loading = document.getElementById('filter-loading');
      const previewImg = document.getElementById('filter-preview-img');
      loading.classList.remove('hidden');

      try {
        const res = await fetch('/api/scan/warp-filter', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            upload_id: state.uploadId,
            corners: state.corners,
            filter_type: state.filterType,
            brightness: state.brightness,
            contrast: state.contrast
          })
        });

        if (!res.ok) throw new Error('Warp and filter processing failed');
        const data = await res.json();
        state.processedPageId = data.page_id;
        previewImg.src = data.page_url + '?t=' + Date.now();
      } catch (err) {
        alert('Processing error: ' + err.message);
      } finally {
        loading.classList.add('hidden');
      }
    }

    // Save Page to Document
    async function saveCurrentScan() {
      if (!state.processedPageId) return;

      if (state.isAddingPage && state.activeDoc) {
        const form = new FormData();
        form.append('page_id', state.processedPageId);
        const res = await fetch(`/api/documents/${state.activeDoc.id}/pages`, {
          method: 'POST',
          body: form
        });
        const doc = await res.json();
        state.isAddingPage = false;
        await fetchDocuments();
        openDocumentDetail(doc);
      } else {
        const res = await fetch('/api/documents', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            page_ids: [state.processedPageId]
          })
        });
        const doc = await res.json();
        await fetchDocuments();
        openDocumentDetail(doc);
      }
    }

    // Event Wire-up
    document.getElementById('btn-camera').onclick = () => {
      state.isAddingPage = false;
      document.getElementById('camera-file-input').click();
    };
    document.getElementById('btn-gallery').onclick = () => {
      state.isAddingPage = false;
      document.getElementById('gallery-file-input').click();
    };

    document.getElementById('camera-file-input').onchange = (e) => {
      if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
      e.target.value = '';
    };
    document.getElementById('gallery-file-input').onchange = (e) => {
      if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
      e.target.value = '';
    };

    document.getElementById('btn-crop-cancel').onclick = () => {
      switchView(state.isAddingPage && state.activeDoc ? 'doc' : 'home');
    };

    document.getElementById('btn-crop-all').onclick = () => {
      state.corners = {
        tl: {x: 0, y: 0},
        tr: {x: state.sourceWidth, y: 0},
        br: {x: state.sourceWidth, y: state.sourceHeight},
        bl: {x: 0, y: state.sourceHeight}
      };
      drawCropCanvas();
    };

    document.getElementById('btn-crop-next').onclick = async () => {
      switchView('filter');
      state.filterType = 'magic';
      state.brightness = 0;
      state.contrast = 0;
      document.getElementById('slider-brightness').value = 0;
      document.getElementById('slider-contrast').value = 0;
      document.querySelectorAll('.filter-pill').forEach(p => {
        p.classList.toggle('active', p.dataset.filter === 'magic');
      });
      await requestWarpAndFilter();
    };

    document.getElementById('btn-filter-back').onclick = () => {
      switchView('crop');
    };

    document.querySelectorAll('.filter-pill').forEach(pill => {
      pill.onclick = async () => {
        document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        state.filterType = pill.dataset.filter;
        await requestWarpAndFilter();
      };
    });

    let debTimer = null;
    function debounceWarp() {
      clearTimeout(debTimer);
      debTimer = setTimeout(requestWarpAndFilter, 150);
    }

    document.getElementById('slider-brightness').oninput = (e) => {
      state.brightness = parseInt(e.target.value);
      debounceWarp();
    };
    document.getElementById('slider-contrast').oninput = (e) => {
      state.contrast = parseInt(e.target.value);
      debounceWarp();
    };

    document.getElementById('btn-filter-save').onclick = saveCurrentScan;

    document.getElementById('btn-doc-back').onclick = () => switchView('home');

    document.getElementById('btn-add-page').onclick = () => {
      state.isAddingPage = true;
      document.getElementById('camera-file-input').click();
    };

    document.getElementById('btn-export-pdf').onclick = () => {
      if (state.activeDoc) {
        window.open(`/api/documents/${state.activeDoc.id}/export-pdf?download=true`, '_blank');
      }
    };

    document.getElementById('btn-export-txt').onclick = () => {
      if (state.activeDoc) {
        window.open(`/api/documents/${state.activeDoc.id}/export-txt`, '_blank');
      }
    };

    document.getElementById('btn-export-docx').onclick = () => {
      if (state.activeDoc) {
        window.open(`/api/documents/${state.activeDoc.id}/export-docx`, '_blank');
      }
    };

    document.getElementById('btn-export-zip').onclick = () => {
      if (state.activeDoc) {
        window.open(`/api/documents/${state.activeDoc.id}/export-zip`, '_blank');
      }
    };

    document.getElementById('btn-rename-doc').onclick = () => {
      if (!state.activeDoc) return;
      const t = prompt('New title:', state.activeDoc.title);
      if (t && t.trim()) {
        fetch(`/api/documents/${state.activeDoc.id}`, {
          method: 'PUT',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ title: t.trim() })
        }).then(async () => {
          state.activeDoc.title = t.trim();
          document.getElementById('doc-title-text').textContent = t.trim();
          fetchDocuments();
        });
      }
    };

    document.getElementById('btn-delete-doc').onclick = () => {
      if (!state.activeDoc) return;
      if (confirm(`Delete document "${state.activeDoc.title}"?`)) {
        fetch(`/api/documents/${state.activeDoc.id}`, { method: 'DELETE' }).then(() => {
          fetchDocuments();
          switchView('home');
        });
      }
    };

    document.getElementById('btn-search-toggle').onclick = () => {
      const sc = document.getElementById('search-container');
      sc.classList.toggle('hidden');
      if (!sc.classList.contains('hidden')) {
        document.getElementById('input-search').focus();
      } else {
        document.getElementById('input-search').value = '';
        renderDocuments();
      }
    };

    document.getElementById('input-search').oninput = (e) => {
      renderDocuments(e.target.value);
    };

    document.getElementById('btn-refresh').onclick = fetchDocuments;

    let deferredPrompt = null;
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      const pwaBtn = document.getElementById('pwa-install-btn');
      if (pwaBtn) pwaBtn.classList.remove('hidden');
    });

    function installPWA() {
      if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
          if (choiceResult.outcome === 'accepted') {
            const pwaBtn = document.getElementById('pwa-install-btn');
            if (pwaBtn) pwaBtn.classList.add('hidden');
          }
          deferredPrompt = null;
        });
      }
    }

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {});
    }

    window.onload = () => {
      lucide.createIcons();
      initSignaturePad();
      fetchDocuments();
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the complete CamScanner Pro responsive Mobile & Web SPA."""
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/manifest.json")
async def manifest():
    return JSONResponse(
        content={
            "name": "CamScanner Pro",
            "short_name": "CamScanner",
            "description": "Smart Document Scanner, Perspective Correction & OCR Engine",
            "start_url": "/",
            "display": "standalone",
            "orientation": "portrait",
            "background_color": "#0f1115",
            "theme_color": "#189a80",
            "icons": [
                {
                    "src": "/static/logo_192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/static/logo_96.png",
                    "sizes": "96x96",
                    "type": "image/png"
                },
                {
                    "src": "/static/favicon.png",
                    "sizes": "48x48",
                    "type": "image/png"
                },
                {
                    "src": "/icon.svg",
                    "sizes": "512x512",
                    "type": "image/svg+xml"
                }
            ]
        }
    )


@app.get("/icon.svg")
async def icon_svg():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#189a80"/>
      <stop offset="100%" stop-color="#0b5b4c"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.3"/>
    </filter>
  </defs>
  <rect width="512" height="512" rx="128" fill="url(#grad)"/>
  <g filter="url(#shadow)" fill="none" stroke="#ffffff" stroke-width="28" stroke-linecap="round" stroke-linejoin="round">
    <path d="M120 180 L160 120 L352 120 L392 180 Z" fill="#ffffff" fill-opacity="0.15"/>
    <rect x="96" y="180" width="320" height="230" rx="36"/>
    <circle cx="256" cy="295" r="68" stroke-width="26" fill="#ffffff" fill-opacity="0.2"/>
    <circle cx="360" cy="225" r="14" fill="#ffffff" stroke="none"/>
  </g>
  <path d="M180 295 L230 345 L335 240" fill="none" stroke="#22c55e" stroke-width="24" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/sw.js")
async def service_worker():
    sw_code = """
const CACHE_NAME = 'camscanner-v3';
const ASSETS = [
  '/',
  '/manifest.json',
  '/static/favicon.png',
  '/static/logo_192.png',
  '/static/official/ic_scanner_empty_doc-BWsAcaxT.png',
  '/icon.svg',
  'https://cdn.tailwindcss.com',
  'https://unpkg.com/lucide@latest'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((res) => {
      return res || fetch(e.request).then((netRes) => {
        return netRes;
      }).catch(() => caches.match('/'));
    })
  );
});
"""
    return Response(content=sw_code, media_type="application/javascript")


@app.get("/api/diagnostics")
@app.get("/health")
async def health_diagnostics():
    """Health check endpoint for Render, Uptime monitors, and Keep-Awake workflows."""
    return {
        "status": "healthy",
        "app": "CamScanner Pro",
        "version": "2.0.0",
        "ocr_available": HAS_OCR,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
