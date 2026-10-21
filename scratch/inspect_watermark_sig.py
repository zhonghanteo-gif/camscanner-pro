import os
base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
renderer_dir = os.path.join(base_path, ".vite", "renderer", "main_window", "static")

for fname in ["cs_ic_add_watermark-C1jzuUXM.js", "cs_ic_common_signature-DWDPIfcs.js", "cs_ic_ocr-fdR9qnDw.js"]:
    p = os.path.join(renderer_dir, fname)
    if os.path.exists(p):
        print(f"=== {fname} ===")
        print(open(p, "r", encoding="utf-8").read()[:500])
