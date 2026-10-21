import os
import re

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
renderer_dir = os.path.join(base_path, ".vite", "renderer", "main_window", "static")

files_to_check = [
    "ManagerPage-D_rwQQct.js",
    "RecentPage-L4XZB00B.js",
    "RecyclerPage-_zTGQLDA.js",
    "add_page-Cw-k_VYb.js",
    "cs_ic_add_watermark-C1jzuUXM.js",
    "cs_ic_common_signature-DWDPIfcs.js",
]

for fname in files_to_check:
    fpath = os.path.join(renderer_dir, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        content = open(fpath, "r", encoding="utf-8").read(10000)
        print(f"File: {fname} (Size: {size} bytes)")
        buttons = re.findall(r'["\']([a-zA-Z0-9_\-\s]+(?:btn|button|dialog|modal|tab|menu|card|filter|export|import|action)[a-zA-Z0-9_\-\s]*)["\']', content, re.IGNORECASE)
        print("  Buttons/UI classes:", list(set(buttons))[:10])
