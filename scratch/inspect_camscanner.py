import os
import re
import json

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
backend_main = os.path.join(base_path, ".vite", "build", "main-CW7g0BA8.js")

print("=== REVERSE ENGINEERING CAMSCANNER BACKEND & IPC ARCHITECTURE ===")
with open(backend_main, "r", encoding="utf-8") as f:
    backend_code = f.read()

ipc_handles = re.findall(r'ipcMain\.(?:handle|on)\(["\']([^"\']+)["\']', backend_code)
print(f"Total IPC handlers: {len(ipc_handles)}")
unique_ipc = sorted(set(ipc_handles))
for ch in unique_ipc:
    print(f"  IPC: {ch}")

print("\n=== REVERSE ENGINEERING DATABASE & ENTITIES ===")
entities = re.findall(r'class\s+([A-Za-z0-9_]+Entity|[A-Za-z0-9_]+Model|[A-Za-z0-9_]+Table)', backend_code)
print(f"Entities found: {set(entities)}")

tables = re.findall(r'CREATE TABLE IF NOT EXISTS\s+([A-Za-z0-9_]+)', backend_code, re.IGNORECASE)
print(f"SQLite tables found: {set(tables)}")

print("\n=== REVERSE ENGINEERING FRONTEND PAGES & MODULES ===")
renderer_dir = os.path.join(base_path, ".vite", "renderer", "main_window", "static")
page_modules = []
for fname in os.listdir(renderer_dir):
    if fname.endswith(".js"):
        if any(term in fname for term in ["Page", "Modal", "Tool", "Dialog", "Editor", "Scan", "Ocr", "Convert", "PDF", "Watermark", "Sign"]):
            page_modules.append(fname)

print(f"Frontend feature modules: {len(page_modules)}")
for p in sorted(page_modules)[:30]:
    print(f"  Module: {p}")
