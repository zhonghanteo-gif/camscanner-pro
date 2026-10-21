import os
import re

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
backend_main = os.path.join(base_path, ".vite", "build", "main-CW7g0BA8.js")

with open(backend_main, "r", encoding="utf-8") as f:
    code = f.read()

print("File size:", len(code))

# Search for electron ipc or event emitters
ipc_terms = re.findall(r'(\w+)\.handle\(', code)
print("handle callers:", set(ipc_terms))

ipc_strings = re.findall(r'["\']([a-zA-Z0-9_\-:]+(?:scan|ocr|doc|pdf|page|file|auth|user|crop|dewarp|convert)[a-zA-Z0-9_\-:]*)["\']', code)
print(f"Domain strings found: {len(ipc_strings)}")
sample_strings = sorted(set(s for s in ipc_strings if len(s) < 40 and not s.startswith("http")))
print("Sample domain events/keys (first 50):")
for s in sample_strings[:50]:
    print("  *", s)
