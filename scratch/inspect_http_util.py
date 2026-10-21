import os
import re

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
http_util = os.path.join(base_path, ".vite", "build", "localHttpServerUtility-BcKZMagn.js")

with open(http_util, "r", encoding="utf-8") as f:
    code = f.read()

print("File size:", len(code))
routes = re.findall(r'["\'](/api/[^"\']+)["\']', code)
print("API routes found:", set(routes))

methods = re.findall(r'(GET|POST|PUT|DELETE)\s+([a-zA-Z0-9_/-]+)', code)
print("HTTP methods found:", set(methods)[:20])
