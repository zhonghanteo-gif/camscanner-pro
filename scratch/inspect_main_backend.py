import os
import re

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
main_backend = os.path.join(base_path, ".vite", "build", "main-B7BvPehC.js")

with open(main_backend, "r", encoding="utf-8") as f:
    code = f.read()

print("File size:", len(code))

# IPC channels and endpoints
ipc_calls = re.findall(r'["\']([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+)["\']', code)
unique_ipc = sorted(set(ipc_calls))
print(f"IPC namespaced channels: {len(unique_ipc)}")
print("Sample IPC namespaced channels (top 60):")
for ch in unique_ipc[:60]:
    print("  *", ch)

# Search for service classes
services = re.findall(r'class\s+([A-Za-z0-9_]+Service)', code)
print(f"Service classes: {set(services)}")

# Search for managers
managers = re.findall(r'class\s+([A-Za-z0-9_]+Manager)', code)
print(f"Manager classes: {set(managers)}")
