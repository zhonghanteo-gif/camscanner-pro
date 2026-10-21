import os
import re

base_path = r"C:\Users\teot4\.gemini\antigravity\scratch\camscanner-extracted\app"
renderer_dir = os.path.join(base_path, ".vite", "renderer", "main_window", "static")
recent_path = os.path.join(renderer_dir, "RecentPage-L4XZB00B.js")

with open(recent_path, "r", encoding="utf-8") as f:
    text = f.read()

# Look for vue template strings or jsx or methods
strings = re.findall(r'["\']([^"\']{4,50})["\']', text)
keywords = [s for s in strings if any(k in s.lower() for k in ["folder", "doc", "select", "delete", "share", "tag", "search", "view", "grid", "list", "sort", "create"])]
print("RecentPage keywords found:", len(keywords))
print(list(set(keywords))[:30])
