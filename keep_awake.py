import time
import urllib.request
import urllib.error
from datetime import datetime

# You can add your Render URL here once deployed, or keep-awake the live tunnel
TARGET_URLS = [
    "https://address-metal-kits-springer.trycloudflare.com/health",
]
INTERVAL_SECONDS = 300  # 5 minutes
USER_AGENT = "CamScannerKeepAwake/1.0"


def ping_url(url: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="GET"
    )
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start_time
            print(f"[{timestamp}] PING SUCCESS -> Status: {resp.status} ({elapsed:.2f}s) | {url}", flush=True)
            return True, resp.status
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] PING HTTP {e.code} ({elapsed:.2f}s) | Reason: {e.reason}", flush=True)
        return False, e.code
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] PING ERROR ({elapsed:.2f}s) | Error: {e}", flush=True)
        return False, str(e)


def main():
    print("============================================================")
    print(" CamScanner Pro Keep-Awake Anti-Hibernation Daemon")
    print(f" Monitoring: {TARGET_URLS}")
    print(f" Interval: {INTERVAL_SECONDS}s")
    print("============================================================", flush=True)

    for u in TARGET_URLS:
        ping_url(u)

    while True:
        time.sleep(INTERVAL_SECONDS)
        for u in TARGET_URLS:
            ping_url(u)


if __name__ == "__main__":
    main()
