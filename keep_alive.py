"""
Keep-alive service to prevent Render free app from sleeping
Run in parallel with API: python keep_alive.py
"""

import requests
import time
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")
INTERVAL = 300  # Ping every 5 minutes (Render sleeps after 15 min inactivity)

def keep_alive():
    """Send periodic health checks to keep API alive"""
    while True:
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            status = "✓" if response.status_code == 200 else "✗"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} API Health Check")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    print(f"Starting keep-alive service... (pinging {API_URL} every {INTERVAL}s)")
    keep_alive()
