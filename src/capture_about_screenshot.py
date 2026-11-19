"""
capture_about_screenshot.py — Automated browser screenshot capture via Chrome CDP.
Connects to headless Chrome, waits for Streamlit React components to render,
and captures the exact rendered About section architecture diagram.
"""

import os
import sys
import time
import json
import base64
import asyncio
import subprocess
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
TARGET_URL = "http://localhost:8501/?page=about"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "about_architecture_diagram_rendered.png")

async def capture_cdp_screenshot():
    print(f"Launching Chrome Headless on {TARGET_URL}...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--window-size=1280,2400",
        TARGET_URL
    ]
    proc = subprocess.Popen(cmd)
    try:
        # Give Chrome time to start and connect to CDP
        await asyncio.sleep(4)

        with urllib.request.urlopen("http://localhost:9222/json") as resp:
            tabs = json.loads(resp.read().decode())

        target_tab = None
        for t in tabs:
            if "MaterialMind" in t.get("title", "") or "8503" in t.get("url", ""):
                target_tab = t
                break

        if not target_tab:
            print("Could not find Streamlit tab in Chrome tabs:", tabs)
            return False

        ws_url = target_tab.get("webSocketDebuggerUrl")
        print(f"Connecting to CDP WebSocket: {ws_url}")

        async with websockets.connect(ws_url) as ws:
            # Wait another 4 seconds for Streamlit's WebSocket/DOM to finish rendering
            print("Waiting for Streamlit page DOM to complete rendering...")
            await asyncio.sleep(4)

            # Send Page.captureScreenshot command
            req = {
                "id": 1,
                "method": "Page.captureScreenshot",
                "params": {
                    "format": "png",
                    "fromSurface": True,
                    "captureBeyondViewport": True
                }
            }
            await ws.send(json.dumps(req))

            response_raw = await ws.recv()
            resp_data = json.loads(response_raw)
            img_b64 = resp_data.get("result", {}).get("data")

            if not img_b64:
                print("Failed to capture screenshot data. Response:", resp_data)
                return False

            img_bytes = base64.b64decode(img_b64)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "wb") as f:
                f.write(img_bytes)

            print(f"SUCCESS: Captured rendered screenshot ({len(img_bytes)} bytes) -> {OUTPUT_PATH}")
            return True

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(capture_cdp_screenshot())
