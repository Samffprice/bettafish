import asyncio
import msgpack
import json
from datetime import datetime
from patchright.async_api import async_playwright

async def main():
    frames = []

    async with async_playwright() as p:
        # Use persistent context with real Chrome for best stealth
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./chrome_profile",
            channel="chrome",
            headless=False,
            no_viewport=True,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        # Intercept all WS frames
        page.on("websocket", lambda ws: handle_ws(ws, frames))

        await page.goto("https://colonist.io")

        # Play your game manually, then press Enter here when done
        input("Press Enter when game is finished...")

        # Save frames
        filename = f"game_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filename, "w") as f:
            json.dump(frames, f, indent=2, default=str)

        print(f"Captured {len(frames)} frames → {filename}")
        await context.close()


def handle_ws(ws, frames):
    print(f"[WS] Opened: {ws.url}")

    def on_recv(payload):
        try:
            if isinstance(payload, str):
                decoded = json.loads(payload)
            else:
                decoded = msgpack.unpackb(payload, raw=False)
            frames.append({"dir": "recv", "time": datetime.now().isoformat(), "decoded": decoded})
            msg_type = decoded.get("data", {}).get("type", "?") if isinstance(decoded, dict) else "?"
            print(f"  ← type={msg_type}: {json.dumps(decoded, default=str)[:120]}")
        except Exception as e:
            raw = payload.hex() if isinstance(payload, bytes) else payload[:120]
            frames.append({"dir": "recv", "time": datetime.now().isoformat(), "raw": raw})
            print(f"  ← [decode err: {e}] {raw[:80]}")

    def on_sent(payload):
        try:
            if isinstance(payload, str):
                decoded = json.loads(payload)
            else:
                decoded = msgpack.unpackb(payload, raw=False)
            frames.append({"dir": "sent", "time": datetime.now().isoformat(), "decoded": decoded})
            msg_type = decoded.get("data", {}).get("type", "?") if isinstance(decoded, dict) else "?"
            print(f"  → type={msg_type}: {json.dumps(decoded, default=str)[:120]}")
        except Exception as e:
            raw = payload.hex() if isinstance(payload, bytes) else payload[:120]
            frames.append({"dir": "sent", "time": datetime.now().isoformat(), "raw": raw})
            print(f"  → [decode err: {e}] {raw[:80]}")

    ws.on("framereceived", on_recv)
    ws.on("framesent", on_sent)


asyncio.run(main())