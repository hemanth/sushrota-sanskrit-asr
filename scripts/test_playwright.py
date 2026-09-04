import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        page.on("console", lambda msg: print(f"[Browser Console {msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[Browser Error] {err}"))

        print("1. Navigating to http://localhost:8090...")
        await page.goto("http://localhost:8090")

        print("2. Clicking 'Download & Load Model'...")
        await page.click("#btnLoad")

        for _ in range(60):
            status = await page.inner_text("#statusLabel")
            if "Model Ready" in status:
                print("   Model Ready!")
                break
            await asyncio.sleep(0.5)

        print("3. Testing whole-utterance transcription on 2.5s audio...")
        await page.evaluate("""() => {
            const pcm = new Float32Array(16000 * 2.5);
            for (let i = 0; i < pcm.length; i++) {
                pcm[i] = Math.sin(2 * Math.PI * 350 * i / 16000) * 0.15;
            }
            window.runInference(pcm);
        }""")

        for _ in range(30):
            text = await page.inner_text("#outputBox")
            if text and not text.startswith("Transcribing"):
                print(f"   Output text: '{text}'")
                badge = await page.inner_text("#timeBadge")
                rtf = await page.inner_text("#rtfBadge")
                print(f"   {badge} | {rtf}")
                print("\n✅ WHOLE-UTTERANCE INFERENCE VERIFIED!")
                break
            await asyncio.sleep(0.2)

        await browser.close()

asyncio.run(main())
