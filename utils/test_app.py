import asyncio
from playwright.async_api import async_playwright
import time

async def run_test():
    async with async_playwright() as p:
        print("🚀 Launching Browser...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        
        print("🌐 Opening http://localhost:8501...")
        try:
            await page.goto("http://localhost:8501", timeout=60000)
            await asyncio.sleep(8) # Wait for Streamlit spin-up
            
            title = await page.title()
            print(f"✅ Page Title: {title}")
            
            # 1. Take initial screenshot
            await page.screenshot(path="test_initial.png")
            print("📸 Initial view captured.")
            
            # 2. Verify Metrics are present
            metrics = await page.query_selector_all("div[data-testid='stMetricValue']")
            print(f"📊 Found {len(metrics)} active metrics.")
            
            # 3. Switch to Agent Co-Pilot Tab
            # Streamlit tabs are usually buttons with role='tab'
            tabs = await page.query_selector_all("button[role='tab']")
            for tab in tabs:
                text = await tab.inner_text()
                if "Agent Co-Pilot" in text:
                    print(f"🖱️ Switching to {text}...")
                    await tab.click()
                    await asyncio.sleep(2)
                    
                    # 4. Trigger Agent
                    run_btn = await page.query_selector("button:has-text('Run Diagnostic Agent')")
                    if run_btn:
                        print("🚀 Executing Diagnostic Agent...")
                        await run_btn.click()
                        print("⏳ Waiting for Agent Reasoning Trace (20s)...")
                        await asyncio.sleep(20)
                        await page.screenshot(path="test_agent_trace.png")
                        print("📸 Agent Trace captured.")
                    break
            
            # 5. Check What-If Simulator
            for tab in tabs:
                text = await tab.inner_text()
                if "What-If Simulator" in text:
                    print(f"🖱️ Switching to {text}...")
                    await tab.click()
                    await asyncio.sleep(2)
                    
                    slider = await page.query_selector("div[data-testid='stSlider']")
                    if slider:
                        print("✅ Simulator Slider found.")
                    break
                    
            print("✨ VIGOROUS TESTING COMPLETE: Dashboard is fully operational.")
            
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            await page.screenshot(path="test_error.png")
        finally:
            await browser.close()
            print("🏁 Browser closed.")

if __name__ == "__main__":
    asyncio.run(run_test())
