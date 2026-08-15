from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()

    # --- Desktop: initial form ---
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto("http://127.0.0.1:8000/")
    page.wait_for_timeout(500)
    page.screenshot(path="screenshot_desktop_form.png", full_page=True)
    print("Saved screenshot_desktop_form.png")

    # --- Desktop: submit a valid date, show result ---
    page.fill("#id_target_date", "2026-01-15")
    page.click("button[type=submit]")
    page.wait_for_timeout(700)
    page.screenshot(path="screenshot_desktop_result.png", full_page=True)
    print("Saved screenshot_desktop_result.png")

    # --- Desktop: invalid input (out of range) ---
    page.fill("#id_target_date", "2020-01-01")
    page.click("button[type=submit]")
    page.wait_for_timeout(500)
    page.screenshot(path="screenshot_desktop_error.png", full_page=True)
    print("Saved screenshot_desktop_error.png")
    page.close()

    # --- Mobile width ---
    mobile_page = browser.new_page(viewport={"width": 390, "height": 844})
    mobile_page.goto("http://127.0.0.1:8000/")
    mobile_page.wait_for_timeout(500)
    mobile_page.fill("#id_target_date", "2026-01-15")
    mobile_page.click("button[type=submit]")
    mobile_page.wait_for_timeout(700)
    mobile_page.screenshot(path="screenshot_mobile_result.png", full_page=True)
    print("Saved screenshot_mobile_result.png")
    mobile_page.close()

    browser.close()

print("Done.")
