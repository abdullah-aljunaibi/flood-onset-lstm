"""Screenshots of every screen of the web presentation (and of the replay in use) with Playwright + the installed Chrome: python code/proposal_web/shots.py <html> <outdir>"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

html = pathlib.Path(sys.argv[1]).resolve().as_uri()
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    pg = br.new_page(viewport={"width": 1600, "height": 944})
    errs = []
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    pg.goto(html); pg.wait_for_timeout(1000)
    n = pg.evaluate("slides.length")
    for i in range(n):
        pg.evaluate(f"go({i})"); pg.wait_for_timeout(1300 if i == 8 else 700)
        pg.screenshot(path=str(out / f"s{i + 1:02d}.png"))
    pg.evaluate("go(8)"); pg.wait_for_timeout(400)
    for _ in range(2):
        pg.click("#btnNext"); pg.wait_for_timeout(200)
    for _ in range(3):
        pg.click("#btnStep"); pg.wait_for_timeout(120)
    pg.screenshot(path=str(out / "s09_replay_onset.png"))
    pg.click("#btnPlay"); pg.wait_for_timeout(2500); pg.click("#btnPlay")
    pg.mouse.move(600, 500); pg.wait_for_timeout(300)
    pg.screenshot(path=str(out / "s09_replay_play.png"))
    pg.keyboard.press("n"); pg.wait_for_timeout(500)
    pg.screenshot(path=str(out / "s09_notes.png"))
    print("screens:", n, "| errors:", errs or "none")
    br.close()
