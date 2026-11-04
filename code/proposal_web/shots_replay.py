"""Screenshots of the live demo (live_demo/index.html, or its URL), with its interactions exercised:
python code/proposal_web/shots_replay.py <html-or-url> <outdir> [dark]
With the word 'dark' the script presses D first, so every screenshot shows the dark theme (the theme is read back from <html data-theme>).
Prints the console errors (none expected) and a few numbers read back from the page."""
import pathlib
import sys

from playwright.sync_api import sync_playwright

src = sys.argv[1]
html = src if "://" in src else pathlib.Path(src).resolve().as_uri()
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dark = "dark" in sys.argv[3:]
with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    pg = br.new_page(viewport={"width": 1600, "height": 944})
    errs = []
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    pg.goto(html); pg.wait_for_timeout(1200)
    if dark:
        pg.keyboard.press("d"); pg.wait_for_timeout(400)
    theme = pg.evaluate("document.documentElement.getAttribute('data-theme')")
    assert theme == ("dark" if dark else "light"), theme
    pg.screenshot(path=str(out / "r1_open.png"))
    # two onsets on, three days stepped, a method tooltip
    for _ in range(2):
        pg.click("#btnNext"); pg.wait_for_timeout(150)
    for _ in range(3):
        pg.click("#btnStep"); pg.wait_for_timeout(100)
    pg.hover(".mstrip > div:nth-child(3)"); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "r2_steps_tip.png"))
    # the rain to midnight, 1 day ahead: the real-time radar diamonds; play to the end of the year, then hover the last onset
    pg.mouse.move(5, 5); pg.click("#levels button[data-k='wqa']"); pg.select_option("#selSpeed", "25"); pg.click("#btnPlay"); pg.wait_for_timeout(9000)
    pg.click("#btnPlay") if pg.evaluate("R.playing") else None
    pg.wait_for_timeout(300); pg.screenshot(path=str(out / "r3_wqa_year.png"))
    t = pg.evaluate("(() => { const d = dec(B()); for (let i = R.today; i >= win()[0]; i--) if (onset(d, i, 1)) return i; return -1; })()")
    if t >= 0:
        box = pg.locator("#chart").bounding_box()
        xx = pg.evaluate(f"G.x({t})") * box["width"] / 1180 + box["x"]
        pg.mouse.move(xx, box["y"] + box["height"] * .5); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "r4_onset_hover.png"))
    # the story: the button, every step with the arrow key, Esc at the end
    pg.mouse.move(5, 5); pg.click("#btnStory"); pg.wait_for_timeout(500)
    n = pg.evaluate("STORY.steps.length")
    for i in range(n):
        pg.screenshot(path=str(out / f"s{i + 1}.png"))
        if i < n - 1:
            pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(450)
    pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    story_closed = pg.evaluate("R.story === null && document.getElementById('story').hidden")
    pg.keyboard.press("?"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "r5_help.png")); pg.keyboard.press("Escape")
    # the address #story opens it directly; a phone-width window scales the stage down whole
    pg2 = br.new_page(viewport={"width": 1600, "height": 944}); pg2.goto(html.split("#")[0] + "#story"); pg2.wait_for_timeout(1500)
    hash_opens = pg2.evaluate("R.story === 0"); pg2.close()

    pg.set_viewport_size({"width": 390, "height": 844}); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "r6_phone.png"))
    print("theme:", theme, "| story steps:", n, "| closed by Esc:", story_closed, "| #story opens:", hash_opens, "| errors:", errs or "none")
    br.close()
