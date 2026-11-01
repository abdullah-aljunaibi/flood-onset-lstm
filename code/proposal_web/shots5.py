"""Screenshots of the five-screen web presentation, with its interactions exercised:
python code/proposal_web/shots5.py <html> <outdir> [dark]
With the word 'dark' the script presses D first, so every screenshot shows the dark theme (the theme is read back from <html data-theme>)."""
import pathlib
import sys

from playwright.sync_api import sync_playwright

html = pathlib.Path(sys.argv[1]).resolve().as_uri()
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dark = "dark" in sys.argv[3:]
with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    pg = br.new_page(viewport={"width": 1600, "height": 944})
    errs = []
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    pg.goto(html); pg.wait_for_timeout(1000)
    if dark:
        pg.keyboard.press("d"); pg.wait_for_timeout(400)
    theme = pg.evaluate("document.documentElement.getAttribute('data-theme')")
    assert theme == ("dark" if dark else "light"), theme
    n = pg.evaluate("slides.length")
    for i in range(n):
        pg.evaluate(f"go({i})"); pg.wait_for_timeout(1400)
        pg.screenshot(path=str(out / f"s{i + 1}.png"))
    # screen 2: the third stop and a branch of the mind map
    pg.evaluate("go(1)"); pg.wait_for_timeout(300); pg.keyboard.press("3"); pg.wait_for_timeout(900)
    pg.click("#rqmap .node[data-i='1'] rect", force=True); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "s2_stop3_rq2.png"))
    # screen 3: hover a product; hover a basin of the map; click another basin (its elevation panel replaces the first); click it again (the card empties)
    pg.evaluate("go(3)"); pg.wait_for_timeout(500); pg.hover("#datamap .node[data-i='1']"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "s3_hover.png"))
    pg.hover("#bmap .basin[data-id='02108000'] .hit"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "s3_map_hover.png"))
    pg.click("#bmap .basin[data-id='03473000'] .hit"); pg.mouse.move(1000, 700); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "s3_map_click.png"))
    pg.click("#bmap .basin[data-id='03473000'] .hit"); pg.mouse.move(1000, 700); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "s3_map_empty.png"))
    pg.click("#bmap .basin[data-id='02296500'] .hit"); pg.mouse.move(1000, 700); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "s3_map_south.png"))
    # screen 4: the replay after two onsets and a method tooltip
    pg.evaluate("go(4)"); pg.wait_for_timeout(500)
    for _ in range(2):
        pg.click("#btnNext"); pg.wait_for_timeout(150)
    for _ in range(3):
        pg.click("#btnStep"); pg.wait_for_timeout(100)
    pg.hover(".mstrip > div:nth-child(5)"); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "s4_replay.png"))
    # screen 5: hover a ladder bar, click week 2
    pg.evaluate("go(5)"); pg.wait_for_timeout(1200); pg.click("#tl div[data-i='1']"); pg.hover("#ladder .bar[data-i='3']"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "s5_hover.png"))
    pg.keyboard.press("n"); pg.wait_for_timeout(500); pg.screenshot(path=str(out / "s5_notes.png"))
    pg.keyboard.press("n"); pg.keyboard.press("?"); pg.wait_for_timeout(400); pg.screenshot(path=str(out / "s_help.png")); pg.keyboard.press("Escape")
    print("screens:", n, "| theme:", theme, "| errors:", errs or "none")
    br.close()
