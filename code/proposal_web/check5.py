"""Checks of the five-screen web presentation: python code/proposal_web/check5.py <html>
 1. no console or page errors, in light and in dark, with normal and with reduced motion
 2. the theme: key D toggles <html data-theme>, the choice survives a reload, and a blocked localStorage does not break anything
 3. reduced motion: nothing animates; the cover photograph is still there and decoded
 4. every screen: text that leaves the 1600 x 900 stage
 5. the map of screen 3: 12 labels, none touching a ring, another label, the furniture or the frame; the legend over the sea
 6. the cover: contrast of every text against the photograph (background rendered with the text hidden; 95th percentile of the brightest pixels)
Prints one line per check; exits 1 if something fails."""
import io
import pathlib
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

html = pathlib.Path(sys.argv[1]).resolve().as_uri()
bad = []


def say(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        bad.append(msg)


def lum(rgb):
    c = np.asarray(rgb, float) / 255
    c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    return c @ np.array([.2126, .7152, .0722])


STAGE_JS = """() => { const out = [], st = document.querySelector('#stage').getBoundingClientRect(), k = st.width / 1600;
  const sl = slides[cur]; sl.querySelectorAll('*').forEach(el => {
    if (el.closest('.rain, .rv, .rfoot, .cphoto, .cveil, defs, symbol')) return;
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (!own) return; const r = el.getBoundingClientRect(); if (!r.width) return;
    const l = (r.left - st.left) / k, t = (r.top - st.top) / k, rr = (r.right - st.left) / k, b = (r.bottom - st.top) / k;
    if (l < -1 || t < -1 || rr > 1601 || b > 901) out.push([el.tagName + '.' + (el.className.baseVal ?? el.className) + ' ' + el.textContent.trim().slice(0, 30), Math.round(l), Math.round(t), Math.round(rr), Math.round(b)]);
  }); return out; }"""

MAP_JS = """() => {
  const svg = document.querySelector('#bmap'), u = BM.u, [vx, vy, vw, vh] = MJ.viewBox, labs = [...svg.querySelectorAll('.lab')].filter(t => t.style.display !== 'none');
  const R = labs.map(t => { const b = t.getBBox(); return { id: t.dataset.id, name: t.textContent, box: [b.x, b.y, b.x + b.width, b.y + b.height] }; });
  const keep = [...svg.querySelectorAll('.keep')].map(g => { const b = g.getBBox(); return [g.id, [b.x, b.y, b.x + b.width, b.y + b.height]]; });
  const touch = (A, B) => !(A[2] <= B[0] || B[2] <= A[0] || A[3] <= B[1] || B[3] <= A[1]);
  const hitsRing = (r, c) => { const dx = Math.max(r[0] - c[0], 0, c[0] - r[2]), dy = Math.max(r[1] - c[1], 0, c[1] - r[3]); return dx * dx + dy * dy < c[2] * c[2]; };
  const issues = [];
  R.forEach((a, i) => {
    if (a.box[0] < vx || a.box[2] > vx + vw || a.box[1] < vy || a.box[3] > vy + vh) issues.push(a.name + ' leaves the frame');
    MJ.basins.forEach(b => { if (hitsRing(a.box, b.ring)) issues.push(a.name + ' touches the ring of ' + b.name); });
    keep.forEach(([n, k]) => { if (touch(a.box, k)) issues.push(a.name + ' touches ' + n); });
    R.slice(i + 1).forEach(c => { if (touch(a.box, c.box)) issues.push(a.name + ' touches ' + c.name); });
  });
  // the legend must be over the sea: no point of its box may be land (the coast is a hairline, the land a fill: test a grid of points of the box)
  const land = svg.querySelector('.land'), leg = keep.find(k => k[0] === 'bleg')[1], pt = svg.createSVGPoint(); let onLand = 0;
  for (let x = leg[0]; x <= leg[2]; x += 4 * u) for (let y = leg[1]; y <= leg[3]; y += 4 * u) { pt.x = x; pt.y = y; if (land.isPointInFill(pt)) onLand++; }
  const mb = document.querySelector('#mapbox').getBoundingClientRect(), st = document.querySelector('#stage').getBoundingClientRect(), k = st.width / 1600;
  return { labels: R.length, issues, legendLandPoints: onLand, mapPx: [Math.round(mb.width / k), Math.round(mb.height / k)], leftColBottom: Math.round((document.querySelector('.s3 .lcol').getBoundingClientRect().bottom - st.top) / k),
           capBottom: Math.round((document.querySelector('#mapcap').getBoundingClientRect().bottom - st.top) / k) };
}"""

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    for dark in (False, True):
        for reduced in (False, True):
            tag = ("dark" if dark else "light") + (" + reduced motion" if reduced else "")
            ctx = br.new_context(viewport={"width": 1600, "height": 944}, reduced_motion="reduce" if reduced else "no-preference")
            pg = ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
            pg.goto(html); pg.wait_for_timeout(700)
            if dark:
                pg.keyboard.press("d"); pg.wait_for_timeout(200)
            for i in range(5):
                pg.evaluate(f"go({i})"); pg.wait_for_timeout(900)
                over = pg.evaluate(STAGE_JS)
                say(not over, f"[{tag}] screen {i + 1}: text inside the stage" + (f" -- {over[:4]}" if over else ""))
            pg.evaluate("go(3)"); pg.wait_for_timeout(600)
            m = pg.evaluate(MAP_JS)
            say(m["labels"] == 12 and not m["issues"] and m["legendLandPoints"] == 0, f"[{tag}] map: {m['labels']} labels, legend land points {m['legendLandPoints']}, issues {m['issues'] or 'none'}; map {m['mapPx']} px, caption ends at y={m['capBottom']}")
            say(m["capBottom"] <= 822, f"[{tag}] screen 3: the caption under the map ends at y={m['capBottom']} (the content area of a slide ends at 828)")
            if reduced:
                n = pg.evaluate("document.getAnimations().filter(a => a.playState === 'running').length")
                pg.evaluate("go(0)"); pg.wait_for_timeout(500)
                ph = pg.evaluate("(() => { const i = document.querySelector('.cphoto'); const cs = getComputedStyle(i); return [i.complete, i.naturalWidth, cs.display, cs.opacity] })()")
                say(n == 0 and ph[0] and ph[1] == 1600, f"[{tag}] reduced motion: {n} running animations; cover photograph complete={ph[0]} width={ph[1]} display={ph[2]} opacity={ph[3]}")
            say(not errs, f"[{tag}] console and page errors: {errs or 'none'}")
            ctx.close()
    # the map: the first basin is the one of the cover chart; a click on a label swaps the card, the same click clears it; hover names the basin; Enter on a focused basin picks it
    ctx = br.new_context(viewport={"width": 1600, "height": 944}); pg = ctx.new_page(); pg.goto(html); pg.wait_for_timeout(600); pg.evaluate("go(3)"); pg.wait_for_timeout(600)
    st = lambda: pg.evaluate("[BM.sel, document.querySelector('#elev').classList.contains('empty'), document.querySelector('#elevName').textContent, document.querySelector('#elevImg').naturalWidth]")
    first = st(); cover = pg.evaluate("document.querySelector('#ch-title').textContent")
    pg.click("#bmap .lab[data-id='03473000']"); second = st(); pg.click("#bmap .lab[data-id='03473000']"); third = st()
    pg.hover("#bmap .basin[data-id='02472000'] .hit"); pg.wait_for_timeout(200); tip = pg.evaluate("[getComputedStyle(document.querySelector('#gtip')).display, document.querySelector('#gtip').textContent]")
    pg.mouse.move(800, 800); pg.focus("#bmap .basin[data-id='02221525']"); pg.keyboard.press("Enter"); fourth = st()
    say(first[0] == "02128000" and cover.startswith(first[2] + ",") and first[3] == 330 and second[0] == "03473000" and second[2] == "SF Holston, VA" and third[0] is None and third[1] and fourth[0] == "02221525",
        f"map interactions: first {first[2]} (the cover chart starts: {cover[:24]}), after a click on a label {second[2]}, after the same click {third[0]}/empty={third[1]}, Enter on Murder Creek {fourth[2]}; panel {first[3]} px wide")
    say(tip[0] == "block" and "Leaf River, MS" in tip[1] and "1,928 km" in tip[1] and "63 to 194 m" in tip[1] and "Southern lowlands" in tip[1], f"map hover tooltip: {tip[1][:150]}")
    ctx.close()
    # the theme: toggle, remember, blocked storage
    ctx = br.new_context(viewport={"width": 1600, "height": 944}); pg = ctx.new_page()
    pg.goto(html); pg.wait_for_timeout(500)
    th = lambda: pg.evaluate("document.documentElement.getAttribute('data-theme')")
    a = th(); pg.keyboard.press("d"); b = th(); pg.reload(); pg.wait_for_timeout(500); c = th(); pg.click("#bTheme"); d = th(); pg.reload(); pg.wait_for_timeout(500); e = th()
    say((a, b, c, d, e) == ("light", "dark", "dark", "light", "light"), f"theme: default {a}, after D {b}, after reload {c}, after the button {d}, after reload {e}")
    ctx.close()
    ctx = br.new_context(viewport={"width": 1600, "height": 944}); ctx.add_init_script("Object.defineProperty(window, 'localStorage', {get() { throw new Error('blocked'); }})"); pg = ctx.new_page(); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e))); pg.goto(html); pg.wait_for_timeout(500); a = th(); pg.keyboard.press("d"); b = th()
    say((a, b) == ("light", "dark") and not errs, f"theme with localStorage blocked: {a} -> {b}, errors {errs or 'none'}")
    ctx.close()
    # the cover: contrast of its text against the photograph
    pg = br.new_context(viewport={"width": 1600, "height": 944}).new_page(); pg.goto(html); pg.wait_for_timeout(800); pg.evaluate("go(0)"); pg.wait_for_timeout(500)
    items = pg.evaluate("""() => { const out = [], w = document.createTreeWalker(document.querySelector('.cover-grid > div:first-child'), NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) { if (!n.textContent.trim()) continue; const el = n.parentElement, cs = getComputedStyle(el), rg = document.createRange(); rg.selectNodeContents(n);
        for (const r of rg.getClientRects()) out.push([n.textContent.trim().slice(0, 28), r.left, r.top, r.width, r.height, cs.color, parseFloat(cs.fontSize), cs.fontWeight]); } return out; }""")
    pg.add_style_tag(content=".cover-grid *{color:transparent !important;-webkit-text-fill-color:transparent !important;border-color:transparent !important} .cover .ic{opacity:0 !important} .rain,.rv{display:none !important}")
    pg.wait_for_timeout(300)
    im = Image.open(io.BytesIO(pg.screenshot())).convert("RGB"); A = np.asarray(im)
    worst = (99, "")
    for name, x, y, w, h, col, fs, fw in items:
        if w < 2 or h < 2: continue
        rgb = [int(v) for v in col[col.index("(") + 1:col.index(")")].split(",")[:3]]
        reg = A[int(y):int(y + h) + 1, int(x):int(x + w) + 1].reshape(-1, 3); L = np.percentile(lum(reg), 95)
        cr = (lum(rgb) + .05) / (L + .05); need = 3 if (fs >= 24 or (fs >= 18.66 and int(fw) >= 700) or not any(ch.isalnum() for ch in name)) else 4.5      # large text, and the arrows between the steps (graphics), need 3:1
        if any(ch.isalnum() for ch in name) and cr < worst[0]: worst = (cr, name)
        if cr < need: say(False, f"cover text '{name}' ({fs:g}px) contrast {cr:.1f} < {need}")
    say(worst[0] >= 4.5, f"cover: {len(items)} text lines, lowest contrast against the photograph {worst[0]:.1f}:1 ('{worst[1]}'; the limit is 4.5:1 for small text, 3:1 for large)")
    br.close()
print("\nALL CHECKS PASSED" if not bad else f"\n{len(bad)} CHECK(S) FAILED")
sys.exit(1 if bad else 0)
