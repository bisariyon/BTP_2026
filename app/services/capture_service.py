"""
app/services/capture_service.py

Single-URL capture service.

Uses Playwright's SYNC API inside a ThreadPoolExecutor to avoid the
NotImplementedError that occurs when running async Playwright inside
an already-running asyncio event loop (FastAPI / uvicorn on Windows).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

# ─── Output directories ──────────────────────────────────────────────────────
SCREENSHOTS_DIR = Path("app/static/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

DOM_DIR = Path("data/captures")
DOM_DIR.mkdir(parents=True, exist_ok=True)

MAX_FULLPAGE_HEIGHT = 12_000  # px

# ─── Thread pool (1 worker keeps Playwright happy; increase if needed) ───────
_executor = ThreadPoolExecutor(max_workers=2)


# ─── Internal sync helpers ───────────────────────────────────────────────────

def _setup_page_sync(page):
    page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    def _handler(route):
        blocked = [
            "googletagmanager", "doubleclick", "googlesyndication",
            "adsystem", "analytics",
        ]
        if any(b in route.request.url for b in blocked):
            route.abort()
        else:
            route.continue_()

    page.route("**/*", _handler)


def _hide_fixed_elements_sync(page):
    try:
        page.evaluate("""
          () => {
            const els = Array.from(document.querySelectorAll('*')).filter(el => {
              try {
                const s = window.getComputedStyle(el);
                return (s.position === 'fixed' || s.position === 'sticky')
                       && el.offsetWidth > 0 && el.offsetHeight > 0;
              } catch { return false; }
            });
            els.forEach(el => {
              el.setAttribute('data-ui-hidden','1');
              el._prev_vis = el.style.visibility || '';
              el.style.visibility = 'hidden';
            });
          }
        """)
    except Exception:
        pass


def _restore_hidden_elements_sync(page):
    try:
        page.evaluate("""
          () => {
            Array.from(document.querySelectorAll('[data-ui-hidden]')).forEach(el => {
              el.style.visibility = el._prev_vis || '';
              el.removeAttribute('data-ui-hidden');
              delete el._prev_vis;
            });
          }
        """)
    except Exception:
        pass


def _capture_fullpage_sync(page, out_path: str) -> str:
    dims = page.evaluate(
        "() => ({h: document.documentElement.scrollHeight, vh: window.innerHeight})"
    )
    total_h = int(dims["h"])
    vh = int(dims["vh"])

    _hide_fixed_elements_sync(page)
    import time; time.sleep(0.12)

    if total_h <= MAX_FULLPAGE_HEIGHT:
        page.screenshot(path=out_path, full_page=True)
    else:
        tiles, y, i = [], 0, 0
        while y < total_h and i < 100:
            page.evaluate(f"window.scrollTo(0, {y})")
            time.sleep(0.15)
            tile = out_path.replace(".png", f"_tile{i}.png")
            page.screenshot(path=tile, full_page=False)
            tiles.append(tile)
            y += vh
            i += 1
        imgs = [cv2.imread(t) for t in tiles if os.path.exists(t)]
        if imgs:
            minw = min(img.shape[1] for img in imgs)
            imgs = [
                cv2.resize(img, (minw, int(img.shape[0] * minw / img.shape[1])))
                if img.shape[1] != minw else img
                for img in imgs
            ]
            cv2.imwrite(out_path, cv2.vconcat(imgs))
        for t in tiles:
            try: os.remove(t)
            except: pass

    _restore_hidden_elements_sync(page)
    return out_path


def _extract_dom_sync(page, max_interactive: int = 80) -> dict:
    try:
        dom = page.evaluate(f"""
            () => {{
                try {{
                    function compactText(s, n=120) {{ return s ? s.trim().replace(/\\s+/g,' ').slice(0,n) : null; }}

                    const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4')).slice(0,30).map(h => {{
                        const s = window.getComputedStyle(h);
                        return {{ tag: h.tagName, text: compactText(h.textContent, 120), fontSize: s.fontSize }};
                    }});

                    const interactiveSel = ['a','button','input','textarea','select','label'];
                    let interactive = [];
                    const perSel = Math.max(1, Math.ceil({max_interactive} / interactiveSel.length));
                    interactiveSel.forEach(sel => {{
                        Array.from(document.querySelectorAll(sel)).slice(0, perSel).forEach(el => {{
                            try {{
                                const r = el.getBoundingClientRect();
                                const style = window.getComputedStyle(el);
                                interactive.push({{
                                    tag: el.tagName,
                                    id: el.id || null,
                                    text: compactText(el.textContent, 80),
                                    visible: (r.width>0 && r.height>0 && style.display!=='none'),
                                    x: Math.round(r.x), y: Math.round(r.y),
                                    width: Math.round(r.width), height: Math.round(r.height),
                                    aria: {{ label: el.getAttribute('aria-label') || null, hidden: el.getAttribute('aria-hidden') || null }}
                                }});
                            }} catch {{ }}
                        }});
                    }});

                    const forms = Array.from(document.querySelectorAll('form')).slice(0,20).map(f => ({{
                        action: f.getAttribute('action') || null,
                        method: (f.getAttribute('method')||'get').toLowerCase(),
                        input_count: f.querySelectorAll('input,textarea,select').length
                    }}));

                    const imgs = Array.from(document.querySelectorAll('img')).slice(0,200);
                    const imagesWithoutAlt = imgs.filter(i => !i.getAttribute('alt')).length;

                    const totalElements = document.querySelectorAll('*').length;
                    const flexContainers = document.querySelectorAll('[style*="display: flex"],[class*="flex"]').length;
                    const gridContainers = document.querySelectorAll('[style*="display: grid"],[class*="grid"]').length;

                    const links = Array.from(document.querySelectorAll('a'));
                    const linksWithoutText = links.filter(a => !(a.textContent && a.textContent.trim()) && !a.getAttribute('aria-label')).length;

                    const bodyText = document.body.innerText || '';
                    const wordCount = bodyText.split(/\\s+/).filter(Boolean).length;

                    const domStruct = Array.from(document.querySelectorAll('main,header,nav,footer,section,article,h1,h2,h3')).slice(0,80)
                        .map(e => e.tagName + (e.id||'') + Array.from(e.classList).slice(0,3).join('')).join('');
                    const domHash = btoa(domStruct).slice(0,32);

                    return {{
                        timestamp: new Date().toISOString(),
                        url: location.href,
                        title: document.title,
                        viewport: {{ width: window.innerWidth, height: window.innerHeight }},
                        headings: headings,
                        interactive_count: interactive.length,
                        interactive: interactive.slice(0, {max_interactive}),
                        forms: forms,
                        images_count: imgs.length,
                        imagesWithoutAlt: imagesWithoutAlt,
                        layout: {{ totalElements, flexContainers, gridContainers }},
                        accessibility: {{ linksWithoutText }},
                        textDensity: {{ totalTextLength: bodyText.length, wordCount }},
                        domStructureHash: domHash
                    }};
                }} catch(e) {{
                    return {{ error: 'dom extraction failed: ' + (e && e.message), timestamp: new Date().toISOString(), domStructureHash: 'error' }};
                }}
            }}
        """)
        return dom
    except Exception as exc:
        return {"error": str(exc), "timestamp": datetime.now().isoformat(), "domStructureHash": "error"}


# ─── Main sync capture function (runs in thread) ─────────────────────────────

def _do_capture(url: str, img_path: str, dom_path: str) -> dict:
    """Playwright sync capture — runs inside a thread."""
    import time

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            device_scale_factor=1,
        )
        page = context.new_page()
        try:
            _setup_page_sync(page)
            page.goto(url, wait_until="networkidle", timeout=45_000)
            time.sleep(1.5)

            dom = _extract_dom_sync(page)
            with open(dom_path, "w", encoding="utf-8") as f:
                json.dump(dom, f, indent=2, ensure_ascii=False)

            _capture_fullpage_sync(page, img_path)

            page_title = page.title()
            page_url = page.url
        finally:
            browser.close()

    return {"page_title": page_title, "page_url": page_url}


# ─── Public async API ────────────────────────────────────────────────────────

async def capture_single_url(url: str) -> dict:
    """
    Async wrapper: runs the sync Playwright capture in a thread pool
    so it doesn't block or conflict with the FastAPI event loop.

    Returns:
        {
            "image_path": str,
            "dom_json_path": str,
            "session_id": str,
            "page_title": str,
            "page_url": str,
            "screenshot_url": str,   # relative URL for <img src="...">
        }
    """
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]

    img_filename = f"{session_id}.png"
    img_path = str(SCREENSHOTS_DIR / img_filename)

    dom_filename = f"{session_id}_dom.json"
    dom_path = str(DOM_DIR / dom_filename)

    loop = asyncio.get_event_loop()
    extra = await loop.run_in_executor(
        _executor,
        _do_capture,
        url, img_path, dom_path,
    )

    return {
        "image_path": img_path,
        "dom_json_path": dom_path,
        "session_id": session_id,
        "page_title": extra["page_title"],
        "page_url": extra["page_url"],
        "screenshot_url": f"/static/screenshots/{img_filename}",
    }
