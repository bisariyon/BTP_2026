# smart_capture_ml_ready_with_dom.py
# Playwright capture tool with full-page screenshots + ML-friendly DOM extraction.

import asyncio
from playwright.async_api import async_playwright
import os
import json
import time
import random
from datetime import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ------- CONFIG -------
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SESSION_ID = datetime.now().strftime("session_%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(OUTPUT_DIR, SESSION_ID)
os.makedirs(SESSION_DIR, exist_ok=True)

# thresholds
SIMILARITY_THRESHOLD = 0.75
MIN_CHANGE_INTERVAL = 2.5   # seconds
STABILITY_CHECK_DURATION = 1.8
MAX_FULLPAGE_HEIGHT = 12000  # if page taller than this, use tiled capture (stitching)

# color clustering size for palette (small k to save time)
COLOR_K = 4

# ------- HELPER / DETECTOR -------
class UIChangeDetector:
    def __init__(self, similarity_threshold=SIMILARITY_THRESHOLD, min_change_interval=MIN_CHANGE_INTERVAL, stability_check_duration=STABILITY_CHECK_DURATION):
        self.previous_screenshot = None  # path to last saved full-page screenshot
        self.previous_dom_hash = None
        self.similarity_threshold = similarity_threshold
        self.min_change_interval = min_change_interval
        self.stability_check_duration = stability_check_duration
        self.last_capture_time = 0
        self.capture_count = 0
        self.recent_changes = deque()

    def calculate_image_similarity(self, img1_path, img2_path):
        try:
            a = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            if a is None or b is None:
                return 0.0
            # resize b to a
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            score = ssim(a, b, full=False)
            return float(score)
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.0

    async def wait_for_stability(self, page, check_duration=None):
        if check_duration is None:
            check_duration = self.stability_check_duration
        stable_start = time.time()
        last_dom = None
        try:
            while time.time() - stable_start < check_duration:
                current_dom = await page.evaluate("""
                    () => {
                        const elems = Array.from(document.querySelectorAll('main, article, section, header, nav, footer')).slice(0,60);
                        return btoa(elems.map(e=> e.tagName + (e.id||'') + Array.from(e.classList).slice(0,2).join('')).join('')).slice(0,16);
                    }
                """)
                if last_dom and last_dom != current_dom:
                    stable_start = time.time()
                last_dom = current_dom
                await asyncio.sleep(0.25)
        except Exception as e:
            print(f"Stability check error: {e}")
        return True

    def should_capture(self, current_dom_hash, temp_screenshot_path=None):
        now = time.time()
        if now - self.last_capture_time < self.min_change_interval:
            return False, 'Too soon since last capture'

        dom_changed = (self.previous_dom_hash != current_dom_hash)
        visual_changed = True
        visual_sim = 1.0
        if self.previous_screenshot and temp_screenshot_path:
            visual_sim = self.calculate_image_similarity(self.previous_screenshot, temp_screenshot_path)
            visual_changed = visual_sim < self.similarity_threshold

        should = (dom_changed and visual_changed) or (visual_sim < 0.7)
        reasons = []
        if dom_changed: reasons.append('DOM changed')
        if visual_changed: reasons.append(f'Visual changed (sim={visual_sim:.3f})')
        if not should: reasons.append('No significant change')
        return should, ', '.join(reasons)


# ------- VISUAL FEATURE HELPERS -------

def extract_color_palette(image_path, k=COLOR_K):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        small = cv2.resize(img, (200,200), interpolation=cv2.INTER_AREA)
        data = small.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(int)
        palette = [f"rgb({int(c[2])},{int(c[1])},{int(c[0])})" for c in centers]
        return palette
    except Exception as e:
        print(f"Palette error: {e}")
        return []


def compute_whitespace_ratio(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        h,w = img.shape
        edges = cv2.Canny(img, 80, 150)
        content = np.count_nonzero(edges)
        total = h*w
        whitespace = 1.0 - (content/total)
        return float(max(0.0, min(1.0, whitespace)))
    except Exception as e:
        print(f"Whitespace error: {e}")
        return 0.0


# ------- PAGE CAPTURE HELPERS -------
async def hide_fixed_elements(page):
    """Hide fixed/sticky elements to avoid them being repeated in full-page screenshots.
    Returns number of elements hidden."""
    try:
        cnt = await page.evaluate("""
            () => {
                const els = Array.from(document.querySelectorAll('*')).filter(el=>{
                    try{ const s = window.getComputedStyle(el); return (s.position==='fixed' || s.position==='sticky') && el.offsetWidth>0 && el.offsetHeight>0; }catch(e){ return false; }
                });
                els.forEach(el=>{ el.setAttribute('data-ui-capture-hidden','1'); el._ui_prev_visibility = el.style.visibility || ''; el.style.visibility='hidden'; });
                return els.length;
            }
        """)
        return cnt
    except Exception as e:
        print(f"hide_fixed_elements error: {e}")
        return 0

async def restore_hidden_elements(page):
    try:
        cnt = await page.evaluate("""
            () => {
                const els = Array.from(document.querySelectorAll('[data-ui-capture-hidden]'));
                els.forEach(el=>{ el.style.visibility = el._ui_prev_visibility || ''; el.removeAttribute('data-ui-capture-hidden'); delete el._ui_prev_visibility; });
                return els.length;
            }
        """)
        return cnt
    except Exception as e:
        print(f"restore_hidden_elements error: {e}")
        return 0

async def capture_fullpage_with_fallback(page, out_path_base):
    """Try Playwright full_page capture after hiding fixed elements. If page height is huge, fall back to tiled captures and stitch.
    Returns path to final stitched image (out_path_base + .png)."""
    try:
        # get page dimensions
        dims = await page.evaluate("() => ({w: Math.max(document.documentElement.clientWidth, window.innerWidth||0), h: document.documentElement.scrollHeight, vh: window.innerHeight, dpr: window.devicePixelRatio || 1})")
        width = int(dims['w'])
        total_height = int(dims['h'])
        viewport_h = int(dims['vh'])

        final_path = out_path_base + '.png'

        # Hide fixed elements
        await hide_fixed_elements(page)
        await asyncio.sleep(0.12)

        if total_height <= MAX_FULLPAGE_HEIGHT:
            # Playwright full page screenshot
            await page.screenshot(path=final_path, full_page=True)
            await restore_hidden_elements(page)
            return final_path
        else:
            # tiled capture and stitch
            tiles = []
            y = 0
            i = 0
            while y < total_height:
                await page.evaluate(f'window.scrollTo(0, {y})')
                await asyncio.sleep(0.15)
                tile_path = f"{out_path_base}_tile_{i}.png"
                await page.screenshot(path=tile_path, full_page=False)
                tiles.append(tile_path)
                i += 1
                y += viewport_h
                # safety cap
                if i > 100:
                    break

            # stitch tiles vertically
            imgs = [cv2.imread(t) for t in tiles if cv2.imread(t) is not None]
            if not imgs:
                await restore_hidden_elements(page)
                raise RuntimeError('No tiles captured')

            # ensure consistent width
            minw = min(img.shape[1] for img in imgs)
            resized = [cv2.resize(img, (minw, int(img.shape[0] * (minw/img.shape[1])))) if img.shape[1]!=minw else img for img in imgs]
            stitched = cv2.vconcat(resized)
            cv2.imwrite(final_path, stitched)

            # cleanup tiles
            for t in tiles:
                try: os.remove(t)
                except: pass

            await restore_hidden_elements(page)
            return final_path
    except Exception as e:
        try: await restore_hidden_elements(page)
        except: pass
        raise


# ------- DOM ANALYSIS & EVENT INJECTION -------
async def inject_event_listeners(page):
    try:
        await page.evaluate("""
            (()=>{
                if (window.__ui_capture_installed) return true;
                window.__ui_events = [];
                function push(e){ try{ window.__ui_events.push(e); if(window.__ui_events.length>400) window.__ui_events.shift(); }catch(e){} }
                ['click','submit','input','change'].forEach(ev=> document.addEventListener(ev, function(event){ try{ const t=event.target; push({type: ev, ts: Date.now(), tag: t && t.tagName, id: t && t.id, classes: t && t.className && t.className.toString().slice(0,200) }); }catch(e){} }, true));
                window.__ui_capture_installed = true; return true;
            })();
        """)
    except Exception as e:
        print(f"inject listeners error: {e}")

async def pull_ui_events(page):
    try:
        evs = await page.evaluate("""(() => { const e = window.__ui_events ? window.__ui_events.splice(0, window.__ui_events.length) : []; return e; })()""")
        return evs or []
    except Exception as e:
        print(f"pull_ui_events error: {e}")
        return []

async def get_dom_hash(page):
    try:
        h = await page.evaluate("""
            () => {
                const elems = Array.from(document.querySelectorAll('main, article, section, header, nav, footer, h1, h2, h3')).slice(0,80);
                const s = elems.map(e=> e.tagName + (e.id||'') + Array.from(e.classList).slice(0,2).join('')).join('');
                return btoa(s).slice(0,24);
            }
        """)
        return h
    except Exception as e:
        print(f"dom hash error: {e}")
        return 'error'

# NEW: Extract relevant, ML-friendly DOM info
# Replace your existing extract_relevant_dom with this corrected function
async def extract_relevant_dom(page, max_interactive=120):
    """
    Return a compact, ML-friendly DOM summary:
     - headings, interactive elements with geometry & computed styles,
     - forms summary, images & alt analysis, layout stats, accessibility counts.
    Limits the size of arrays so metadata stays reasonable.
    """
    try:
        dom = await page.evaluate(f"""
            () => {{
                try {{
                    function compactText(s, n=120) {{ return s ? s.trim().replace(/\\s+/g,' ').slice(0,n) : null; }}

                    // Viewport
                    const viewport = {{ width: window.innerWidth, height: window.innerHeight, scrollX: window.scrollX, scrollY: window.scrollY }};

                    // Headings
                    const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4')).slice(0,30).map(h => {{
                        const s = window.getComputedStyle(h);
                        return {{ tag: h.tagName, text: compactText(h.textContent, 120), fontSize: s.fontSize }};
                    }});

                    // Interactive elements (balanced across selectors)
                    const interactiveSel = ['a','button','input','textarea','select','label'];
                    let interactive = [];
                    const perSel = Math.max(1, Math.ceil({max_interactive} / interactiveSel.length));
                    interactiveSel.forEach(sel => {{
                        Array.from(document.querySelectorAll(sel)).slice(0, perSel).forEach(el => {{
                            try {{
                                const r = el.getBoundingClientRect();
                                const style = window.getComputedStyle(el);
                                const role = el.getAttribute('role') || null;
                                const aria = {{
                                    label: el.getAttribute('aria-label') || null,
                                    hidden: el.getAttribute('aria-hidden') || null
                                }};
                                interactive.push({{
                                    tag: el.tagName,
                                    id: el.id || null,
                                    classes: Array.from(el.classList).slice(0,5),
                                    role: role,
                                    aria: aria,
                                    name: el.getAttribute('name') || null,
                                    type: el.getAttribute('type') || null,
                                    href: el.getAttribute('href') || null,
                                    text: compactText(el.textContent, 80),
                                    placeholder: el.getAttribute('placeholder') || null,
                                    visible: (r.width>0 && r.height>0 && style.display!=='none'),
                                    x: Math.round(r.x),
                                    y: Math.round(r.y),
                                    width: Math.round(r.width),
                                    height: Math.round(r.height),
                                    zIndex: style.zIndex || null,
                                    computed: {{
                                        fontSize: style.fontSize,
                                        fontFamily: (style.fontFamily || '').split(',')[0] || null,
                                        color: style.color,
                                        backgroundColor: style.backgroundColor,
                                        display: style.display
                                    }}
                                }});
                            }} catch(e) {{ /* ignore element */ }}
                        }});
                    }});

                    // Forms summary
                    const forms = Array.from(document.querySelectorAll('form')).slice(0,40).map(f => {{
                        const inputs = Array.from(f.querySelectorAll('input,textarea,select')).map(i => ({{
                            tag: i.tagName, type: i.getAttribute('type') || null, name: i.getAttribute('name') || null, placeholder: i.getAttribute('placeholder') || null
                        }}));
                        return {{ action: f.getAttribute('action') || null, method: (f.getAttribute('method')||'').toLowerCase() || 'get', input_count: inputs.length }};
                    }});

                    // Images
                    const imgs = Array.from(document.querySelectorAll('img')).slice(0,200).map(img => {{
                        return {{ src: img.currentSrc || img.src || null, alt: img.getAttribute('alt') || null, width: img.naturalWidth || null, height: img.naturalHeight || null }};
                    }});
                    const imagesWithoutAlt = imgs.filter(i => !i.alt).length;

                    // layout & performance stats
                    const totalElements = document.querySelectorAll('*').length;
                    const flexContainers = document.querySelectorAll('[style*=\"display: flex\"], [class*=\"flex\"]').length;
                    const gridContainers = document.querySelectorAll('[style*=\"display: grid\"], [class*=\"grid\"]').length;
                    const fixedElements = Array.from(document.querySelectorAll('*')).filter(el => {{
                        try {{ return window.getComputedStyle(el).position === 'fixed'; }} catch(e) {{ return false; }}
                    }}).length;

                    // accessibility quick checks
                    const links = Array.from(document.querySelectorAll('a'));
                    const linksWithoutText = links.filter(a => !(a.textContent && a.textContent.trim()) && !a.getAttribute('aria-label')).length;

                    // text density
                    const bodyText = document.body.innerText || '';
                    const wordCount = bodyText.split(/\\s+/).filter(Boolean).length;

                    // focusable order (tabindex, limited)
                    const focusables = Array.from(document.querySelectorAll('a[href], button, input, textarea, select, [tabindex]')).slice(0,200).map(el => {{
                        return {{ tag: el.tagName, id: el.id || null, classes: Array.from(el.classList).slice(0,3), text: compactText(el.textContent,40) }};
                    }});

                    // small dom structure hash
                    const important = Array.from(document.querySelectorAll('main, header, nav, footer, section, article, h1, h2, h3')).slice(0,80);
                    const domStruct = important.map(e => e.tagName + (e.id||'') + Array.from(e.classList).slice(0,3).join('')).join('');
                    const domHash = btoa(domStruct).slice(0,32);

                    return {{
                        timestamp: new Date().toISOString(),
                        url: location.href,
                        title: document.title,
                        viewport: viewport,
                        headings: headings,
                        interactive_count: interactive.length,
                        interactive: interactive.slice(0, {max_interactive}),
                        forms: forms,
                        images_count: imgs.length,
                        imagesWithoutAlt: imagesWithoutAlt,
                        layout: {{ totalElements: totalElements, flexContainers: flexContainers, gridContainers: gridContainers, fixedElements: fixedElements }},
                        accessibility: {{ linksWithoutText: linksWithoutText }},
                        textDensity: {{ totalTextLength: bodyText.length, wordCount: wordCount }},
                        focusable: focusables,
                        domStructureHash: domHash
                    }};
                }} catch(e) {{
                    return {{ error: 'dom extraction failed: ' + (e && e.message), timestamp: new Date().toISOString(), domStructureHash: 'error' }};
                }}
            }}
        """)
        return dom
    except Exception as e:
        print(f"extract_relevant_dom error: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat(), "domStructureHash": "error"}


# ------- FLOW GRAPH STORAGE -------
class FlowTracker:
    def __init__(self):
        self.nodes = {}  # url -> info
        self.edges = []  # {from,to,action,capture_id,ts}
        self.last_url = None

    def add_capture(self, from_url, to_url, action, capture_id, title=None):
        ts = datetime.now().isoformat()
        if to_url not in self.nodes:
            self.nodes[to_url] = {'first_seen': ts, 'title': title}
        if from_url and from_url not in self.nodes:
            self.nodes[from_url] = {'first_seen': ts}
        if from_url:
            self.edges.append({'from': from_url, 'to': to_url, 'action': action, 'capture_id': capture_id, 'ts': ts})
        self.last_url = to_url

    def save(self, path):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'nodes': self.nodes, 'edges': self.edges}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"flow save error: {e}")


# ------- CAPTURE WRAPPER -------
async def capture_with_analysis(page, label, change_detector: UIChangeDetector, flow: FlowTracker, manual=False):
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    capture_id = f"{ts}_{change_detector.capture_count:03d}_{label}"
    try:
        if label == 'auto':
            await change_detector.wait_for_stability(page)

        # Extract DOM info BEFORE hiding fixed elements (so computed styles/positions match rendered state)
        dom_info = await extract_relevant_dom(page)

        dom_hash = dom_info.get('domStructureHash') if isinstance(dom_info, dict) else await get_dom_hash(page)

        # quick check: still take a temp fullpage to compare visuals if needed
        temp_base = os.path.join(SESSION_DIR, f'temp_{capture_id}')
        temp_img = await capture_fullpage_with_fallback(page, temp_base)

        if not temp_img or not os.path.exists(temp_img):
            print('[Error] temp image not created')
            return None

        should, reason = change_detector.should_capture(dom_hash, temp_img)
        if (not should) and (not manual) and label not in ('initial','final','navigation','forced'):
            # remove temp image
            try: os.remove(temp_img)
            except: pass
            print(f'[Skipped] {reason}')
            return None

        # keep final
        final_img = os.path.join(SESSION_DIR, f'{capture_id}.png')
        os.replace(temp_img, final_img)

        # gather visual metrics
        palette = extract_color_palette(final_img)
        whitespace = compute_whitespace_ratio(final_img)

        page_title = await page.title()
        ua = await page.evaluate('navigator.userAgent')
        ui_events = await pull_ui_events(page)

        metadata = {
            'session_id': SESSION_ID,
            'capture_id': capture_id,
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'manual': bool(manual),
            'reason': reason,
            'sequence': change_detector.capture_count,
            'page': {'url': page.url, 'title': page_title, 'user_agent': ua},
            'dom_hash': dom_hash,
            'dom': dom_info,  # <-- ML-friendly DOM included here
            'visual_metrics': {'palette': palette, 'whitespace_ratio': whitespace},
            'ui_events': ui_events
        }

        # save metadata
        with open(os.path.join(SESSION_DIR, f'{capture_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # update detector & flow
        prev_url = change_detector.previous_dom_hash_url if hasattr(change_detector, 'previous_dom_hash_url') else None
        change_detector.previous_dom_hash_url = page.url
        change_detector.previous_screenshot = final_img
        change_detector.previous_dom_hash = dom_hash
        change_detector.last_capture_time = time.time()
        change_detector.capture_count += 1

        # flow update: connect last recorded URL to this one
        flow.add_capture(flow.last_url, page.url, label, capture_id, title=page_title)
        flow.save(os.path.join(SESSION_DIR, 'flow_graph.json'))

        print(f"[Captured {change_detector.capture_count}] {label} -> {page.url}")
        print(f"  Reason: {reason}")
        return capture_id
    except Exception as e:
        print(f"capture error: {e}")
        return None


# ------- MONITOR & MANUAL LOOP -------
async def monitor_page_changes(page, change_detector, flow):
    while True:
        try:
            await capture_with_analysis(page, 'auto', change_detector, flow, manual=False)
            await asyncio.sleep(max(1.5, change_detector.min_change_interval))
        except Exception as e:
            print(f"monitor error: {e}")
            await asyncio.sleep(2)

async def manual_input_loop(trigger_q):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(1) as pool:
        while True:
            line = await loop.run_in_executor(pool, input, '')
            if not line:
                continue
            parts = line.strip().split(maxsplit=1)
            cmd = parts[0].lower()
            label = parts[1] if len(parts)>1 else 'manual'
            if cmd in ('m','capture'):
                await trigger_q.put({'type':'manual','label':label})
            elif cmd in ('exit','quit'):
                await trigger_q.put({'type':'exit'})
                return
            else:
                print("Unknown command. Use 'capture <label>' or 'm <label>' to capture, 'exit' to stop.")


# ------- MAIN -------
async def setup_page(page):
    # reduce noise
    await page.set_extra_http_headers({'Accept-Language':'en-US,en;q=0.9'})
    await page.route('**/*', lambda route: (route.abort() if any(x in route.request.url for x in ['googletagmanager','doubleclick','googlesyndication','adsystem','analytics']) else route.continue_()))
    await inject_event_listeners(page)

async def auto_explore_page(page, detector, flow, delay_range=(4, 8)):
    """
    Automatically explores the page:
    - Scrolls down gradually
    - Clicks visible buttons/links at random
    - Waits for DOM stability and triggers captures
    """
    try:
        # scroll in steps
        total_height = await page.evaluate("document.body.scrollHeight")
        step = int(total_height / 5)
        for y in range(0, total_height, step):
            await page.evaluate(f"window.scrollTo(0, {y});")
            await asyncio.sleep(random.uniform(1.0, 2.0))

        # click random visible elements
        candidates = await page.query_selector_all("a, button, [role='button']")
        visible = []
        for el in candidates:
            box = await el.bounding_box()
            if box and box['width'] > 40 and box['height'] > 15:
                visible.append(el)

        random.shuffle(visible)
        max_clicks = min(4, len(visible))
        print(f"🧭 Auto explore: found {len(visible)} clickable elements, trying {max_clicks}")

        for i in range(max_clicks):
            el = visible[i]
            try:
                desc = await el.evaluate("el => el.innerText || el.getAttribute('aria-label') || el.tagName")
                print(f"👉 Clicking: {desc[:80]}")
                await el.click(timeout=5000)
                await detector.wait_for_stability(page)
                await asyncio.sleep(random.uniform(2.0, 3.5))
                await capture_with_analysis(page, f"auto_click_{i}", detector, flow, manual=False)
            except Exception as e:
                print(f"⚠️ Auto click failed: {e}")

        # random delay between exploration cycles
        await asyncio.sleep(random.uniform(*delay_range))
    except Exception as e:
        print(f"auto_explore_page error: {e}")

async def main():
    detector = UIChangeDetector()
    flow = FlowTracker()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=60)
        context = await browser.new_context(viewport={'width':1366,'height':900}, device_scale_factor=1)
        page = await context.new_page()

        await setup_page(page)

        print('Open site to monitor. Type capture <label> to force capture or exit to stop.')
        await page.goto('https://jsfiddle.net', wait_until='networkidle', timeout=60000)

        # initial capture
        await asyncio.sleep(1.5)
        await capture_with_analysis(page, 'initial', detector, flow, manual=True)

        # start monitor + manual input
        trigger_q = asyncio.Queue()
        manual_task = asyncio.create_task(manual_input_loop(trigger_q))
        monitor_task = asyncio.create_task(monitor_page_changes(page, detector, flow))

        # ✅ Auto exploration loop
        try:
            while True:
                # perform one round of automated exploration
                await auto_explore_page(page, detector, flow)

                # check for manual commands
                if not trigger_q.empty():
                    trigger = await trigger_q.get()
                    if trigger['type'] == 'exit':
                        print("Exit requested")
                        break
                    if trigger['type'] == 'manual':
                        label = trigger.get('label', 'manual')
                        await capture_with_analysis(page, f'manual_{label}', detector, flow, manual=True)
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            monitor_task.cancel(); manual_task.cancel()
            await capture_with_analysis(page, 'final', detector, flow, manual=True)
            await browser.close()
            print('Session saved in', SESSION_DIR)



if __name__ == '__main__':
    # dependencies: playwright, opencv-python, scikit-image, numpy
    print('Ensure you have installed: playwright (and run `playwright install`), opencv-python, scikit-image, numpy')
    asyncio.run(main())
