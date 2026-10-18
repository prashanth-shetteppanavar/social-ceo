"""render.py v2 - multi-style short-video renderer (cloud-friendly: only ffmpeg + Python packages).
Usage: python pipeline/render.py script.json out_dir [--preview t1 t2 ...]
Script JSON: see samples/. Style options: festival, neon_tech, clean_news, warm_story, finance_green.
"""
import json, math, os, random, re, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from piper import PiperVoice

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
W, H, FPS = 1080, 1920, 30
def _font(name):
    for d in (os.path.join(ROOT, "assets"), ROOT, HERE):
        f = os.path.join(d, name)
        if os.path.exists(f): return f
    raise FileNotFoundError(name + " not found in assets/, repo root or pipeline/")
B = _font("Poppins-Bold.ttf"); R = _font("Poppins-Regular.ttf")

script_path, out_dir = sys.argv[1], sys.argv[2]
PREVIEW = [float(x) for x in sys.argv[4:]] if len(sys.argv) > 3 and sys.argv[3] == "--preview" else None
os.makedirs(out_dir, exist_ok=True)
DATA = json.load(open(script_path, encoding="utf-8")); SC = DATA["scenes"]
def _txt(v):
    if isinstance(v, (list, tuple)): return ", ".join(str(x) for x in v if x)
    return "" if v is None else str(v)
DATA["sources"] = _txt(DATA.get("sources"))

# ------------------------------------------------------------------ styles
FEST = [(110, 130.81, 164.81), (87.31, 110, 130.81), (130.81, 164.81, 196), (98, 123.47, 146.83)]
STYLES = {
 "festival": dict(top=(11, 27, 58), bot=(25, 37, 78), text=(255, 255, 255), sub=(204, 214, 235), acc=(255, 153, 51), acc2=(255, 214, 102),
    foot=(143, 160, 196), fx="bokeh", cap="pill", pillbg=(6, 12, 30), pilltxt=(255, 255, 255), cardfill=(255, 255, 255), trans="wipe",
    mascot=True, voice="en-us-ryan-high", music=dict(bpm=96, chords=FEST, lead="pluck", kick=False), stroke=(0, 0, 0)),
 "neon_tech": dict(top=(5, 8, 22), bot=(28, 6, 52), text=(240, 248, 255), sub=(150, 200, 230), acc=(0, 229, 255), acc2=(255, 64, 160),
    foot=(110, 140, 180), fx="grid", cap="outline", pillbg=(0, 0, 0), pilltxt=(255, 255, 255), cardfill=(255, 255, 255), trans="flash",
    mascot=False, voice="en-us-ryan-high", music=dict(bpm=118, chords=[(110, 130.81, 164.81), (98, 123.47, 146.83), (87.31, 110, 130.81), (98, 123.47, 146.83)], lead="synth", kick=True), stroke=(2, 4, 18)),
 "clean_news": dict(top=(246, 248, 252), bot=(220, 228, 242), text=(18, 22, 36), sub=(70, 80, 105), acc=(214, 40, 40), acc2=(20, 70, 160),
    foot=(100, 110, 135), fx="stripes", cap="band", pillbg=(255, 255, 255), pilltxt=(18, 22, 36), cardfill=(0, 0, 0), trans="bars",
    mascot=False, voice="en-us-lessac-medium", music=dict(bpm=104, chords=[(98, 123.47, 146.83), (110, 138.59, 164.81), (87.31, 110, 130.81), (98, 123.47, 146.83)], lead="pluck", kick=False), stroke=(255, 255, 255)),
 "warm_story": dict(top=(255, 244, 226), bot=(255, 220, 192), text=(58, 38, 30), sub=(120, 84, 70), acc=(232, 90, 60), acc2=(40, 140, 120),
    foot=(140, 100, 85), fx="blobs", cap="pill", pillbg=(255, 255, 255), pilltxt=(58, 38, 30), cardfill=(0, 0, 0), trans="iris",
    mascot=True, voice="en-us-ryan-high", music=dict(bpm=92, chords=[(130.81, 164.81, 196), (110, 130.81, 164.81), (87.31, 110, 130.81), (98, 123.47, 146.83)], lead="pluck", kick=False), stroke=(255, 255, 255)),
 "finance_green": dict(top=(6, 26, 22), bot=(10, 50, 40), text=(240, 255, 248), sub=(160, 210, 190), acc=(255, 196, 0), acc2=(60, 220, 140),
    foot=(110, 160, 140), fx="chartgrid", cap="outline", pillbg=(0, 0, 0), pilltxt=(255, 255, 255), cardfill=(255, 255, 255), trans="wipe",
    mascot=False, voice="en-us-ryan-high", music=dict(bpm=100, chords=[(98, 123.47, 146.83), (87.31, 110, 130.81), (110, 130.81, 164.81), (98, 123.47, 146.83)], lead="synth", kick=True), stroke=(2, 20, 16)),
}
STYLE = DATA.get("style") if DATA.get("style") in STYLES else "festival"
ST = STYLES[STYLE]; MASCOT = ST["mascot"]
FOOT = ("Source: " + DATA["sources"]) if DATA["sources"] else ""

_f = {}
def F(p, s):
    k = (p, int(s))
    if k not in _f: _f[k] = ImageFont.truetype(p, int(s))
    return _f[k]
cl = lambda x, a=0, b=1: max(a, min(b, x))
eo = lambda p: 1 - (1 - cl(p)) ** 3
def back(p):
    p = cl(p); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2
lerp = lambda a, b, p: a + (b - a) * p
bare = lambda w: re.sub(r"\W", "", w).lower()
mix = lambda c1, c2, k: tuple(int(c1[i] * (1 - k) + c2[i] * k) for i in range(3))

# ------------------------------------------------------------------ narration
VOICE = os.environ.get("PIPER_VOICE") or os.path.join(ROOT, "models", DATA.get("voice") or ST["voice"])
if not os.path.exists(VOICE + ".onnx"): VOICE = os.path.join(ROOT, "models", "en-us-ryan-high")
voice = PiperVoice.load(VOICE + ".onnx", config_path=VOICE + ".onnx.json"); sr = voice.config.sample_rate
LEAD, TAIL = 0.3, 0.6
clips, S0, DUR = [], [], []; t0 = 0.0
for k, sc in enumerate(SC):
    fn = os.path.join(out_dir, f"_n{k}.wav")
    with wave.open(fn, "wb") as w: voice.synthesize_wav(sc["spoken"], w)
    with wave.open(fn) as w: a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    dur = LEAD + len(a) / sr + TAIL + (1.4 if k == len(SC) - 1 else 0)
    clips.append((a, t0 + LEAD)); S0.append(t0); DUR.append(dur); t0 += dur
total = t0; nF = int(total * FPS) + 1
nar = np.zeros(int(total * sr) + sr, np.float32)
for a, ts in clips:
    i = int(ts * sr); nar[i:i + len(a)] = a.astype(np.float32) / 32768

WT = []
for k, sc in enumerate(SC):
    a, ts = clips[k]; words = sc.get("caption", sc["spoken"]).split(); wts = []
    for w_ in words:
        wt = max(1, len(bare(w_)))
        if w_.endswith(","): wt += 2
        if w_[-1] in ".?!": wt += 3
        wts.append(wt)
    hop = int(sr * 0.01); n = max(1, len(a) // hop)
    e = np.array([np.sqrt(np.mean((a[i * hop:(i + 1) * hop].astype(np.float32) / 32768) ** 2)) for i in range(n)])
    e = e + 0.2 * e.mean() + 1e-6; cum = np.cumsum(e); cum /= cum[-1]; tt = (np.arange(n) + 1) * 0.01
    ws = np.cumsum(wts) / np.sum(wts); ends = np.interp(ws, cum, tt); starts = np.concatenate([[0], ends[:-1]])
    WT.append([(words[i], ts + starts[i], ts + ends[i]) for i in range(len(words))])
def tw(k, key, default=1.0):
    for w_, s, e in WT[k]:
        if bare(w_) == bare(key): return s
    return S0[k] + default
CH = []
for k in range(len(SC)):
    ch, cur = [], []
    for i, (w_, s, e) in enumerate(WT[k]):
        cur.append(i)
        if len(cur) >= 4 or w_[-1] in ",.?!": ch.append(cur); cur = []
    if cur: ch.append(cur)
    CH.append(ch)
KEY = {bare(x) for s in SC for x in (s.get("keywords") or [])}
env = np.array([np.sqrt(np.mean(nar[int(i * sr / FPS):int(i * sr / FPS) + int(sr / 15)] ** 2) + 1e-9) for i in range(nF)])
ref = np.percentile(env[env > 0.01], 90) if (env > 0.01).any() else 0.1
mouth = np.clip(env / (ref * 0.9), 0, 1); sm = np.zeros_like(mouth)
for i in range(1, len(mouth)): sm[i] = 0.55 * mouth[i] + 0.45 * sm[i - 1]
mouth = sm

# ------------------------------------------------------------------ meaning -> visual
AUTO = [("globe", "world global globe international countries worldwide abroad"),
        ("coin", "rupee rupees money price prices cost crore lakh cash pay paid payment salary tax refund"),
        ("chart", "growth grow increase rise rising percent percentage data rate report sales market stock trend"),
        ("phone", "phone mobile app apps upi online smartphone whatsapp message"),
        ("shield", "safe safety secure security scam scams fraud protect pin password lock privacy"),
        ("calendar", "date deadline october november december january february march april may june july august september today tomorrow"),
        ("clock", "minute minutes seconds hour hours quick fast"),
        ("battery", "battery charge charging power"),
        ("warning", "warning alert danger risk banned ban illegal"),
        ("bulb", "idea tip tips trick learn smart hack")]
VIS = []   # per scene: (name, time, text) or None
for k, sc in enumerate(SC):
    v = None
    if sc.get("chips") and MASCOT: VIS.append(None); continue
    if sc.get("number"): VIS.append(None); continue
    name = sc.get("visual")
    if name and name in dict(AUTO): v = (name, tw(k, sc["visual_at"], 0.8) if sc.get("visual_at") else S0[k] + 0.9)
    elif not name:
        for w_, s, e in WT[k]:
            for nm, kws in AUTO:
                if bare(w_) in kws.split(): v = (nm, s); break
            if v: break
    VIS.append((v[0], v[1], sc.get("visual_text", "")) if v else None)

ACC = []; CHIPS = []; NUM = []
for k, sc in enumerate(SC):
    ACC.append(tw(k, sc["accent_word"], 1.0) if sc.get("accent") else None)
    CHIPS.append([(c["text"], tw(k, c["at"], 1.0 + 0.8 * j)) for j, c in enumerate(sc.get("chips", [])[:4])])
    n = sc.get("number"); NUM.append((n, tw(k, n["at"], 1.0) if n and n.get("at") else S0[k] + 1.0) if n else None)
ACC_T = [t for t in ACC if t is not None]; LAST = len(SC) - 1
PUNCH = [(t, 0.07) for t in ACC_T] + [(n[1], 0.05) for n in NUM if n]
SPARK = [(t, 540, 640, 26, 10 + i) for i, t in enumerate(ACC_T)]

# ------------------------------------------------------------------ B-roll (optional)
BROLL = {}
bj = os.path.join(out_dir, "broll", "broll.json")
if os.path.exists(bj) and not PREVIEW:
    try: BROLL = {int(k): v["file"] for k, v in json.load(open(bj)).items() if os.path.exists(v["file"])}
    except Exception as ex: print("broll index unreadable:", ex)
class Reader:
    def __init__(self, path, dur):
        self.p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1", "-i", path, "-t", str(dur + 1), "-an",
            "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE)
    def read(self):
        b = self.p.stdout.read(W * H * 3)
        return Image.frombytes("RGB", (W, H), b) if len(b) == W * H * 3 else None
    def close(self):
        try: self.p.kill()
        except Exception: pass
_readers = {}

# ------------------------------------------------------------------ drawing helpers
def TX(dr, xy, txt, path, size, fill, a=1.0, anchor="mm", stroke=0, sfill=(0, 0, 0)):
    if a <= 0.01: return
    kw = dict(stroke_width=stroke, stroke_fill=sfill[:3] + (int(255 * cl(a)),)) if stroke else {}
    dr.text(xy, txt, font=F(path, size), fill=fill[:3] + (int(255 * cl(a)),), anchor=anchor, **kw)
def card(dr, cx, cy, w, h, sc=1.0, a=1.0, outline=(255, 255, 255), fill_a=26):
    w *= sc; h *= sc; cf = ST["cardfill"]
    dr.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=28, fill=cf + (int(fill_a * a),), outline=outline + (int(150 * a),), width=3)
bg = Image.new("RGB", (W, H)); _px = ImageDraw.Draw(bg)
for y in range(H): _px.line([(0, y), (W, y)], fill=mix(ST["top"], ST["bot"], y / H))
_px.rectangle([0, 0, W, 14], fill=ST["acc"]); _px.rectangle([0, H - 14, W, H], fill=ST["acc"])
if FOOT: _px.text((W / 2, 1840), FOOT[:70], font=F(R, 30), fill=ST["foot"], anchor="mm")
random.seed(7)
SP_ = [(random.randint(30, 1050), random.randint(0, 1900), random.uniform(25, 70), random.uniform(0, 6), random.randint(3, 7)) for _ in range(22)]
BLOBS = [(random.randint(100, 980), random.randint(200, 1700), random.randint(150, 320), random.uniform(0, 6)) for _ in range(5)]

def bg_fx(fr, t):
    d = ImageDraw.Draw(fr, "RGBA"); A = ST["acc"]; A2 = ST["acc2"]
    if ST["fx"] == "bokeh":
        for x, y0, v, ph, r in SP_:
            y = (y0 - v * t) % 1850; b = 0.5 + 0.5 * math.sin(t * 2 + ph)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(int(255 * (0.35 + 0.4 * b)), int(190 * (0.35 + 0.4 * b)), int(90 * (0.3 + 0.3 * b)), 255))
    elif ST["fx"] == "grid":
        for i in range(26):
            y = (i * 80 + t * 70) % (H + 80) - 40; d.line([(0, y), (W, y)], fill=A + (int(14 + 28 * (y / H)),), width=2)
        for x in range(0, W + 1, 90): d.line([(x, 0), (x, H)], fill=A + (22,), width=2)
        for r, al in ((420, 16), (300, 22), (180, 30)): d.ellipse([900 - r, 260 - r, 900 + r, 260 + r], fill=A2 + (al,))
    elif ST["fx"] == "stripes":
        for i in range(-3, 9):
            x0 = (i * 260 + t * 45) % 1900 - 500; d.polygon([(x0, 0), (x0 + 110, 0), (x0 + 110 - 760, H), (x0 - 760, H)], fill=A2 + (12,))
        d.rectangle([0, 14, W, 22], fill=A2 + (255,))
    elif ST["fx"] == "blobs":
        for x, y, r, ph in BLOBS:
            cx = x + 40 * math.sin(t * 0.6 + ph); cy = y + 50 * math.cos(t * 0.5 + ph)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(A if int(ph) % 2 else A2) + (34,))
    elif ST["fx"] == "chartgrid":
        for y in range(120, H, 160): d.line([(0, y), (W, y)], fill=A2 + (22,), width=2)
        pts = [(x, 1480 + 90 * math.sin(x / 140 + t * 0.9) + 40 * math.sin(x / 55 - t * 1.4) - x * 0.12) for x in range(0, W + 1, 30)]
        d.line(pts, fill=A + (70,), width=6, joint="curve")

# ------------------------------------------------------------------ semantic visual library
def visual_img(name, size, t, text=""):
    S = 1000; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im, "RGBA")
    A = ST["acc"]; A2 = ST["acc2"]; TXT = ST["text"]; DARK = mix(ST["top"], (0, 0, 0), 0.45); LIGHT = (245, 247, 252)
    if name == "globe":
        d.ellipse([100, 100, 900, 900], fill=mix(A2, DARK, 0.55) + (255,), outline=A + (255,), width=22)
        rot = t * 0.9
        for k in range(6):
            w = abs(math.cos(rot + k * math.pi / 6)) * 400; d.ellipse([500 - w, 100, 500 + w, 900], outline=A + (190,), width=8)
        for lat in (-0.55, 0, 0.55):
            y = 500 + lat * 400; hf = math.sqrt(400 ** 2 - (lat * 400) ** 2); d.line([500 - hf, y, 500 + hf, y], fill=A + (190,), width=8)
        ang = t * 1.6; mx, my = 500 + math.cos(ang) * 470, 500 + math.sin(ang) * 190
        d.ellipse([mx - 28, my - 28, mx + 28, my + 28], fill=A2 + (255,), outline=LIGHT + (255,), width=6)
    elif name == "coin":
        sx = max(0.12, abs(math.cos(t * 2.4))); r = 390; w = r * sx
        d.ellipse([500 - w, 500 - r, 500 + w, 500 + r], fill=(255, 196, 0, 255), outline=(190, 130, 0, 255), width=18)
        if sx > 0.25: d.ellipse([500 - w * 0.8, 500 - r * 0.8, 500 + w * 0.8, 500 + r * 0.8], outline=(255, 236, 150, 255), width=10)
        if sx > 0.5: d.text((500, 505), "\u20b9", font=F(B, 460), fill=(150, 95, 0, 255), anchor="mm")
    elif name == "chart":
        d.line([140, 860, 140, 140], fill=TXT + (200,), width=12); d.line([140, 860, 880, 860], fill=TXT + (200,), width=12)
        cyc = t % 4.0; hs = [0.25, 0.4, 0.35, 0.62, 0.88]; pts = []
        for i, hv in enumerate(hs):
            g = eo((cyc - 0.12 * i) / 0.6); x = 215 + i * 133; top = 860 - hv * 640 * g
            d.rounded_rectangle([x, top, x + 100, 860], radius=16, fill=(A if i == 4 else A2) + (255,)); pts.append((x + 50, top - 45))
        d.line(pts, fill=TXT + (235,), width=14, joint="curve")
        if cyc > 0.9: ex, ey = pts[-1]; d.polygon([(ex + 10, ey - 36), (ex + 54, ey + 6), (ex - 18, ey + 14)], fill=TXT + (235,))
    elif name == "phone":
        d.rounded_rectangle([290, 60, 710, 940], radius=80, fill=(26, 30, 44, 255), outline=A + (255,), width=14)
        d.rounded_rectangle([320, 130, 680, 870], radius=44, fill=(12, 18, 32, 255)); d.rounded_rectangle([440, 84, 560, 108], radius=12, fill=(60, 66, 86, 255))
        p = 1 + 0.06 * math.sin(t * 4); cx, cy = 500, 420; s = 150 * p
        d.polygon([(cx, cy - s), (cx + s * 0.8, cy - s * 0.55), (cx + s * 0.7, cy + s * 0.2), (cx, cy + s), (cx - s * 0.7, cy + s * 0.2), (cx - s * 0.8, cy - s * 0.55)], fill=A2 + (255,))
        d.line([(cx - 50, cy), (cx - 12, cy + 40), (cx + 56, cy - 40)], fill=(255, 255, 255, 255), width=22, joint="curve")
        for j in range(3):
            wv = 220 - 40 * ((j + int(t * 2)) % 3); d.rounded_rectangle([380, 650 + j * 55, 380 + wv, 680 + j * 55], radius=14, fill=(70, 78, 104, 255))
    elif name == "shield":
        pts = [(500, 70), (880, 200), (850, 620), (500, 940), (150, 620), (120, 200)]
        d.polygon(pts, fill=mix(A2, DARK, 0.15) + (255,), outline=A + (255,), width=20)
        pr = eo((t - 0.2) / 0.7); p1, p2, p3 = (320, 500), (450, 640), (700, 340)
        if pr > 0:
            seg = [p1] + ([(lerp(p1[0], p2[0], cl(pr * 2)), lerp(p1[1], p2[1], cl(pr * 2)))]) + ([(lerp(p2[0], p3[0], cl(pr * 2 - 1)), lerp(p2[1], p3[1], cl(pr * 2 - 1)))] if pr > 0.5 else [])
            d.line(seg, fill=(255, 255, 255, 255), width=64, joint="curve")
    elif name == "calendar":
        parts = (text or "OCT 15").split(); mo = parts[0].upper(); dy = parts[1] if len(parts) > 1 else ""
        d.rounded_rectangle([110, 150, 890, 900], radius=80, fill=LIGHT + (255,)); d.rounded_rectangle([110, 150, 890, 380], radius=80, fill=A + (255,)); d.rectangle([110, 300, 890, 380], fill=A + (255,))
        for x in (300, 700): d.rounded_rectangle([x - 22, 90, x + 22, 220], radius=22, fill=(40, 44, 60, 255))
        d.text((500, 270), mo, font=F(B, 150), fill=(255, 255, 255, 255), anchor="mm"); d.text((500, 650), dy or "31", font=F(B, 360), fill=(30, 34, 50, 255), anchor="mm")
    elif name == "clock":
        d.ellipse([110, 110, 890, 890], fill=DARK + (255,), outline=A + (255,), width=30)
        for i in range(12):
            a_ = i * math.pi / 6; d.line([500 + math.sin(a_) * 330, 500 - math.cos(a_) * 330, 500 + math.sin(a_) * 380, 500 - math.cos(a_) * 380], fill=TXT + (255,), width=14)
        for ln, sp, wd, col in ((290, 2.0, 18, A2), (200, 0.17, 26, TXT)):
            a_ = t * sp; d.line([500, 500, 500 + math.sin(a_) * ln, 500 - math.cos(a_) * ln], fill=col + (255,), width=wd)
        d.ellipse([470, 470, 530, 530], fill=A + (255,))
    elif name == "battery":
        lvl = 0.15 + 0.85 * eo((t % 3.2) / 2.4); col = (230, 60, 60) if lvl < 0.3 else A2
        d.rounded_rectangle([120, 300, 840, 720], radius=60, outline=TXT + (255,), width=26); d.rounded_rectangle([850, 440, 920, 580], radius=20, fill=TXT + (255,))
        d.rounded_rectangle([160, 340, 160 + 640 * lvl, 680], radius=34, fill=col + (255,))
        d.polygon([(520, 330), (420, 540), (500, 540), (470, 690), (600, 470), (520, 470)], fill=(255, 255, 255, 235))
    elif name == "warning":
        p = 1 + 0.05 * math.sin(t * 5); cx = 500
        pts = [(cx, 90), (cx + 440 * p, 860), (cx - 440 * p, 860)]
        d.polygon(pts, fill=(255, 196, 0, 255), outline=(40, 30, 10, 255), width=14)
        d.rounded_rectangle([cx - 30, 330, cx + 30, 640], radius=26, fill=(40, 30, 10, 255)); d.ellipse([cx - 40, 690, cx + 40, 770], fill=(40, 30, 10, 255))
    elif name == "bulb":
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        for r, al in ((430, 20), (360, 30), (300, 40)): d.ellipse([500 - r, 420 - r, 500 + r, 420 + r], fill=A2 + (int(al * (0.6 + 0.6 * pulse)),))
        d.ellipse([240, 140, 760, 660], fill=(255, 226, 120, 255), outline=(200, 150, 20, 255), width=14)
        d.rounded_rectangle([360, 650, 640, 760], radius=30, fill=(140, 150, 170, 255)); d.rounded_rectangle([400, 770, 600, 840], radius=30, fill=(110, 120, 140, 255))
        d.line([(400, 520), (450, 380), (500, 480), (550, 380), (600, 520)], fill=(200, 140, 20, 255), width=16, joint="curve")
        for i in range(7):
            a_ = -math.pi / 2 + (i - 3) * 0.42 + math.sin(t) * 0.03; d.line([500 + math.cos(a_) * 430, 420 + math.sin(a_) * 430, 500 + math.cos(a_) * 490, 420 + math.sin(a_) * 490], fill=A2 + (255,), width=18)
    return im.resize((size, size), Image.LANCZOS)

# ------------------------------------------------------------------ mascot
def flame(dr, cx, base, Hh, Wd, t, col, sc=2):
    L = []; Rr = []
    for k in range(21):
        s = k / 20; w = Wd * math.sin(math.pi * min(1, s ** 0.7)) * (1 - 0.35 * s)
        sway = 10 * (s ** 2) * math.sin(t * 9 + s * 2) + 5 * s * math.sin(t * 15); y = base - s * Hh * (1 + 0.04 * math.sin(t * 11))
        L.append(((cx + sway - w) * sc, y * sc)); Rr.append(((cx + sway + w) * sc, y * sc))
    dr.polygon(L + Rr[::-1], fill=col)
def character(t, m, exc):
    sc = 2; C = Image.new("RGBA", (600 * sc, 650 * sc), (0, 0, 0, 0)); dr = ImageDraw.Draw(C, "RGBA")
    E = lambda x0, y0, x1, y1, f: dr.ellipse([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=f)
    dy = 7 * math.sin(t * 3.2); wave_ = math.sin(t * 7) * (10 + 30 * m + 25 * exc); E(100, 600, 500, 640, (0, 0, 0, 70))
    for r, a in ((150, 18), (110, 28), (75, 40)): E(300 - r, 150 + dy - r, 300 + r, 150 + dy + r, (255, 190, 90, a))
    TERRA = (214, 104, 48); LIGHT = (236, 140, 70)
    for sx in (-1, 1):
        x0 = 300 + sx * 205; y0 = 430 + dy; x1 = 300 + sx * 272
        y1 = (380 if sx < 0 else 365) + dy - (wave_ if sx > 0 else wave_ * (0.3 + 0.7 * exc)) - exc * 30
        dr.line([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=TERRA, width=34 * sc); E(x0 - 17, y0 - 17, x0 + 17, y0 + 17, TERRA); E(x1 - 26, y1 - 26, x1 + 26, y1 + 26, LIGHT)
    E(70, 200 + dy, 530, 590 + dy, TERRA); E(60, 180 + dy, 540, 290 + dy, LIGHT); E(95, 195 + dy, 505, 272 + dy, (92, 44, 20))
    fl = 1 + 0.25 * exc; flame(dr, 300, 245 + dy, 175 * fl, 38, t, (255, 120, 20, 255)); flame(dr, 300, 242 + dy, 120 * fl, 22, t + 0.4, (255, 205, 70, 255))
    blink = 0.12 if (t % 3.4) > 3.27 else 1.0; lx = 3 * math.sin(t * 1.3)
    for ex in (228, 372):
        E(ex - 38, 390 + dy - 42 * blink, ex + 38, 390 + dy + 42 * blink, (255, 255, 255, 255)); E(ex - 19 + lx, 390 + dy - 19 * blink, ex + 19 + lx, 390 + dy + 19 * blink, (30, 20, 20, 255)); E(ex - 6 + lx, 380 + dy - 8 * blink, ex + 4 + lx, 388 + dy, (255, 255, 255, 255))
    E(150, 440 + dy, 200, 490 + dy, (255, 140, 140, 150)); E(400, 440 + dy, 450, 490 + dy, (255, 140, 140, 150))
    mh = 8 + m * 52; mw = 70 + m * 18; E(300 - mw / 2, 478 + dy - mh / 2 + mh * 0.1, 300 + mw / 2, 478 + dy + mh / 2 + mh * 0.1, (115, 20, 30, 255))
    if m > 0.4: E(300 - mw * 0.3, 478 + dy + mh * 0.05, 300 + mw * 0.3, 478 + dy + mh * 0.55, (240, 110, 120, 255))
    return C.resize((600, 650), Image.LANCZOS)

# ------------------------------------------------------------------ scene drawing
def sc_index(t):
    k = 0
    for j in range(len(S0)):
        if S0[j] <= t: k = j
    return k
CAP_Y = 1040 if MASCOT else 1600
def draw_head(dr, k, lt):
    sc = SC[k]; l = lt - (0.0 if k == 0 else 0.2)
    TX(dr, (540, 200), sc["label"], B, 44, ST["acc"], eo(l / 0.3))
    size = sc.get("size", 76)
    for j, ln in enumerate(sc["main"]):
        p = eo((l - 0.1 - 0.12 * j) / 0.4); TX(dr, (540, 330 + 112 * j + (1 - p) * 50), ln, B, size, ST["text"], p, "mt")
def draw_caption(dr, k, t):
    words = WT[k]; fs = 60; sp = 24
    live = [ch for ch in CH[k] if words[ch[0]][1] - 0.08 <= t <= words[ch[-1]][2] + 0.18]
    if live: live = [max(live, key=lambda c: words[c[0]][1])]     # only the newest chunk, so captions never overlap
    for ch in live:
        s = words[ch[0]][1] - 0.08; e = words[ch[-1]][2] + 0.18
        p = eo((t - s) / 0.14); ws = [F(B, fs).getlength(words[i][0]) for i in ch]; tot = sum(ws) + sp * (len(ch) - 1)
        cy = CAP_Y + (1 - p) * 18; x0 = 540 - tot / 2
        if ST["cap"] == "pill": dr.rounded_rectangle([x0 - 34, cy - 52, x0 + tot + 34, cy + 52], radius=40, fill=ST["pillbg"] + (int(225 * p),))
        elif ST["cap"] == "band": dr.rectangle([0, cy - 66, W, cy + 66], fill=ST["acc"] + (int(238 * p),)); dr.rectangle([0, cy - 66, W, cy - 60], fill=ST["acc2"] + (int(255 * p),))
        x = x0
        for n_, i in enumerate(ch):
            w_, s_, e_ = words[i]; cx = x + ws[n_] / 2; act = s_ <= t < e_ + 0.02
            if ST["cap"] == "pill": base = ST["pilltxt"]; hi = ST["acc"]
            elif ST["cap"] == "band": base = (255, 255, 255); hi = (255, 240, 150)
            else: base = ST["text"]; hi = ST["acc"]
            if act: pp = cl((t - s_) / 0.12); fz = int(fs * (1.08 - 0.05 * pp)); col = hi
            else: fz = fs; col = (ST["acc2"] if (bare(w_) in KEY and ST["cap"] != "band") else base)
            TX(dr, (cx, cy), w_, B, fz, col, p, stroke=7 if ST["cap"] == "outline" else 0, sfill=ST["stroke"]); x += ws[n_] + sp
CHIP_POS = [(330, 860), (760, 860), (330, 940), (760, 940)] if MASCOT else [(330, 800), (760, 800), (330, 880), (760, 880)]
def draw_scene(ov, dr, k, t, lt):
    sc = SC[k]; draw_head(dr, k, lt)
    if sc.get("accent") and ACC[k] is not None and t >= ACC[k]:
        ty = ACC[k]; p = back((t - ty) / 0.45); ay = 330 + 112 * len(sc["main"]) + 14
        TX(dr, (540, ay), sc["accent"], B, int(sc.get("accent_size", 128) * max(p, 0.05)), ST["acc"], cl(p * 3), "mt")
        uw = eo((t - ty - 0.25) / 0.5) * 620
        if uw > 1: dr.rounded_rectangle([540 - uw / 2, ay + 158, 540 + uw / 2, ay + 168], radius=5, fill=ST["acc2"] + (255,))
    for i, (txt, te) in enumerate(CHIPS[k]):
        if t < te: continue
        cx, cy = CHIP_POS[i]; p = back((t - te) / 0.35); w = F(B, 40).getlength(txt) + 70
        card(dr, cx, cy, w, 66, p, cl(p * 2), ST["acc"], 30); TX(dr, (cx, cy), txt, B, 40 * cl(p, 0.6, 1.1), ST["text"], cl(p * 2))
    if NUM[k] and t >= NUM[k][1]:
        n, tn = NUM[k]; p = back((t - tn) / 0.4); val = float(n["value"]) * eo((t - tn) / 1.3)
        s = f"{int(round(val)):,}" if float(n["value"]) >= 100 else (f"{val:.1f}".rstrip("0").rstrip(".") if val % 1 else f"{int(val)}")
        txt = f"{n.get('prefix', '')}{s}{n.get('suffix', '')}"; cy = 820 if MASCOT else 1130
        TX(dr, (540, cy), txt, B, int(190 * max(0.2, min(p, 1.05))) if len(txt) < 8 else int(130 * max(0.2, min(p, 1.05))), ST["acc2"], cl(p * 3), stroke=0)
        if n.get("label"): TX(dr, (540, cy + 130), n["label"], R, 46, ST["sub"], cl((t - tn) / 0.4))
    v = VIS[k]
    if v and t >= v[1]:
        name, tv, txt = v; p = back((t - tv) / 0.5); size = 300 if MASCOT else 540; cy = 860 if MASCOT else 1180
        im = visual_img(name, size, t - tv, txt); s = int(size * cl(p, 0.05, 1.08)); im = im.resize((s, s), Image.BILINEAR)
        if p < 1: al = im.getchannel("A").point(lambda q: int(q * cl(p * 2))); im.putalpha(al)
        ov.alpha_composite(im, (int(540 - s / 2), int(cy - s / 2)))

# ------------------------------------------------------------------ transitions
def draw_trans(fr, t):
    kind = ST["trans"]
    for s in S0[1:]:
        if not (s - 0.27 <= t <= s + 0.27): continue
        if kind == "wipe":
            d = ImageDraw.Draw(fr); SK = 110
            if t < s - 0.05: L = lerp(W + SK, -SK, eo((t - (s - 0.25)) / 0.2)); Rr = W + 2 * SK
            elif t <= s + 0.05: L = -SK; Rr = W + 2 * SK
            else: L = -SK; Rr = lerp(W + 2 * SK, -SK, eo((t - (s + 0.05)) / 0.2))
            if Rr > L + 2:
                for off, col in ((70, mix(ST["acc"], (0, 0, 0), 0.25)), (0, ST["acc"])):
                    d.polygon([(L + SK + off, 0), (Rr + SK + off, 0), (Rr - SK + off, H), (L - SK + off, H)], fill=col)
        elif kind == "bars":
            d = ImageDraw.Draw(fr); n = 6; bw = W / n + 1
            for i in range(n):
                col = ST["acc"] if i % 2 == 0 else ST["acc2"]
                if t < s - 0.05: f = eo((t - (s - 0.26) - 0.02 * i) / 0.16); top, bot = 0, H * f
                elif t <= s + 0.05: top, bot = 0, H
                else: r = eo((t - (s + 0.05) - 0.02 * i) / 0.16); top, bot = H * r, H
                if bot > top + 1: d.rectangle([i * bw, top, (i + 1) * bw, bot], fill=col)
        elif kind == "iris":
            m = Image.new("L", (W, H), 0); md = ImageDraw.Draw(m); cx, cy = 540, 960
            if t < s - 0.05: r = eo((t - (s - 0.25)) / 0.2) * 1500; md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            elif t <= s + 0.05: md.rectangle([0, 0, W, H], fill=255)
            else: r = eo((t - (s + 0.05)) / 0.2) * 1500; md.rectangle([0, 0, W, H], fill=255); md.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
            fr.paste(ST["acc"], mask=m)
        elif kind == "flash":
            a = cl((t - (s - 0.12)) / 0.12) if t < s else cl(1 - (t - s) / 0.22)
            if a > 0: fr.paste((255, 255, 255), mask=Image.new("L", (W, H), int(255 * a * 0.95)))
def render(t):
    k = sc_index(t); lt = t - S0[k]; fr = None
    if k in BROLL:
        if k not in _readers: _readers[k] = Reader(BROLL[k], DUR[k]); [_readers.pop(j).close() for j in list(_readers) if j != k]
        fr = _readers[k].read()
        if fr is not None: fr = Image.blend(fr, Image.new("RGB", (W, H), mix(ST["top"], (0, 0, 0), 0.5)), 0.58)
    if fr is None: fr = bg.copy(); bg_fx(fr, t)
    d0 = ImageDraw.Draw(fr); d0.rectangle([0, 14, int(W * t / total), 22], fill=ST["acc2"])
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov, "RGBA")
    draw_scene(ov, dr, k, t, lt); draw_caption(dr, k, t); fr.paste(ov, (0, 0), ov)
    if MASCOT:
        exc = 0.0
        for te in ACC_T:
            if te <= t < te + 2.0: exc = max(exc, 1 - (t - te) / 2.0)
        pulse = 1.0
        for te, amp in PUNCH:
            dt = t - te
            if 0 <= dt < 0.35: pulse = max(pulse, 1 + amp * (1 - dt / 0.35) ** 2)
        ch = character(t, float(mouth[min(int(t * FPS), len(mouth) - 1)]), exc); s = pulse * (back(t / 0.8) if t < 1.0 else 1.0); cw = int(600 * s); chh = int(650 * s)
        if cw > 4: ch = ch.resize((cw, chh), Image.BILINEAR); fr.paste(ch, (540 - cw // 2, 1800 - chh), ch)
    dd = ImageDraw.Draw(fr)
    for ts_, x, y, n, sd in SPARK:
        dt = t - ts_
        if 0 <= dt < 1.0:
            rg = random.Random(sd)
            for _ in range(n):
                ang = rg.uniform(0, 6.283); spd = rg.uniform(250, 720); px = x + math.cos(ang) * spd * dt; py = y + math.sin(ang) * spd * dt + 300 * dt * dt; f = 1 - dt; r = 2 + 5 * f
                dd.ellipse([px - r, py - r, px + r, py + r], fill=mix((0, 0, 0), ST["acc2"], f))
    draw_trans(fr, t)
    dirn = 1 if k % 2 == 0 else -1; z = (1 + 0.03 * cl(lt / DUR[k])) if dirn > 0 else (1.04 - 0.03 * cl(lt / DUR[k]))
    for te, amp in PUNCH:
        dt = t - te
        if 0 <= dt < 0.35: z += amp * (1 - dt / 0.35) ** 2
    if z > 1.002:
        cw_, ch_ = W / z, H / z; top = max(0, (H - ch_) / 2 - 20 * (z - 1))
        fr = fr.crop((int((W - cw_) / 2), int(top), int((W + cw_) / 2), int(top + ch_))).resize((W, H), Image.BILINEAR)
    return fr

if PREVIEW:
    ims = [render(t).resize((270, 480)) for t in PREVIEW]
    sh = Image.new("RGB", (270 * 4, 480 * ((len(ims) + 3) // 4)))
    for i, im in enumerate(ims): sh.paste(im, ((i % 4) * 270, (i // 4) * 480))
    sh.save(os.path.join(out_dir, "preview.png")); print("style", STYLE, "scene starts", [round(x, 1) for x in S0], "total", round(total, 1)); sys.exit()

# ------------------------------------------------------------------ audio
SR = 44100; rng = np.random.default_rng(3)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-ar", str(SR), os.path.join(out_dir, "_n44.wav")], input=nar.tobytes(), check=True)
with wave.open(os.path.join(out_dir, "_n44.wav")) as w: n44 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
N = int(total * SR); mix_ = np.zeros(N, np.float32); mix_[:min(N, len(n44))] += n44[:N]
def lp(x, a0, a1):
    y = np.zeros_like(x); s = 0.0; n = len(x)
    for i in range(n): s += (a0 + (a1 - a0) * i / n) * (x[i] - s); y[i] = s
    return y
def place(sfx, t, g=1.0):
    i = int(max(0, t) * SR); e = min(N, i + len(sfx))
    if e > i: mix_[i:e] += sfx[:e - i] * g
def whoosh(d=0.5, hi=0.4):
    n = int(d * SR); y = lp(rng.standard_normal(n).astype(np.float32), 0.01, hi) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2; return y / np.abs(y).max() * 0.30
def pop(f0=640, d=0.16):
    n = int(d * SR); t = np.arange(n) / SR; f = f0 * (1 + 1.1 * np.exp(-t / 0.02)); return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05) * 0.30).astype(np.float32)
def impact(d=0.8):
    n = int(d * SR); t = np.arange(n) / SR; f = 42 + 90 * np.exp(-t / 0.06); lo = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    nz = lp(rng.standard_normal(n).astype(np.float32), 0.2, 0.05) * np.exp(-t / 0.05); return (lo * 0.7 + nz * 0.5).astype(np.float32) * 0.8
def riser(d=1.2):
    n = int(d * SR); t = np.arange(n) / SR; nz = lp(rng.standard_normal(n).astype(np.float32), 0.005, 0.5) * (t / d) ** 2
    f = 180 + 950 * (t / d) ** 2; s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) ** 2 * 0.5; y = nz * 3 + s; return (y / np.abs(y).max() * 0.28).astype(np.float32)
def chime(d=1.6):
    n = int(d * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in ((1046, 1, 0.9), (1568, 0.6, 0.7), (2093, 0.4, 0.5), (1318, 0.3, 0.6))); return (y / np.abs(y).max() * 0.22).astype(np.float32)
for s in S0[1:]:
    place(whoosh(0.5, 0.55 if ST["trans"] == "flash" else 0.4), s - 0.27)
    if ST["trans"] == "flash": place(impact(0.4), s - 0.02, 0.35)
place(pop(520), 0.35); place(chime(), 0.1, 0.6)
for k in range(len(SC)):
    for i, (txt, te) in enumerate(CHIPS[k]): place(pop(560 + 40 * i), te)
    if ACC[k] is not None: place(riser(1.0), ACC[k] - 1.0); place(impact(), ACC[k])
    if NUM[k]: place(pop(700), NUM[k][1]); place(impact(0.5), NUM[k][1], 0.4)
    if VIS[k]: place(pop(480), VIS[k][1]); place(whoosh(0.3), VIS[k][1] - 0.1, 0.6)
place(chime(), S0[LAST] + 1.0, 0.8)
MU = ST["music"]; beat = 60 / MU["bpm"]; t_ = np.arange(N) / SR; mus = np.zeros(N, np.float32); bar = beat * 8; chords = MU["chords"]
for i in range(int(total / bar) + 2):
    ch = chords[i % 4]; s0 = int(i * bar * SR)
    if s0 >= N: break
    n = min(int(bar * SR), N - s0); tt = np.arange(n) / SR; ev = np.minimum(1, tt / 0.6) * np.minimum(1, (bar - tt) / 0.8)
    for f in ch: mus[s0:s0 + n] += np.sin(2 * np.pi * f * tt) * 0.05 * ev + np.sin(2 * np.pi * 2 * f * tt) * 0.015 * ev
    pat = [0, 1, 2, 1, 2, 1, 0, 2]
    for j in range(16):
        ts = int((i * bar + j * beat / 2) * SR)
        if ts >= N: break
        f = ch[pat[j % 8]] * 4; m = min(int(0.4 * SR), N - ts); tq = np.arange(m) / SR
        if MU["lead"] == "synth": wv = sum(np.sin(2 * np.pi * f * h * tq) / h for h in (1, 2, 3, 4)) * 0.5; dec = 0.16
        else: wv = np.sin(2 * np.pi * f * tq) + 0.3 * np.sin(4 * np.pi * f * tq); dec = 0.22
        mus[ts:ts + m] += wv * np.exp(-tq / dec) * np.minimum(1, tq / 0.005) * 0.05
        if j % 2 == 1:
            h = min(int(0.03 * SR), N - ts); mus[ts:ts + h] += np.diff(rng.standard_normal(h + 1)).astype(np.float32) * np.exp(-np.arange(h) / SR / 0.012) * 0.04
        if MU["kick"] and j % 2 == 0:
            kn = min(int(0.18 * SR), N - ts); kt = np.arange(kn) / SR; mus[ts:ts + kn] += np.sin(2 * np.pi * np.cumsum(50 + 80 * np.exp(-kt / 0.03)) / SR) * np.exp(-kt / 0.08) * 0.10
nn = np.zeros(N, np.float32); nn[:min(N, len(n44))] = n44[:N]
w_ = int(0.05 * SR); rms = np.sqrt(np.convolve(nn ** 2, np.ones(w_) / w_, mode="same")); act = (rms > 0.02).astype(np.float32); kk = int(0.3 * SR); act = np.convolve(act, np.ones(kk) / kk, mode="same")
duck = 1 - 0.65 * np.clip(act * 1.5, 0, 1); fade = np.minimum(1, t_ / 0.8) * np.minimum(1, (total - t_) / 1.6)
mix_ += mus * duck * fade; mix_ = np.tanh(mix_ * 1.2) / 1.2
mixpath = os.path.join(out_dir, "_mix.wav")
with wave.open(mixpath, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix_, -1, 1) * 32767).astype(np.int16).tobytes())
OUT = os.path.join(out_dir, "video.mp4")
enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", mixpath,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.5", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", OUT], stdin=subprocess.PIPE)
for i in range(nF):
    enc.stdin.write(render(i / FPS).tobytes())
    if i % 150 == 0: print(i, nF, flush=True)
enc.stdin.close(); enc.wait()
for r_ in _readers.values(): r_.close()
meta = {k: DATA.get(k) for k in ("title", "description", "tags", "needs_review", "topic", "sources", "style")}
meta["duration_seconds"] = round(total, 1); meta["style"] = STYLE
if os.path.exists(bj): meta["broll"] = json.load(open(bj))
json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for f in os.listdir(out_dir):
    if f.startswith("_"): os.remove(os.path.join(out_dir, f))
print("DONE", OUT)
