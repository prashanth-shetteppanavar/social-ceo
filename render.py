"""Render a 9:16 explainer video (Deepu the Diya narrator) from a script JSON.
Usage: python pipeline/render.py script.json out_dir [--preview t1 t2 ...]
Needs: ffmpeg on PATH, pip install -r requirements.txt, a Piper voice in models/.
"""
import json, math, os, random, re, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from piper import PiperVoice

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W, H, FPS = 1080, 1920, 30
B = os.path.join(ROOT, "assets", "Poppins-Bold.ttf")
R = os.path.join(ROOT, "assets", "Poppins-Regular.ttf")
VOICE = os.environ.get("PIPER_VOICE", os.path.join(ROOT, "models", "en-us-ryan-high"))
NAVY = (11, 27, 58); SAFF = (255, 153, 51); GOLD = (255, 214, 102); WHITE = (255, 255, 255)

script_path, out_dir = sys.argv[1], sys.argv[2]
PREVIEW = [float(x) for x in sys.argv[4:]] if len(sys.argv) > 3 and sys.argv[3] == "--preview" else None
os.makedirs(out_dir, exist_ok=True)
DATA = json.load(open(script_path, encoding="utf-8"))
SC = DATA["scenes"]
FOOT = "Source: " + DATA.get("sources", "") if DATA.get("sources") else ""

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

# ---------- narration ----------
voice = PiperVoice.load(VOICE + ".onnx", config_path=VOICE + ".onnx.json")
sr = voice.config.sample_rate
LEAD, TAIL = 0.3, 0.6
clips, S0, DUR = [], [], []
t0 = 0.0
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

# word timing from speech energy
WT = []
for k, sc in enumerate(SC):
    a, ts = clips[k]
    words = sc.get("caption", sc["spoken"]).split()
    wts = []
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

# ---------- events ----------
ACC = []   # (scene, time)
CHIPS = [] # per scene list of (text, time)
for k, sc in enumerate(SC):
    ACC.append(tw(k, sc["accent_word"], 1.0) if sc.get("accent") else None)
    ch = []
    items = sc.get("chips", [])[:4]
    for j, c in enumerate(items):
        ch.append((c["text"], tw(k, c["at"], 1.0 + 0.8 * j)))
    CHIPS.append(ch)
ACC_T = [t for t in ACC if t is not None]
PUNCH = [(t, 0.07) for t in ACC_T]
SPARK = [(t, 540, 640, 26, 10 + i) for i, t in enumerate(ACC_T)]
LAST = len(SC) - 1

# ---------- drawing helpers ----------
def TX(dr, xy, txt, path, size, fill, a=1.0, anchor="mm"):
    if a <= 0.01: return
    dr.text(xy, txt, font=F(path, size), fill=fill[:3] + (int(255 * cl(a)),), anchor=anchor)
def card(dr, cx, cy, w, h, sc=1.0, a=1.0, outline=(255, 255, 255), fill_a=26):
    w *= sc; h *= sc
    dr.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=28,
                         fill=(255, 255, 255, int(fill_a * a)), outline=outline + (int(110 * a),), width=3)
bg = Image.new("RGB", (W, H), NAVY); _d = ImageDraw.Draw(bg)
_d.rectangle([0, 0, W, 14], fill=SAFF); _d.rectangle([0, H - 14, W, H], fill=SAFF)
if FOOT: _d.text((W / 2, 1840), FOOT[:70], font=F(R, 30), fill=(143, 160, 196), anchor="mm")
random.seed(7)
SP_ = [(random.randint(30, 1050), random.randint(0, 1900), random.uniform(25, 70), random.uniform(0, 6), random.randint(3, 7)) for _ in range(22)]
SPK = []
for (te, x, y, n, sd) in SPARK:
    rg = random.Random(sd); SPK.append((te, x, y, [(rg.uniform(0, 6.283), rg.uniform(250, 720)) for _ in range(n)]))

def flame(dr, cx, base, Hh, Wd, t, col, sc=2):
    L = []; Rr = []
    for k in range(21):
        s = k / 20; w = Wd * math.sin(math.pi * min(1, s ** 0.7)) * (1 - 0.35 * s)
        sway = 10 * (s ** 2) * math.sin(t * 9 + s * 2) + 5 * s * math.sin(t * 15)
        y = base - s * Hh * (1 + 0.04 * math.sin(t * 11))
        L.append(((cx + sway - w) * sc, y * sc)); Rr.append(((cx + sway + w) * sc, y * sc))
    dr.polygon(L + Rr[::-1], fill=col)

def character(t, m, exc):
    sc = 2; C = Image.new("RGBA", (600 * sc, 650 * sc), (0, 0, 0, 0)); dr = ImageDraw.Draw(C, "RGBA")
    E = lambda x0, y0, x1, y1, f: dr.ellipse([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=f)
    dy = 7 * math.sin(t * 3.2); wave_ = math.sin(t * 7) * (10 + 30 * m + 25 * exc)
    E(100, 600, 500, 640, (0, 0, 0, 70))
    for r, a in ((150, 18), (110, 28), (75, 40)): E(300 - r, 150 + dy - r, 300 + r, 150 + dy + r, (255, 190, 90, a))
    TERRA = (214, 104, 48); LIGHT = (236, 140, 70)
    for sx in (-1, 1):
        x0 = 300 + sx * 205; y0 = 430 + dy; x1 = 300 + sx * 272
        y1 = (380 if sx < 0 else 365) + dy - (wave_ if sx > 0 else wave_ * (0.3 + 0.7 * exc)) - exc * 30
        dr.line([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=TERRA, width=34 * sc)
        E(x0 - 17, y0 - 17, x0 + 17, y0 + 17, TERRA); E(x1 - 26, y1 - 26, x1 + 26, y1 + 26, LIGHT)
    E(70, 200 + dy, 530, 590 + dy, TERRA); E(60, 180 + dy, 540, 290 + dy, LIGHT); E(95, 195 + dy, 505, 272 + dy, (92, 44, 20))
    fl = 1 + 0.25 * exc
    flame(dr, 300, 245 + dy, 175 * fl, 38, t, (255, 120, 20, 255)); flame(dr, 300, 242 + dy, 120 * fl, 22, t + 0.4, (255, 205, 70, 255))
    blink = 0.12 if (t % 3.4) > 3.27 else 1.0; lx = 3 * math.sin(t * 1.3)
    for ex in (228, 372):
        E(ex - 38, 390 + dy - 42 * blink, ex + 38, 390 + dy + 42 * blink, (255, 255, 255, 255))
        E(ex - 19 + lx, 390 + dy - 19 * blink, ex + 19 + lx, 390 + dy + 19 * blink, (30, 20, 20, 255))
        E(ex - 6 + lx, 380 + dy - 8 * blink, ex + 4 + lx, 388 + dy, (255, 255, 255, 255))
    E(150, 440 + dy, 200, 490 + dy, (255, 140, 140, 150)); E(400, 440 + dy, 450, 490 + dy, (255, 140, 140, 150))
    mh = 8 + m * 52; mw = 70 + m * 18
    E(300 - mw / 2, 478 + dy - mh / 2 + mh * 0.1, 300 + mw / 2, 478 + dy + mh / 2 + mh * 0.1, (115, 20, 30, 255))
    if m > 0.4: E(300 - mw * 0.3, 478 + dy + mh * 0.05, 300 + mw * 0.3, 478 + dy + mh * 0.55, (240, 110, 120, 255))
    return C.resize((600, 650), Image.LANCZOS)

def sc_index(t):
    k = 0
    for j in range(len(S0)):
        if S0[j] <= t: k = j
    return k

def draw_head(dr, k, lt):
    sc = SC[k]; l = lt - (0.0 if k == 0 else 0.2)
    TX(dr, (540, 200), sc["label"], B, 44, SAFF, eo(l / 0.3))
    size = sc.get("size", 76)
    for j, ln in enumerate(sc["main"]):
        p = eo((l - 0.1 - 0.12 * j) / 0.4); TX(dr, (540, 330 + 112 * j + (1 - p) * 50), ln, B, size, WHITE, p, "mt")

def draw_caption(dr, k, t):
    words = WT[k]
    for ch in CH[k]:
        s = words[ch[0]][1] - 0.08; e = words[ch[-1]][2] + 0.18
        if not (s <= t <= e): continue
        p = eo((t - s) / 0.14); fs = 60; sp = 24
        ws = [F(B, fs).getlength(words[i][0]) for i in ch]; tot = sum(ws) + sp * (len(ch) - 1)
        cy = 1040 + (1 - p) * 18; x0 = 540 - tot / 2
        dr.rounded_rectangle([x0 - 34, cy - 52, x0 + tot + 34, cy + 52], radius=40, fill=(6, 12, 30, int(215 * p)))
        x = x0
        for n_, i in enumerate(ch):
            w_, s_, e_ = words[i]; cx = x + ws[n_] / 2
            if s_ <= t < e_ + 0.02:
                pp = cl((t - s_) / 0.12); fz = int(fs * (1.06 - 0.04 * pp)); col = SAFF
            else:
                fz = fs; col = GOLD if bare(w_) in KEY else WHITE
            TX(dr, (cx, cy), w_, B, fz, col, p); x += ws[n_] + sp

CHIP_POS = [(330, 860), (760, 860), (330, 940), (760, 940)]
def draw_scene(ov, dr, k, t, lt):
    sc = SC[k]; draw_head(dr, k, lt)
    if sc.get("accent") and ACC[k] is not None and t >= ACC[k]:
        ty = ACC[k]; p = back((t - ty) / 0.45); ay = 330 + 112 * len(sc["main"]) + 14
        TX(dr, (540, ay), sc["accent"], B, int(sc.get("accent_size", 128) * max(p, 0.05)), SAFF, cl(p * 3), "mt")
        uw = eo((t - ty - 0.25) / 0.5) * 620
        if uw > 1: dr.rounded_rectangle([540 - uw / 2, ay + 158, 540 + uw / 2, ay + 168], radius=5, fill=GOLD + (255,))
    for i, (txt, te) in enumerate(CHIPS[k]):
        if t < te: continue
        cx, cy = CHIP_POS[i]; p = back((t - te) / 0.35)
        w = F(B, 40).getlength(txt) + 70
        card(dr, cx, cy, w, 66, p, cl(p * 2), SAFF, 18); TX(dr, (cx, cy), txt, B, 40 * cl(p, 0.6, 1.1), WHITE, cl(p * 2))

def draw_wipe(fr, t):
    d = ImageDraw.Draw(fr); SK = 110
    for s in S0[1:]:
        if not (s - 0.26 <= t <= s + 0.26): continue
        if t < s - 0.05: L = lerp(W + SK, -SK, eo((t - (s - 0.25)) / 0.2)); Rr = W + 2 * SK
        elif t <= s + 0.05: L = -SK; Rr = W + 2 * SK
        else: L = -SK; Rr = lerp(W + 2 * SK, -SK, eo((t - (s + 0.05)) / 0.2))
        if Rr <= L + 2: continue
        for off, col in ((70, (200, 95, 20)), (0, SAFF)):
            d.polygon([(L + SK + off, 0), (Rr + SK + off, 0), (Rr - SK + off, H), (L - SK + off, H)], fill=col)
        if s - 0.06 <= t <= s + 0.06: flame(ImageDraw.Draw(fr), 540, 1010, 170, 40, t, (11, 27, 58), sc=1)

def render(t):
    k = sc_index(t); lt = t - S0[k]
    fr = bg.copy(); d0 = ImageDraw.Draw(fr)
    d0.rectangle([0, 14, int(W * t / total), 22], fill=GOLD)
    for x, y0, v, ph, r in SP_:
        y = (y0 - v * t) % 1850; b = 0.5 + 0.5 * math.sin(t * 2 + ph)
        d0.ellipse([x - r, y - r, x + r, y + r], fill=(int(255 * (0.35 + 0.4 * b)), int(190 * (0.35 + 0.4 * b)), int(90 * (0.3 + 0.3 * b))))
    ov = Image.new("RGBA", (W, 1150), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov, "RGBA")
    draw_scene(ov, dr, k, t, lt); draw_caption(dr, k, t); fr.paste(ov, (0, 0), ov)
    exc = 0.0
    for te in ACC_T:
        if te <= t < te + 2.0: exc = max(exc, 1 - (t - te) / 2.0)
    pulse = 1.0
    for te, amp in PUNCH:
        dt = t - te
        if 0 <= dt < 0.35: pulse = max(pulse, 1 + amp * (1 - dt / 0.35) ** 2)
    ch = character(t, float(mouth[min(int(t * FPS), len(mouth) - 1)]), exc)
    s = pulse * (back(t / 0.8) if t < 1.0 else 1.0)
    cw = int(600 * s); chh = int(650 * s)
    if cw > 4: ch = ch.resize((cw, chh), Image.BILINEAR); fr.paste(ch, (540 - cw // 2, 1800 - chh), ch)
    dd = ImageDraw.Draw(fr)
    for te, x, y, pts in SPK:
        dt = t - te
        if 0 <= dt < 1.0:
            for ang, spd in pts:
                px = x + math.cos(ang) * spd * dt; py = y + math.sin(ang) * spd * dt + 300 * dt * dt; f = 1 - dt; r = 2 + 5 * f
                dd.ellipse([px - r, py - r, px + r, py + r], fill=(int(255 * f), int(190 * f + 30 * (1 - f)), int(80 * f)))
    draw_wipe(fr, t)
    z = 1 + 0.03 * cl(lt / DUR[k])
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
    sh.save(os.path.join(out_dir, "preview.png")); print("scene starts", [round(x, 1) for x in S0], "total", round(total, 1)); sys.exit()

# ---------- audio: sfx + music ----------
SR = 44100; rng = np.random.default_rng(3)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-ar", str(SR),
                os.path.join(out_dir, "_n44.wav")], input=nar.tobytes(), check=True)
with wave.open(os.path.join(out_dir, "_n44.wav")) as w: n44 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
N = int(total * SR); mix = np.zeros(N, np.float32); mix[:min(N, len(n44))] += n44[:N]
def lp(x, a0, a1):
    y = np.zeros_like(x); s = 0.0; n = len(x)
    for i in range(n): s += (a0 + (a1 - a0) * i / n) * (x[i] - s); y[i] = s
    return y
def place(sfx, t, g=1.0):
    i = int(max(0, t) * SR); e = min(N, i + len(sfx))
    if e > i: mix[i:e] += sfx[:e - i] * g
def whoosh(d=0.5):
    n = int(d * SR); y = lp(rng.standard_normal(n).astype(np.float32), 0.01, 0.4) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return y / np.abs(y).max() * 0.30
def pop(f0=640, d=0.16):
    n = int(d * SR); t = np.arange(n) / SR; f = f0 * (1 + 1.1 * np.exp(-t / 0.02))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05) * 0.30).astype(np.float32)
def impact(d=0.8):
    n = int(d * SR); t = np.arange(n) / SR; f = 42 + 90 * np.exp(-t / 0.06); lo = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    nz = lp(rng.standard_normal(n).astype(np.float32), 0.2, 0.05) * np.exp(-t / 0.05); return (lo * 0.7 + nz * 0.5).astype(np.float32) * 0.8
def riser(d=1.2):
    n = int(d * SR); t = np.arange(n) / SR; nz = lp(rng.standard_normal(n).astype(np.float32), 0.005, 0.5) * (t / d) ** 2
    f = 180 + 950 * (t / d) ** 2; s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) ** 2 * 0.5; y = nz * 3 + s
    return (y / np.abs(y).max() * 0.28).astype(np.float32)
def chime(d=1.6):
    n = int(d * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, a, tau in ((1046, 1, 0.9), (1568, 0.6, 0.7), (2093, 0.4, 0.5), (1318, 0.3, 0.6)))
    return (y / np.abs(y).max() * 0.22).astype(np.float32)
for s in S0[1:]: place(whoosh(0.5), s - 0.27)
place(pop(520), 0.35); place(chime(), 0.1, 0.6)
for k in range(len(SC)):
    for i, (txt, te) in enumerate(CHIPS[k]): place(pop(560 + 40 * i), te)
    if ACC[k] is not None: place(riser(1.0), ACC[k] - 1.0); place(impact(), ACC[k])
place(chime(), S0[LAST] + 1.0, 0.8)
bpm = 96; beat = 60 / bpm; t_ = np.arange(N) / SR; mus = np.zeros(N, np.float32)
chords = [(110, 130.81, 164.81), (87.31, 110, 130.81), (130.81, 164.81, 196), (98, 123.47, 146.83)]
bar = beat * 8
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
        mus[ts:ts + m] += (np.sin(2 * np.pi * f * tq) + 0.3 * np.sin(4 * np.pi * f * tq)) * np.exp(-tq / 0.22) * np.minimum(1, tq / 0.005) * 0.05
        if j % 2 == 1:
            h = min(int(0.03 * SR), N - ts); mus[ts:ts + h] += np.diff(rng.standard_normal(h + 1)).astype(np.float32) * np.exp(-np.arange(h) / SR / 0.012) * 0.04
nn = np.zeros(N, np.float32); nn[:min(N, len(n44))] = n44[:N]
w_ = int(0.05 * SR); rms = np.sqrt(np.convolve(nn ** 2, np.ones(w_) / w_, mode="same"))
act = (rms > 0.02).astype(np.float32); kk = int(0.3 * SR); act = np.convolve(act, np.ones(kk) / kk, mode="same")
duck = 1 - 0.65 * np.clip(act * 1.5, 0, 1); fade = np.minimum(1, t_ / 0.8) * np.minimum(1, (total - t_) / 1.6)
mix += mus * duck * fade; mix = np.tanh(mix * 1.2) / 1.2
mixpath = os.path.join(out_dir, "_mix.wav")
with wave.open(mixpath, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
OUT = os.path.join(out_dir, "video.mp4")
enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", mixpath,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-15:TP=-1.5", "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart", "-shortest", OUT], stdin=subprocess.PIPE)
for i in range(nF):
    enc.stdin.write(render(i / FPS).tobytes())
    if i % 150 == 0: print(i, nF, flush=True)
enc.stdin.close(); enc.wait()
meta = {k: DATA.get(k) for k in ("title", "description", "tags", "needs_review", "topic", "sources")}
meta["duration_seconds"] = round(total, 1)
json.dump(meta, open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for f in os.listdir(out_dir):
    if f.startswith("_"): os.remove(os.path.join(out_dir, f))
print("DONE", OUT)
