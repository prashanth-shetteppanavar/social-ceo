"""Pick a safe trending topic and write a script JSON using FREE LLM APIs.
Usage: python pipeline/generate.py script.json
Secrets (any one is enough; tried in this order): OPENROUTER_API_KEY, MISTRAL_API_KEY, NVIDIA_API_KEY
Optional model overrides (tried first): OPENROUTER_MODEL, MISTRAL_MODEL, NVIDIA_MODEL
Offline test: GENERATE_MOCK=path/to/response.txt python pipeline/generate.py out.json
If every provider fails, a built-in evergreen script is used (needs_review stays true).
"""
import datetime, json, os, re, sys, time, urllib.error, urllib.request, xml.etree.ElementTree as ET

NEWS_RSS = os.environ.get("NEWS_RSS", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en")
DEADLINE = time.time() + 14 * 60          # stop trying after 14 minutes in total
BAD = re.compile(r"embed|rerank|vision|-vl|vl-|image|whisper|guard|safeguard|moderation|reward|parse|clip|vila|neva|kosmos|paligemma|fuyu|riva|safety|tts|audio|ocr|retriev", re.I)

SYSTEM = """You write scripts for a 30-45 second vertical explainer video narrated by a friendly cartoon diya (oil lamp) named Deepu.
RULES (strict):
- Use ONLY facts that appear in the provided headlines. Never invent numbers, quotes, names, dates or causes.
- Pick ONE topic that is safe and useful: festivals, consumer rules, science, tech, sports results, education, travel, money basics.
- SKIP: politics and elections, crime, accidents, deaths, tragedies, religion disputes, health or legal advice, rumours, celebrities' private lives.
- If the topic is a news item or a legal/government rule, set needs_review to true. Evergreen topics may be false.
- Plain simple English for Indian viewers. Hook in the first sentence. No greeting at the start.
- 4 to 6 scenes. Each scene: label (<=26 chars, caps), main (1-2 lines, each <=22 chars), spoken (one or two short sentences).
- Optional per scene: accent (<=12 chars caps) with accent_word (a word that appears in spoken), chips (up to 4 items, each text <=14 chars and at = a word in spoken), keywords (words to highlight).
- Last scene is a call to action asking viewers to share.
Return ONLY JSON: {"topic":"","title":"<=70 chars","description":"2-3 sentences + 3-5 hashtags","tags":[],"needs_review":true,"sources":"names of outlets from the headlines","scenes":[{"label":"","main":[""],"spoken":"","accent":"","accent_word":"","chips":[{"text":"","at":""}],"keywords":[]}]}"""

class HttpErr(Exception):
    def __init__(self, code, msg): super().__init__(f"HTTP {code}: {msg}"); self.code = code

def http_json(url, data=None, key=None, timeout=120):
    headers = {"Content-Type": "application/json", "User-Agent": "social-ceo/1.1"}
    if key: headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:240].replace("\n", " ")
        raise HttpErr(e.code, body)

def headlines():
    with urllib.request.urlopen(urllib.request.Request(NEWS_RSS, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
        root = ET.fromstring(r.read())
    return [t for t in ((it.findtext("title") or "").strip() for it in root.iter("item")) if t][:40]

def chat(url, key, model, hl, merge_system=False):
    user = "Today's headlines (title - outlet):\n" + "\n".join("- " + h for h in hl)
    msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user}] if merge_system else [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    d = http_json(url, {"model": model, "temperature": 0.4, "max_tokens": 1800, "messages": msgs}, key)
    return d["choices"][0]["message"]["content"]

def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])

bare = lambda w: re.sub(r"\W", "", w).lower()
def validate(d):
    sc = d.get("scenes")
    if not isinstance(sc, list) or not 3 <= len(sc) <= 8: raise ValueError("need 3-8 scenes")
    d["needs_review"] = bool(d.get("needs_review", True))
    for k in ("title", "description"):
        if not d.get(k): raise ValueError("missing " + k)
    flat = lambda v: ", ".join(str(x) for x in v if x) if isinstance(v, (list, tuple)) else ("" if v is None else str(v))
    d["title"] = flat(d["title"])[:95]; d["description"] = flat(d["description"]); d["sources"] = flat(d.get("sources"))
    t = d.get("tags") or []
    d["tags"] = [x.strip() for x in t.split(",") if x.strip()] if isinstance(t, str) else [str(x) for x in t]
    if isinstance(d["needs_review"], str): d["needs_review"] = d["needs_review"].strip().lower() != "false"
    for s in sc:
        for k in ("label", "main", "spoken"):
            if not s.get(k): raise ValueError("scene missing " + k)
        if isinstance(s["main"], str): s["main"] = [s["main"]]
        s["main"] = [str(m)[:26] for m in s["main"][:2]]; s["label"] = str(s["label"])[:30].upper(); s["spoken"] = str(s["spoken"])
        if s.get("accent") is not None: s["accent"] = str(s["accent"])[:14]
        words = {bare(w) for w in s["spoken"].split()}
        if s.get("accent") and bare(s.get("accent_word", "")) not in words: s.pop("accent"); s.pop("accent_word", None)
        s["chips"] = [c for c in (s.get("chips") or [])[:4] if isinstance(c, dict) and c.get("text") and bare(c.get("at", "")) in words]
        s["keywords"] = [w for w in (s.get("keywords") or []) if isinstance(w, str)]
    return d

# ---- model discovery: ask each provider which models exist TODAY ----
def models_openrouter(key):
    out = []
    for m in http_json("https://openrouter.ai/api/v1/models", None, None, 30).get("data", []):
        mid = m.get("id", ""); p = m.get("pricing") or {}
        free = mid.endswith(":free") or (str(p.get("prompt")) == "0" and str(p.get("completion")) == "0")
        if free and not BAD.search(mid): out.append((m.get("context_length") or 0, mid))
    return [m for _, m in sorted(out, reverse=True)][:8]

def models_mistral(key):
    want = ["mistral-small-latest", "open-mistral-nemo", "mistral-medium-latest", "ministral-8b-latest", "mistral-large-latest"]
    try:
        have = {m["id"] for m in http_json("https://api.mistral.ai/v1/models", None, key, 30).get("data", [])}
        got = [m for m in want if m in have]
        return got or want
    except Exception: return want

def models_nvidia(key):
    pref = ["llama-3.3-70b", "llama-3.1-70b", "llama-3.1-405b", "mistral", "qwen", "gemma", "deepseek", "llama"]
    try:
        ids = [m["id"] for m in http_json("https://integrate.api.nvidia.com/v1/models", None, key, 30).get("data", [])]
    except Exception: return ["meta/llama-3.3-70b-instruct"]
    ids = [i for i in ids if not BAD.search(i) and re.search(r"instruct|chat|-it\b", i, re.I)]
    ids.sort(key=lambda i: next((n for n, p in enumerate(pref) if p in i.lower()), 99))
    return ids[:6]

PROVIDERS = [
    ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "https://openrouter.ai/api/v1/chat/completions", models_openrouter),
    ("MISTRAL_API_KEY", "MISTRAL_MODEL", "https://api.mistral.ai/v1/chat/completions", models_mistral),
    ("NVIDIA_API_KEY", "NVIDIA_MODEL", "https://integrate.api.nvidia.com/v1/chat/completions", models_nvidia),
]

def try_model(url, key, model, hl):
    """Up to 3 attempts; waits on rate limits; retries once without a system role."""
    merge = False
    for attempt in range(3):
        if time.time() > DEADLINE: raise TimeoutError("overall deadline reached")
        try:
            return validate(parse(chat(url, key, model, hl, merge)))
        except HttpErr as e:
            if e.code == 429 or e.code >= 500: time.sleep(20 * (attempt + 1)); continue
            if e.code == 400 and not merge: merge = True; continue
            raise
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            if attempt == 2: raise
    raise RuntimeError("rate limited")

# ---- evergreen fallback (used only if every AI provider fails) ----
def S(label, main, spoken, **kw): return dict(label=label, main=main, spoken=spoken, **kw)
FALLBACKS = [
 {"topic": "Diwali safety basics", "title": "3 Simple Diwali Safety Tips for Your Family", "needs_review": True, "sources": "General safety guidance",
  "tags": ["Diwali", "safety", "Diwali tips"], "description": "Three simple safety habits for a safer festival with family. #Diwali #Safety #Shorts",
  "scenes": [S("DIWALI SAFETY", ["3 simple tips for", "a safer Diwali"], "Want a safer Diwali? Here are three simple tips.", keywords=["safer", "diwali"]),
   S("TIP ONE", ["Keep water", "close by"], "One. Keep a bucket of water or sand close by, just in case.", chips=[{"text": "Water", "at": "water"}, {"text": "Sand", "at": "sand"}], keywords=["water", "sand"]),
   S("TIP TWO", ["Choose your", "clothes wisely"], "Two. Wear snug cotton clothes and keep loose dupattas and sleeves away from diyas.", keywords=["cotton", "diyas"]),
   S("TIP THREE", ["Children need", "an adult nearby"], "Three. Keep children with an adult near any flame. Please share this with your family group.", accent="SHARE", accent_word="share", keywords=["children", "share"])]},
 {"topic": "UPI safety", "title": "UPI Safety: 3 Rules Everyone Should Know", "needs_review": True, "sources": "General safety guidance",
  "tags": ["UPI", "online safety", "scam alert"], "description": "Three simple rules to stay safe while using UPI. #UPI #ScamAlert #Shorts",
  "scenes": [S("UPI SAFETY", ["3 rules to avoid", "UPI scams"], "Do you use UPI? Here are three rules that help you avoid scams.", keywords=["upi", "scams"]),
   S("RULE ONE", ["Never share", "your UPI PIN"], "One. Never share your UPI PIN with anyone, not even someone who says they are from your bank.", accent="NEVER", accent_word="never", keywords=["pin", "never"]),
   S("RULE TWO", ["You do not enter a PIN", "to receive money"], "Two. You never need to enter your PIN to receive money. If a request asks for your PIN, it is a trap.", keywords=["pin", "receive", "trap"]),
   S("RULE THREE", ["Check the name", "before you pay"], "Three. Check the name shown before you pay. Share this with someone who needs it.", accent="SHARE", accent_word="share", keywords=["check", "share"])]},
 {"topic": "Phone battery care", "title": "Make Your Phone Battery Last Longer", "needs_review": True, "sources": "General tech guidance",
  "tags": ["phone battery", "tech tips"], "description": "Simple habits that are kind to your phone battery. #TechTips #Battery #Shorts",
  "scenes": [S("PHONE TIPS", ["Keep your battery", "healthy for longer"], "Want your phone battery to last longer? Try these habits.", keywords=["battery", "longer"]),
   S("HABIT ONE", ["Avoid", "extreme heat"], "One. Keep your phone away from heat, like a hot car dashboard or direct sun.", chips=[{"text": "Hot car", "at": "car"}, {"text": "Direct sun", "at": "sun"}], keywords=["heat", "sun"]),
   S("HABIT TWO", ["Use the", "right charger"], "Two. Use the charger that came with your phone, or a good quality one.", keywords=["charger"]),
   S("HABIT THREE", ["Share the", "tip"], "Three. Turn on battery saver when you are low. Share this with a friend.", accent="SHARE", accent_word="share", keywords=["saver", "share"])]},
]

def fallback():
    d = json.loads(json.dumps(FALLBACKS[datetime.date.today().toordinal() % len(FALLBACKS)])); return validate(d)

def save(d, out, note):
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print(note)

def main(out):
    mock = os.environ.get("GENERATE_MOCK")
    if mock: save(validate(parse(open(mock, encoding="utf-8").read())), out, "mock ok"); return
    try: hl = headlines()
    except Exception as ex:
        print("news fetch failed:", ex, file=sys.stderr); save(fallback(), out, "USING FALLBACK SCRIPT (news unavailable)"); return
    for env, menv, url, discover in PROVIDERS:
        key = os.environ.get(env)
        if not key: continue
        models = []
        if os.environ.get(menv): models.append(os.environ[menv])
        try: models += [m for m in discover(key) if m not in models]
        except Exception as ex: print(f"{env}: could not list models ({ex})", file=sys.stderr)
        print(f"{env}: will try {models[:8]}")
        for model in models[:8]:
            try:
                d = try_model(url, key, model, hl); save(d, out, f"script from {env} / {model}"); return
            except TimeoutError: break
            except Exception as ex: print(f"  failed {model}: {str(ex)[:200]}", file=sys.stderr)
    save(fallback(), out, "USING FALLBACK SCRIPT (all AI providers failed)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "script.json")
