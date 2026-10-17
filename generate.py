"""Pick a safe trending topic and write a script JSON using a FREE LLM API.
Usage: python pipeline/generate.py script.json
Env (set as GitHub Secrets; any one is enough, they are tried in this order):
  OPENROUTER_API_KEY (+ optional OPENROUTER_MODEL, default below)
  MISTRAL_API_KEY    (+ optional MISTRAL_MODEL)
  NVIDIA_API_KEY     (+ optional NVIDIA_MODEL)
Test without network: GENERATE_MOCK=path/to/response.txt python pipeline/generate.py out.json
"""
import json, os, re, sys, urllib.request, xml.etree.ElementTree as ET

NEWS_RSS = os.environ.get("NEWS_RSS", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en")
PROVIDERS = [
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")),
    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions", os.environ.get("MISTRAL_MODEL", "mistral-small-latest")),
    ("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1/chat/completions", os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")),
]
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

def headlines():
    with urllib.request.urlopen(urllib.request.Request(NEWS_RSS, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.iter("item"):
        t = (it.findtext("title") or "").strip()
        if t: out.append(t)
    return out[:40]

def call(url, key, model, headlines_):
    body = json.dumps({"model": model, "temperature": 0.4, "max_tokens": 1800,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Today's headlines (title - outlet):\n" + "\n".join("- " + h for h in headlines_)}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def parse(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])

bare = lambda w: re.sub(r"\W", "", w).lower()
def validate(d):
    """Return a cleaned script or raise ValueError."""
    sc = d.get("scenes")
    if not isinstance(sc, list) or not 3 <= len(sc) <= 8: raise ValueError("need 3-8 scenes")
    d["needs_review"] = bool(d.get("needs_review", True))
    for k in ("title", "description"):
        if not d.get(k): raise ValueError("missing " + k)
    d["title"] = d["title"][:95]; d.setdefault("tags", []); d.setdefault("sources", "")
    for s in sc:
        for k in ("label", "main", "spoken"):
            if not s.get(k): raise ValueError("scene missing " + k)
        if isinstance(s["main"], str): s["main"] = [s["main"]]
        s["main"] = [m[:26] for m in s["main"][:2]]; s["label"] = s["label"][:30].upper()
        words = {bare(w) for w in s["spoken"].split()}
        if s.get("accent"):
            if bare(s.get("accent_word", "")) not in words: s.pop("accent"); s.pop("accent_word", None)
        s["chips"] = [c for c in (s.get("chips") or [])[:4] if isinstance(c, dict) and c.get("text") and bare(c.get("at", "")) in words]
        s["keywords"] = [w for w in (s.get("keywords") or []) if isinstance(w, str)]
    return d

def main(out):
    mock = os.environ.get("GENERATE_MOCK")
    if mock:
        d = validate(parse(open(mock, encoding="utf-8").read())); json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print("mock ok"); return
    hl = headlines(); errors = []
    for env, url, model in PROVIDERS:
        key = os.environ.get(env)
        if not key: continue
        try:
            d = validate(parse(call(url, key, model, hl)))
            json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print("script from", env, model); return
        except Exception as ex:
            errors.append(f"{env}: {ex}"); print("provider failed:", env, ex, file=sys.stderr)
    sys.exit("No provider produced a valid script: " + " | ".join(errors))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "script.json")
