"""Optional: download licensed stock clips (Pixabay, free API key) for scenes that have a "broll" search term.
Usage: python pipeline/broll.py script.json out_dir        (needs env PIXABAY_API_KEY; exits quietly without it)
Writes out_dir/broll/broll.json which render.py picks up. Pixabay Content License: free commercial use, no attribution required;
we still record the source of every clip in metadata.json.
"""
import json, os, sys, urllib.parse, urllib.request

script, out = sys.argv[1], sys.argv[2]
key = os.environ.get("PIXABAY_API_KEY")
if not key: print("PIXABAY_API_KEY not set - skipping B-roll (graphics only)."); sys.exit(0)
data = json.load(open(script, encoding="utf-8")); folder = os.path.join(out, "broll"); os.makedirs(folder, exist_ok=True)
index = {}; used = 0
for k, sc in enumerate(data["scenes"]):
    q = str(sc.get("broll") or "").strip()[:60]
    if not q or used >= 2: continue
    try:
        url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode({"key": key, "q": q, "per_page": 12, "safesearch": "true"})
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "social-ceo/1.1"}), timeout=30) as r: hits = json.load(r).get("hits", [])
        hits = [h for h in hits if 4 <= h.get("duration", 0) <= 40]
        if not hits: print(f"scene {k}: no clips for '{q}'"); continue
        h = hits[0]; v = h["videos"]; pick = next((v[s] for s in ("medium", "small", "large", "tiny") if v.get(s, {}).get("url")), None)
        if not pick: continue
        dest = os.path.join(folder, f"s{k}.mp4")
        with urllib.request.urlopen(urllib.request.Request(pick["url"], headers={"User-Agent": "social-ceo/1.1"}), timeout=90) as r, open(dest, "wb") as f: f.write(r.read())
        index[str(k)] = {"file": dest, "query": q, "pixabay_id": h["id"], "page": h.get("pageURL"), "author": h.get("user"), "license": "Pixabay Content License"}
        used += 1; print(f"scene {k}: '{q}' -> {dest}")
    except Exception as ex:
        print(f"scene {k}: B-roll failed ({ex}) - using graphics instead")
json.dump(index, open(os.path.join(folder, "broll.json"), "w"), indent=2)
