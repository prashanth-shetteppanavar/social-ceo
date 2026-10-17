# social-ceo — free daily video pipeline

Every day at 08:00 IST, GitHub Actions (free) does this:
1. Reads today's headlines (Google News RSS, free).
2. Asks a FREE LLM API (OpenRouter / Mistral / NVIDIA, whichever key you set) for a safe 30-45 s script.
3. Narrates it with Piper (free, offline voice) and renders a 9:16 video with FFmpeg: animated captions, Deepu the Diya, sound design, music.
4. Publishes `video.mp4` + `metadata.json` as a GitHub Release. If the topic is news/legal, the release is marked **pre-release = needs review**.

Nothing here posts to social media by itself. Posting is step 2 below.

## One-time setup (about 15 minutes)
1. Create a GitHub repository (public repos get unlimited free Actions minutes; private ones get a monthly allowance — check GitHub's current limits). Upload every file in this folder (keep the `.github` folder).
2. Repo → Settings → Secrets and variables → Actions → New repository secret. Add at least one of: `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `NVIDIA_API_KEY`. Never put keys in code or chat.
   - Free model names change. If a run fails with "model not found", set `OPENROUTER_MODEL` (or the Mistral/NVIDIA one) as a repository *variable* or edit the default in `pipeline/generate.py`.
3. Actions tab → "daily-video" → Run workflow. Wait ~10–20 minutes. A Release with the video appears.
4. Install the GitHub mobile app and turn on notifications for the repo, so you hear about each new Release.

## Posting (choose one)
- **Manual, 2 minutes/day (works today):** download `video.mp4`, read `metadata.json`, upload to YouTube/Instagram yourself, or drop it in your Drive "Social media" folder and ask Claude to schedule it through Metricool.
- **Automatic, free (needs testing):** a Make.com free-plan scenario (1,000 credits/month, 2 scenarios, 15-minute schedule) that watches for the new Release, downloads the video, and posts to YouTube / Facebook Page / Instagram using Make's official modules. I have NOT built or tested this. Only post automatically when `needs_review` is false.
- **Do not** upload with your own YouTube API project: videos from new unaudited API projects are forced to *private*. Metricool's API is paid-plan only, so it cannot be called from here on the free plan.

## Safety rules built in
- The prompt forbids politics, crime, tragedies, health/legal advice, and invented facts. News/legal topics always need review.
- Headline-only sourcing can still produce mistakes. Read the script (`script.json` in the Release) before posting anything that mentions courts, laws or people.

## Test locally
    pip install -r requirements.txt   # also install ffmpeg
    # put en-us-ryan-high.onnx and .onnx.json in models/
    python pipeline/render.py samples/sample.json out
    GENERATE_MOCK=samples/mock_llm_response.txt python pipeline/generate.py /tmp/x.json   # offline validator test
