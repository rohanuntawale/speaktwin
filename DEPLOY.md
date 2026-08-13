# Deploying SpeakTwin

The frontend and backend deploy to different places, and that split is not
arbitrary — it comes from what each half actually needs.

---

## Why not all of it on Vercel

Vercel runs serverless functions. This backend needs two things serverless
does not give you.

**Size.** Measured from a real install:

| Bundle | Size | Vercel limit |
|---|---|---|
| FastAPI + numpy + scipy + soundfile | 216 MB | 250 MB |
| **+ faster-whisper + ctranslate2 + onnxruntime** | **324 MB** | ❌ over |
| + torch (the optional ML layer) | +497 MB | ❌❌ |

Dropping local Whisper squeaks under at 216 MB, but that leaves ~34 MB of
headroom before the Groq and OpenAI clients — a build that breaks on the next
dependency bump.

**State.** Serverless functions are stateless. The session store lives in
process memory, so between invocations you would lose:

- cross-chunk confidence smoothing
- cumulative filler and keyword totals
- the adaptive silence gate
- decoder context carried across chunk seams
- session reports

Every chunk would be analysed in isolation — precisely the behaviour the
session layer exists to fix. Add a 10-second Hobby function timeout against
Whisper's cold start and no persistent filesystem for model weights.

**So:** frontend on Vercel, which it is genuinely good at. Backend on a host
that keeps a process alive.

---

## Step 1 — Backend

Both options read the existing `Dockerfile`. Pick one.

### Render (simplest, has a free tier)

1. Push this repo to GitHub.
2. [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint** →
   select the repo. It reads `render.yaml`.
3. Set these in the dashboard (they are marked `sync: false` so they never
   enter git):
   - `GROQ_API_KEY` — get one free at [console.groq.com](https://console.groq.com)
   - `CORS_ORIGINS` — your Vercel URL, once you have it
   - `API_KEY` — any long random string
4. Deploy, then confirm: `https://<your-app>.onrender.com/api/health`

> Free instances have 512 MB of RAM and sleep after 15 minutes idle, so the
> first request after a nap takes ~30–60 s. `render.yaml` therefore sets
> `STT_ENGINE=groq` — local Whisper risks an OOM kill in 512 MB.

### Fly.io (better if you want local Whisper)

```bash
fly launch --no-deploy
fly secrets set GROQ_API_KEY=... CORS_ORIGINS=https://your-app.vercel.app
fly deploy
```

`fly.toml` requests 1 GB, which comfortably holds `tiny.en`, so it keeps
`STT_ENGINE=local` and nothing leaves the machine.

---

## Step 2 — Frontend on Vercel

1. Edit **`vercel.json`** and replace the placeholder with your backend URL:

   ```json
   "destination": "https://your-app.onrender.com/api/:path*"
   ```

2. Deploy:

   ```bash
   npm i -g vercel
   vercel login        # opens your browser — must be you
   vercel --prod
   ```

   Or import the repo at [vercel.com/new](https://vercel.com/new). No build
   command; output directory is `frontend`.

3. Set `CORS_ORIGINS` on the backend to the Vercel URL and redeploy it.

### Why the rewrite matters

`vercel.json` proxies `/api/*` to your backend, so the browser only ever
talks to your Vercel domain. Requests are **same-origin**, which means no
CORS preflight, no mixed-content problems, and no API base to configure in
the frontend. It costs one extra network hop.

If you would rather call the backend directly, drop the rewrite and set the
base before `app.js` loads:

```html
<script>window.SPEAKTWIN_API_BASE = "https://your-app.onrender.com";</script>
```

Then `CORS_ORIGINS` on the backend must list your Vercel domain exactly.

---

## Step 3 — Check it

```bash
curl https://your-app.vercel.app/api/health
```

Then open the site and **press the mic**. Browsers only grant microphone
access over HTTPS or on `localhost`; Vercel serves HTTPS, so this is fine —
but the `Permissions-Policy: microphone=(self)` header in `vercel.json` is
what stops some browsers refusing the request outright.

---

## Before you make the URL public

| | Why |
|---|---|
| Set `API_KEY` | `/api/analyze` spends money at Groq/OpenRouter on every call. Without a key, anyone who finds the URL can drain your credit. |
| Set `CORS_ORIGINS` | Defaults to `*`, which is fine locally and careless in public. |
| Lower `RATE_LIMIT_PER_MINUTE` | 60/min per client suits one speaker; a chunk every 2.5 s is ~24/min. |
| Keep one instance | The session store and rate limiter are in-process. Multiple replicas each keep their own copy, so a user's session would vanish depending on which one answers. Scaling out needs Redis first. |

If you set `API_KEY`, the frontend has to send it. Add to `app.js` in the
`fetch` calls:

```js
headers: { "X-API-Key": window.SPEAKTWIN_API_KEY }
```

…which puts the key in client-side JavaScript, where anyone can read it. For
a personal deployment that is an acceptable trade for keeping crawlers out.
For anything real, put a login in front instead.

---

## Cost

| Piece | Free tier | Notes |
|---|---|---|
| Vercel (frontend) | Generous | Static files; this will not exceed it |
| Render (backend) | 750 hrs/month | Sleeps when idle |
| Fly.io (backend) | Small allowance | ~$2–5/month for always-on |
| Groq (STT) | Free tier available | Fastest option by a distance |
| OpenRouter (LLM) | Pay per call | Optional; throttled to one call per 8 s per session |

Running local Whisper on Fly costs nothing per request and keeps audio on
your own machine.
