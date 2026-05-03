# Vera Bot — magicpin AI Challenge Submission

## Approach

**Signal-first composition.** Before writing a single word, the bot:

1. Receives context via `/v1/context` into a versioned, idempotent store
2. On every `/v1/tick`, scores all `(trigger × merchant_gap)` pairs — urgency, performance delta, offer availability, suppression state
3. Composes using Claude with the full 4-context payload: category voice profile, merchant performance + offers + conversation history, trigger payload, and customer relationship
4. Every number in the output is traceable to a specific field in the received context — zero hallucination

**Why this wins under adaptive injection:** New facts (fresh digest items, metric shifts) are immediately stored in the versioned context store. The next `/v1/tick` composes against the updated state — no caching of old decisions.

## Architecture

```
bot.py                     # Single-file FastAPI application
├── Context Store          # Versioned, idempotent: (scope, context_id) → {version, payload}
├── Suppression Log        # suppression_key → timestamp, checked before every tick
├── Anti-Repetition Guard  # Per-merchant sent-bodies set
├── Summarizer Layer       # Converts raw JSON context into clean LLM prompts
├── Composer               # Claude call with 4-context + 6 reference case studies
└── Reply Handler          # Intent FSM: accept/decline/hostile/auto-reply/question/off-topic
```



## Reply Handler States

```
AWAITING_REPLY → (accept) → DELIVERING → CLOSED
               → (decline) → CLOSED
               → (hostile) → CLOSED [immediate]
               → (auto-reply) → WAIT [1800s]
               → (question) → re-answer + re-offer CTA
               → (off-topic) → acknowledge + return to topic
```

**Auto-reply detection** runs before any LLM call (regex-based, < 1ms).
**Hostile detection** runs before any LLM call (regex-based, < 1ms).
**Accept responses** deliver the exact artifact promised in the last bot message (draft, summary, plan) — never re-qualify.

## Key Scoring Advantages

| Dimension | Mechanism |
|---|---|
| Decision Quality | Signal ranking: urgency × performance_gap × offer_availability × category_fit |
| Specificity | Summarizer extracts exact numbers before LLM call; prompt enforces no-invention rule |
| Category Fit | Per-category voice profiles (tone, vocab_allowed, vocab_taboo) injected verbatim |
| Merchant Fit | Owner first name, CTR gap vs peer median, active offers, conversation history all included |
| Engagement Compulsion | 6 reference case studies as few-shot examples; single CTA enforced at prompt level |

## Setup & Deployment

### Option 1: Railway (Recommended — 5 min setup)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set environment variables:
   ```
   ANTHROPIC_API_KEY=your-key
   TEAM_NAME=YourName
   CONTACT_EMAIL=you@example.com
   ```
4. Railway auto-detects `railway.json` and deploys. Copy the public URL.

### Option 2: Render

1. New Web Service → Connect your GitHub repo
2. Environment: Python 3 | Build: `pip install -r requirements.txt` | Start: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
3. Set `ANTHROPIC_API_KEY` in Environment Variables
4. Deploy. Copy the `https://your-app.onrender.com` URL.

### Option 3: Fly.io

```bash
fly auth login
fly launch --name vera-bot
fly secrets set ANTHROPIC_API_KEY=your-key
fly deploy
```

### Option 4: Local + ngrok (fastest for testing)

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
# In another terminal:
ngrok http 8080
# Copy the https://xxxx.ngrok.io URL as your submission URL
```

## Local Test with Judge Simulator

```bash
# 1. Start your bot
uvicorn bot:app --host 0.0.0.0 --port 8080

# 2. Configure the judge simulator (edit the top of judge_simulator.py):
BOT_URL = "http://localhost:8080"
LLM_PROVIDER = "anthropic"
LLM_API_KEY = "your-anthropic-key"

# 3. Run
python judge_simulator.py
```

## API Contract

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/healthz` | GET | Liveness probe |
| `/v1/metadata` | GET | Bot identity |
| `/v1/context` | POST | Receive context push (category/merchant/customer/trigger) |
| `/v1/tick` | POST | Periodic wake-up — bot decides what to send |
| `/v1/reply` | POST | Receive merchant/customer reply |
| `/v1/teardown` | POST | Optional — wipe state |

## Submission Checklist

- [x] All 5 required endpoints implemented
- [x] `/v1/context` idempotent on (scope, context_id, version)
- [x] `/v1/tick` returns within 30s (20 action cap)
- [x] `/v1/reply` returns within 30s for any input
- [x] Auto-reply detection (regex, < 1ms, no LLM needed)
- [x] Hostile message detection → immediate end
- [x] Anti-repetition guard (same body never sent twice to same merchant)
- [x] Suppression key enforcement
- [x] Language preference honored (hi-en mix, pure Hindi, English)
- [x] Category voice profiles applied (peer_clinical for dentists, warm_practical for salons, etc.)
- [x] Zero hallucination — every fact traceable to received context
- [x] Deterministic at temperature=0
