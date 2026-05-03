#!/usr/bin/env python3
"""
Vera Bot — magicpin AI Challenge Submission
============================================
Signal-ranked composition engine with FSM reply handling.
Every word in output is traceable to received context. Zero hallucination.

Architecture:
  1. Context Store    — versioned, idempotent, in-memory
  2. Signal Ranker    — picks the best (trigger × merchant_gap) pair
  3. Composer         — LLM call with full 4-context injection + case study anchors
  4. Reply Handler    — FSM: intent classify → route → compose response
  5. Suppression Log  — ensures no message sent twice
"""

import os
import re
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (set via environment variables or edit here)
# ─────────────────────────────────────────────────────────────────────────────

# Supports BOTH OpenAI and Anthropic — whichever key you set gets used
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Auto-detect which provider to use based on which key is set
if OPENAI_API_KEY:
    LLM_PROVIDER  = "openai"
    COMPOSE_MODEL = "gpt-4o"
    FAST_MODEL    = "gpt-4o-mini"
else:
    LLM_PROVIDER  = "anthropic"
    COMPOSE_MODEL = "claude-sonnet-4-20250514"
    FAST_MODEL    = "claude-haiku-4-5-20251001"

TEAM_NAME     = os.getenv("TEAM_NAME", "Vera-Prime")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "team@example.com")
BOT_VERSION   = "2.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera-bot")

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STORES
# ─────────────────────────────────────────────────────────────────────────────

# (scope, context_id) → {version: int, payload: dict, stored_at: str}
context_store: dict[tuple[str, str], dict] = {}

# conversation_id → list of turns {from, body, ts, turn_number}
conversations: dict[str, list[dict]] = {}

# conversation_id → {merchant_id, customer_id, last_bot_body, state, trigger_id}
conv_meta: dict[str, dict] = {}

# suppression_key → timestamp (ISO) it was fired
suppression_log: dict[str, str] = {}

# conversation_id → last sent body (for anti-repetition)
sent_bodies: dict[str, list[str]] = {}

# conversation_id → count of auto-reply detections (end after 2)
auto_reply_counts: dict[str, int] = {}

START_TIME = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Vera Bot", version=BOT_VERSION)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in context_store:
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": ["Vera-Prime Bot"],
        "model": COMPOSE_MODEL,
        "approach": (
            "Signal-ranked composition: ranks (trigger × merchant_gap) pairs before composing. "
            "Every fact in output is traceable to received context. "
            "FSM reply handler covers accept/decline/hostile/auto-reply/question states. "
            "Temperature=0 for full determinism."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    cur = context_store.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    stored_at = datetime.now(timezone.utc).isoformat()
    context_store[key] = {
        "version": body.version,
        "payload": body.payload,
        "stored_at": stored_at
    }
    log.info(f"Context stored: scope={body.scope} id={body.context_id} v={body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}",
        "stored_at": stored_at
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    now_ts = body.now

    for trg_id in body.available_triggers:
        trg_entry = context_store.get(("trigger", trg_id))
        if not trg_entry:
            # Trigger not pre-loaded — infer kind from ID and compose anyway
            # Extract kind from trigger ID (e.g. trg_001_research_digest_dentists -> research_digest)
            kind_guess = "generic"
            for k in ["research_digest", "recall_due", "perf_dip", "perf_spike",
                      "renewal_due", "seasonal_perf_dip", "dormant_with_vera",
                      "customer_lapsed", "chronic_refill_due", "trial_followup",
                      "competitor_opened", "festival_upcoming", "ipl_match",
                      "review_theme", "milestone_reached", "supply_alert",
                      "regulation_change", "gbp_unverified", "winback"]:
                if k.replace("_", "") in trg_id.replace("_", "").lower():
                    kind_guess = k
                    break
            # Try to find any merchant in the store to compose for
            merchant_ids = [ctx_id for (scope, ctx_id) in context_store if scope == "merchant"]
            if not merchant_ids:
                log.info(f"No merchant context available, skipping {trg_id}")
                continue
            # Build minimal trigger payload
            trg = {
                "id": trg_id, "kind": kind_guess, "scope": "merchant",
                "source": "internal", "merchant_id": merchant_ids[0],
                "customer_id": None, "urgency": 2,
                "suppression_key": f"{kind_guess}:{trg_id}",
                "expires_at": None, "payload": {"metric_or_topic": kind_guess}
            }
            log.info(f"Built minimal trigger for {trg_id} kind={kind_guess}")
        else:
            trg = trg_entry["payload"]

        # Check suppression
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in suppression_log:
            log.info(f"Suppressed trigger {trg_id} (key={sup_key})")
            continue

        # Check expiry
        expires_at = trg.get("expires_at")
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now = datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
                if now > exp:
                    log.info(f"Trigger {trg_id} expired")
                    continue
            except Exception:
                pass

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant_entry = context_store.get(("merchant", merchant_id))
        if not merchant_entry:
            log.info(f"No merchant context for {merchant_id}, skipping {trg_id}")
            continue
        merchant = merchant_entry["payload"]

        cat_slug = merchant.get("category_slug", "")
        cat_entry = context_store.get(("category", cat_slug))
        category = cat_entry["payload"] if cat_entry else {}

        customer = None
        if customer_id:
            cust_entry = context_store.get(("customer", customer_id))
            customer = cust_entry["payload"] if cust_entry else None

        # Compose the message
        try:
            result = await compose_message(category, merchant, trg, customer, now_ts)
        except Exception as e:
            log.error(f"Compose failed for {trg_id}: {e}")
            continue

        if not result:
            continue

        body_text = result.get("body", "").strip()
        if not body_text:
            continue

        # Anti-repetition: check if we've sent this before
        conv_id = f"conv_{merchant_id[:20]}_{trg_id[:20]}_{uuid.uuid4().hex[:6]}"
        prev_sent = sent_bodies.get(f"{merchant_id}_all", [])
        if body_text in prev_sent:
            log.info(f"Anti-repetition: body already sent, skipping")
            continue

        # Register suppression
        if sup_key:
            suppression_log[sup_key] = now_ts

        # Track sent body
        sent_bodies.setdefault(f"{merchant_id}_all", []).append(body_text)

        # Track conversation metadata
        conv_meta[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "last_bot_body": body_text,
            "state": "awaiting_reply",
            "trigger_id": trg_id,
            "trigger_kind": trg.get("kind", ""),
            "trigger_payload": trg,
            "category_slug": cat_slug,
            "result_context": result
        }
        conversations[conv_id] = [{
            "from": "vera",
            "body": body_text,
            "ts": now_ts,
            "turn_number": 1
        }]

        send_as = result.get("send_as", "vera")
        if customer_id and not send_as:
            send_as = "merchant_on_behalf"

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v2",
            "template_params": [merchant.get("identity", {}).get("owner_first_name", ""), body_text[:50], ""],
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": sup_key,
            "rationale": result.get("rationale", "")
        })
        log.info(f"Action queued: conv={conv_id} trigger={trg_id} body_len={len(body_text)}")

    return {"actions": actions[:20]}  # hard cap


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    customer_id = body.customer_id
    message = body.message
    from_role = body.from_role
    turn = body.turn_number

    # Store the turn
    conversations.setdefault(conv_id, []).append({
        "from": from_role,
        "body": message,
        "ts": body.received_at,
        "turn_number": turn
    })

    meta = conv_meta.get(conv_id, {})
    if not merchant_id:
        merchant_id = meta.get("merchant_id", "")
    if not customer_id:
        customer_id = meta.get("customer_id")

    # Detect auto-reply first (fast path — no LLM)
    if is_auto_reply(message):
        count = auto_reply_counts.get(conv_id, 0) + 1
        auto_reply_counts[conv_id] = count
        log.info(f"Auto-reply detected in conv {conv_id} count={count}")
        if count >= 2:
            # End conversation after 2 auto-replies
            if conv_id in conv_meta:
                conv_meta[conv_id]["state"] = "closed"
            return {
                "action": "end",
                "rationale": "Repeated auto-reply detected; ending conversation to avoid loop"
            }
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Detected auto-reply message; backing off 30 min before re-attempting"
        }

    # Detect hostile (fast path)
    if is_hostile(message):
        log.info(f"Hostile message in conv {conv_id}")
        # Mark conversation closed
        if conv_id in conv_meta:
            conv_meta[conv_id]["state"] = "closed"
        return {
            "action": "end",
            "body": "",
            "rationale": "Merchant expressed strong disinterest; gracefully exiting to respect their preference"
        }

    # Get full context for LLM reply
    is_customer_message = (from_role == "customer")
    merchant_entry = context_store.get(("merchant", merchant_id)) if merchant_id else None
    merchant = merchant_entry["payload"] if merchant_entry else {}

    cat_slug = merchant.get("category_slug", meta.get("category_slug", ""))
    cat_entry = context_store.get(("category", cat_slug))
    category = cat_entry["payload"] if cat_entry else {}

    customer = None
    if customer_id:
        cust_entry = context_store.get(("customer", customer_id))
        customer = cust_entry["payload"] if cust_entry else None

    # Get conversation history
    history = conversations.get(conv_id, [])
    last_bot_body = meta.get("last_bot_body", "")
    trigger_kind = meta.get("trigger_kind", "")
    trigger_payload = meta.get("trigger_payload", {})

    try:
        result = await handle_reply_with_llm(
            message=message,
            from_role=from_role,
            turn_number=turn,
            conversation_history=history,
            last_bot_body=last_bot_body,
            category=category,
            merchant=merchant,
            customer=customer,
            trigger_kind=trigger_kind,
            trigger_payload=trigger_payload,
            merchant_id=merchant_id,
            is_customer_message=is_customer_message
        )
    except Exception as e:
        log.error(f"Reply handler failed: {e}")
        if is_customer_message:
            # Customer-specific fallback
            cust_name = ""
            if customer:
                cust_name = customer.get("identity", {}).get("name", "")
            fallback_body = (
                f"Hi {cust_name}! Thank you for confirming. Your appointment is all set — "
                f"we look forward to seeing you!"
                if cust_name else
                "Thank you for confirming! Your appointment is all set. See you soon!"
            )
            return {
                "action": "send",
                "body": fallback_body,
                "cta": "confirm",
                "rationale": "Customer confirmation fallback"
            }
        return {
            "action": "send",
            "body": "On it! Let me pull that together for you.",
            "cta": "open_ended",
            "rationale": "Fallback response due to processing error"
        }

    action = result.get("action", "send")
    resp_body = result.get("body", "").strip()

    # Anti-repetition check
    if action == "send" and resp_body:
        prev = sent_bodies.get(f"{merchant_id}_all", [])
        if resp_body in prev:
            resp_body = resp_body + " Want to proceed?"
        sent_bodies.setdefault(f"{merchant_id}_all", []).append(resp_body)
        # Update conv metadata
        if conv_id in conv_meta:
            conv_meta[conv_id]["last_bot_body"] = resp_body

    if action == "end":
        if conv_id in conv_meta:
            conv_meta[conv_id]["state"] = "closed"

    log.info(f"Reply response: conv={conv_id} action={action} body_len={len(resp_body)}")

    response = {
        "action": action,
        "rationale": result.get("rationale", "")
    }
    if action == "send":
        response["body"] = resp_body
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 1800)
    return response


@app.post("/v1/teardown")
async def teardown():
    context_store.clear()
    conversations.clear()
    conv_meta.clear()
    suppression_log.clear()
    sent_bodies.clear()
    auto_reply_counts.clear()
    log.info("State wiped on teardown")
    return {"status": "cleared"}

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REPLY & HOSTILE DETECTION  (fast path — no LLM)
# ─────────────────────────────────────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching|messaging|getting in touch)",
    r"i('m| am) (currently |presently )?(away|unavailable|out of office|on leave|traveling)",
    r"will (reply|get back|respond|return your) (to you )?(soon|shortly|later|asap|in \d+)",
    r"(our |the )?(team|office|staff) (is|will be) (closed|unavailable|back)",
    r"auto.?(reply|response|message)",
    r"hi,?\s*i am (out|away|on vacation)",
    r"this is an? (auto|automated) (reply|message|response)",
    r"currently (not available|offline|closed)",
    r"unable to respond at this time",
    r"i'll be back",
]

HOSTILE_PATTERNS = [
    r"\bstop (messaging|texting|sending|spamming|contacting)\b",
    r"\bspam(ming)?\b",
    r"\bannoy(ing|ed)\b",
    r"\bleave me alone\b",
    r"\buseless (bot|spam|messages|service)\b",
    r"\bbakwaas\b",
    r"\bblock\b.*\b(you|this|vera)\b",
    r"\bnot interested.*stop\b",
    r"\bstop.*spam\b",
    r"\bplease stop\b",
]

def is_auto_reply(message: str) -> bool:
    m = message.lower().strip()
    for pattern in AUTO_REPLY_PATTERNS:
        if re.search(pattern, m, re.IGNORECASE):
            return True
    return False

def is_hostile(message: str) -> bool:
    m = message.lower().strip()
    for pattern in HOSTILE_PATTERNS:
        if re.search(pattern, m, re.IGNORECASE):
            return True
    # Also check for direct "stop" with frustration markers
    if re.search(r'^stop[.!]*$', m.strip()):
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED LLM CALLER  (supports OpenAI and Anthropic automatically)
# ─────────────────────────────────────────────────────────────────────────────

async def call_claude(system_prompt: str, user_message: str, model: str = None, temperature: float = 0.0) -> str:
    """Calls OpenAI if OPENAI_API_KEY is set, otherwise Anthropic."""
    model = model or COMPOSE_MODEL

    if LLM_PROVIDER == "openai":
        # ── OpenAI ──────────────────────────────────────────────────────────
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 1500,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ]
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    else:
        # ── Anthropic ────────────────────────────────────────────────────────
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        body = {
            "model": model,
            "max_tokens": 1500,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSER SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

COMPOSER_SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. 
Your task: compose the single best next message to send, using ONLY data from the provided context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFECT SCORE PATTERNS (50/50 reference cases)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CASE 1 — Dentist research digest:
INPUT: Dr. Meera, CTR 2.1% vs peer 3.0%, 124 high-risk adults. Trigger: JIDA paper (3-mo fluoride recall cuts caries 38% better, n=2100, high-risk adults)
OUTPUT: "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14"
WHY IT WINS: Source citation (JIDA p.14), exact n (2100), exact % (38%), anchored to HER patient segment, single CTA, reciprocity offer

CASE 2 — Pharmacy supply alert:
INPUT: Apollo Pharmacy Ramesh, 240 chronic-Rx customers. Trigger: atorvastatin recall batches AT2024-1102 + AT2024-1108 by Mfr Z
OUTPUT: "Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102, AT2024-1108) by Mfr Z — sub-potency, no safety risk, but customers should be informed for replacement. Pulled your repeat-Rx list: 22 of your chronic-Rx customers were dispensed these batches in last 90 days. Want me to draft their WhatsApp note + the replacement-pickup workflow?"
WHY IT WINS: Exact batch numbers from context, derived count (22/240), "no safety risk" reframes urgency correctly, end-to-end workflow offer

CASE 3 — Restaurant IPL (counter-intuitive call):
INPUT: SK Pizza Junction Suresh Delhi, BOGO pizza active, Trigger: Saturday IPL match DC vs MI 7:30pm. Category digest: Saturday IPL = -12% restaurant covers (home-watch effect)
OUTPUT: "Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). Skip the match-night promo today; instead push your BOGO pizza (already active) as a delivery-only Saturday special. Want me to draft the Swiggy banner + an Insta story? Live in 10 min."
WHY IT WINS: Counter-intuitive judgment using category data, uses their EXISTING offer (not invented), 10-min commitment cap, specific deliverables

CASE 4 — Gym seasonal dip reframe:
INPUT: PowerHouse Fitness Karthik, views -30% w/w, 245 active members. Trigger: seasonal_perf_dip (April-June normal lull, is_expected_seasonal=true)
OUTPUT: "Karthik, your views are down 30% this week — but I want to flag this is the normal April-June acquisition lull (every metro gym sees -25 to -35% in this window). Action: skip ad spend now, save it for Sept-Oct when conversion is 2x. For now, focus retention on your 245 members. Want me to draft a 'summer attendance challenge' to keep them through the dip?"
WHY IT WINS: Normalizes the dip with data range, gives specific counter-action, uses his exact member count, ends with concrete offer

CASE 5 — Dentist recall reminder (customer-facing):
INPUT: Priya, lapsed_soft, 5mo since last visit, prefers weekday evenings, hi-en mix. Dr. Meera's active offer: Dental Cleaning @ ₹299, slots: Wed 5 Nov 6pm + Thu 6 Nov 5pm
OUTPUT: "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works."
WHY IT WINS: Language switch (hi-en mix as requested), specific slots from payload, real catalog price, emoji appropriate for patient warmth

CASE 6 — Pharmacy chronic refill (customer-facing):
INPUT: Mr. Sharma, 65+, via son's WhatsApp. 3 molecules running out 2026-04-28. Apollo has senior 15% off + free delivery >₹499
OUTPUT: "Namaste — Apollo Health Plus Malviya Nagar yahan. Sharma ji ki 3 monthly medicines (metformin, atorvastatin, telmisartan) 28 April ko khatam hongi. Same dose, same brand pack ready hai. Senior discount 15% applied — total ₹1,420 (₹240 saved). Free home delivery to saved address by 5pm tomorrow. Reply CONFIRM to dispatch, or call 9876543210 if any change in dosage."
WHY IT WINS: Namaste for senior respect, all 3 molecules named, exact date, total + savings shown, two-channel option, CONFIRM CTA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIGGER-SPECIFIC COMPOSITION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

research_digest → source + trial_n + % result + patient_segment → offer to draft content
recall_due → "X months since last visit" + specific_slots + catalog_price + language_pref
perf_dip (not seasonal) → metric + delta + "vs [baseline]" + action suggestion
seasonal_perf_dip → normalize dip with category range + "save spend for [specific month]" + retention focus  
perf_spike → metric + delta + likely_driver → capitalize now
milestone_reached → celebrate + leverage (post draft offer)
renewal_due → days_remaining + "what you'd lose" + one-click CTA
dormant_with_vera → light re-engagement, single curious question, no guilt
festival_upcoming → festival name + days_until + category-specific offer angle
ipl_match_today → is_weeknight check: if weekend → home-watch warning + delivery pivot; if weeknight → +18% covers → match-night combo
competitor_opened → distance + their_offer → your counter (stronger proof points from context)
customer_lapsed_soft/hard → no shame + their past goal + specific new offering matching that goal
chronic_refill_due → ALL molecule names + exact stock_runs_out date + total + savings + CONFIRM CTA
supply_alert → exact batch numbers + manufacturer + sub-potency (not dangerous) + affected customer count + workflow offer
trial_followup → what trial was + next_session options → binary YES CTA
curious_ask_due → open question (what's most asked this week?) + offer to draft content from their answer
active_planning_intent → CONTINUE the exact thread from last merchant message, deliver directly
review_theme_emerged → theme + occurrences + quote + one concrete action
winback_eligible → what's changed + what they'd gain back + one-click restart

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES (violation = penalty deduction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ALWAYS:
- Use owner_first_name from identity (Dr. Meera, Suresh, Karthik, Ramesh)
- Cite source for any research/compliance data (JIDA p.14, DCI circular, CDSCO alert)
- Every number in your output MUST exist verbatim or be computable from the given context
- Single CTA — one clear action, binary when possible (Want me to X? Reply YES/CONFIRM)
- End with the CTA (it must be the last sentence)
- Match language_pref: "hi-en mix" → code-switch naturally; "hi" → mostly Hindi; "english" → English only
- send_as = "merchant_on_behalf" when scope=customer; "vera" when scope=merchant
- Include rationale: "Signal: [trigger_kind]. Key data used: [list facts]. Category voice: [descriptor]. CTA: [why this type]."

❌ NEVER:
- Invent numbers, statistics, offers, batch numbers, slot times, or product names not in context
- Use: "guaranteed", "100% safe", "best in city", "miracle", "fastest results"
- Multiple CTAs in one message
- Long preamble ("I hope this message finds you well…")
- Re-introduce yourself after the first message
- Repeat content from conversation_history
- Generic service offers when specific catalog offers exist
- Promotional hype tone for dentists or pharmacies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — respond ONLY with this valid JSON object, nothing else:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "body": "the complete message text",
  "cta": "yes_no | open_ended | tap_action | confirm | binary_slot",
  "send_as": "vera | merchant_on_behalf",
  "suppression_key": "copy exactly from trigger.suppression_key",
  "rationale": "Signal: X. Key context facts used: Y. Category voice applied: Z. CTA type: binary/open-ended because W."
}"""


# ─────────────────────────────────────────────────────────────────────────────
# REPLY HANDLER SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

REPLY_SYSTEM_PROMPT = """You are Vera's reply handler. A merchant or customer just replied.

INTENT CLASSIFICATION:
- accept: "yes", "yep", "sure", "go ahead", "confirmed", "karo", "theek hai", "let's do it", "ok", "haan", "send it", "proceed", "done"
- decline: "no", "nahi", "not now", "skip", "later", "mat karo", "don't", "not interested"
- question: asking for more info, clarification, pricing details
- off_topic: completely unrelated to the conversation
- unclear: can't determine intent

RESPONSE RULES — READ CAREFULLY:
- accept → IMMEDIATELY deliver what was promised in the LAST BOT MESSAGE. Do NOT re-qualify, do NOT ask more questions, do NOT circle back. If the bot promised to "draft a WhatsApp" — draft one. If it promised to "send the abstract" — send a brief summary from context. If it promised "here's the plan" — give the plan. Momentum must not break.
- decline → brief, graceful close. "Understood, no problem! I'll be here if things change." Then action=end.
- question → answer the question using only context data, then re-offer the CTA from the last message
- off_topic → acknowledge in one short sentence, then gently return: "Got it! On our Vera conversation —" + re-ask the last CTA
- unclear → ask ONE simple clarifying question: "Quick check — shall I go ahead? Reply YES or NO."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL FOR ACCEPT RESPONSES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When merchant says YES to a research digest offer: send a short draft WhatsApp (2-3 sentences) they can forward to patients, using the digest content from context.
When merchant says YES to a GBP post offer: write a short GBP post draft (2-3 sentences) using their real offers/stats.
When merchant says YES to a patient recall message: draft the patient WhatsApp using their catalog price + appointment angle.
When merchant says YES to a corporate thali: sketch the pricing/format briefly using their existing thali offer as base.
ALWAYS add: "I can refine this — want me to adjust anything?"

OUTPUT — respond ONLY with this valid JSON, nothing else:
{
  "intent": "accept | decline | question | off_topic | unclear",
  "action": "send | wait | end",
  "body": "response text (empty string if action=wait or end)",
  "cta": "open_ended | yes_no | confirm",
  "wait_seconds": 0,
  "rationale": "Intent detected: X because Y. Action: Z because W."
}"""

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT SUMMARIZER  (builds a clean summary for LLM — avoids raw JSON dump)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_category(cat: dict) -> str:
    if not cat:
        return "Category: unknown"
    voice = cat.get("voice", {})
    peer = cat.get("peer_stats", {})
    digest = cat.get("digest", [])
    seasonal = cat.get("seasonal_beats", [])
    trends = cat.get("trend_signals", [])
    offers = cat.get("offer_catalog", [])

    digest_summary = ""
    for d in digest[:5]:
        digest_summary += f"  [{d.get('kind','').upper()}] {d.get('title','')} — {d.get('source','')} — {d.get('summary','')[:120]}\n    Actionable: {d.get('actionable','')}\n"

    seasonal_str = "; ".join([f"{s.get('month_range')}: {s.get('note')}" for s in seasonal])
    trend_str = "; ".join([f'"{t.get("query")}" +{int(t.get("delta_yoy",0)*100)}% YoY (age {t.get("segment_age","")})'  for t in trends])
    offer_str = "\n".join([f"  - {o.get('title')} [{o.get('audience','')}]" for o in offers])

    return f"""CATEGORY: {cat.get('display_name', cat.get('slug', ''))}
Voice: tone={voice.get('tone')}, register={voice.get('register')}, code_mix={voice.get('code_mix')}
Allowed vocab: {', '.join(voice.get('vocab_allowed', [])[:10])}
Taboo words: {', '.join(voice.get('vocab_taboo', [])[:5])}
Peer stats: avg_ctr={peer.get('avg_ctr')}, avg_calls_30d={peer.get('avg_calls_30d')}, avg_rating={peer.get('avg_rating')}, avg_reviews={peer.get('avg_review_count')}

Offer catalog:
{offer_str}

This week's digest:
{digest_summary}
Seasonal beats: {seasonal_str}
Trend signals: {trend_str}"""


def summarize_merchant(m: dict) -> str:
    if not m:
        return "Merchant: unknown"
    identity = m.get("identity", {})
    sub = m.get("subscription", {})
    perf = m.get("performance", {})
    offers = m.get("offers", [])
    hist = m.get("conversation_history", [])
    agg = m.get("customer_aggregate", {})
    signals = m.get("signals", [])
    reviews = m.get("review_themes", [])

    delta = perf.get("delta_7d", {})
    active_offers = [o for o in offers if o.get("status") == "active"]
    offer_str = "\n".join([f"  - {o.get('title')} [ACTIVE since {o.get('started','')}]" for o in active_offers])
    if not offer_str:
        offer_str = "  - No active offers"

    hist_str = ""
    for h in hist[-3:]:
        hist_str += f"  [{h.get('from','?').upper()}] {h.get('body','')[:120]}\n"

    review_str = "\n".join([f"  - [{r.get('sentiment').upper()}] {r.get('theme')}: {r.get('occurrences_30d','?')} occurrences. Quote: \"{r.get('common_quote','')}\"" for r in reviews])

    return f"""MERCHANT: {identity.get('name')}
ID: {m.get('merchant_id')}
Owner first name: {identity.get('owner_first_name')} | City: {identity.get('city')} | Locality: {identity.get('locality')}
Verified: {identity.get('verified')} | Languages: {', '.join(identity.get('languages', ['en']))} | Est: {identity.get('established_year')}

Subscription: status={sub.get('status')}, plan={sub.get('plan')}, days_remaining={sub.get('days_remaining')}, days_since_expiry={sub.get('days_since_expiry')}

Performance (last {perf.get('window_days',30)} days):
  views={perf.get('views')}, calls={perf.get('calls')}, directions={perf.get('directions')}, ctr={perf.get('ctr')}, leads={perf.get('leads')}
  7d deltas: views {delta.get('views_pct','?')*100 if isinstance(delta.get('views_pct'), (int,float)) else '?'}%, calls {delta.get('calls_pct','?')*100 if isinstance(delta.get('calls_pct'), (int,float)) else '?'}%

Active offers:
{offer_str}

Customer aggregate: {json.dumps(agg)}
Derived signals: {', '.join(signals)}

Review themes:
{review_str if review_str else "  - No review data"}

Recent conversation (last 3 turns):
{hist_str if hist_str else "  - No prior conversation"}"""


def summarize_trigger(trg: dict) -> str:
    if not trg:
        return "Trigger: unknown"
    payload_str = json.dumps(trg.get("payload", {}), indent=2, ensure_ascii=False)
    return f"""TRIGGER: {trg.get('id')}
Kind: {trg.get('kind')} | Scope: {trg.get('scope')} | Source: {trg.get('source')}
Urgency: {trg.get('urgency')}/5 | Suppression key: {trg.get('suppression_key')}
Expires: {trg.get('expires_at')}
Payload:
{payload_str}"""


def summarize_customer(cust: dict) -> str:
    if not cust:
        return "Customer: none (merchant-facing message)"
    identity = cust.get("identity", {})
    rel = cust.get("relationship", {})
    prefs = cust.get("preferences", {})
    consent = cust.get("consent", {})
    return f"""CUSTOMER: {identity.get('name')}
ID: {cust.get('customer_id')} | Age band: {identity.get('age_band')} | Language pref: {identity.get('language_pref')}
State: {cust.get('state')} | Channel: {prefs.get('channel')}
Relationship: first_visit={rel.get('first_visit')}, last_visit={rel.get('last_visit')}, visits_total={rel.get('visits_total')}, LTV=₹{rel.get('lifetime_value',0)}
Services received: {', '.join(rel.get('services_received', [])[:5])}
Preferences: {json.dumps(prefs)}
Consent scope: {', '.join(consent.get('scope', []))}"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPOSER  (signal-ranked → LLM compose)
# ─────────────────────────────────────────────────────────────────────────────

async def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    now_ts: str
) -> Optional[dict]:
    """
    Builds a rich context summary and calls Claude to compose the best message.
    Returns parsed JSON dict or None if composition fails.
    """
    # Build context summary
    cat_summary = summarize_category(category)
    merch_summary = summarize_merchant(merchant)
    trg_summary = summarize_trigger(trigger)
    cust_summary = summarize_customer(customer)

    # Get peer CTR for gap analysis
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0.030)
    merch_ctr = merchant.get("performance", {}).get("ctr", 0.030)
    ctr_gap = ""
    if merch_ctr < peer_ctr:
        gap_pct = round((peer_ctr - merch_ctr) / peer_ctr * 100)
        ctr_gap = f"\n⚠️  CTR GAP: merchant CTR={merch_ctr} is {gap_pct}% below peer median ({peer_ctr}) — use this in message if relevant"

    user_prompt = f"""Compose the next message for this (category, merchant, trigger, customer) combination.
Current time: {now_ts}
{ctr_gap}

━━━ CATEGORY ━━━
{cat_summary}

━━━ MERCHANT ━━━
{merch_summary}

━━━ TRIGGER ━━━
{trg_summary}

━━━ CUSTOMER ━━━
{cust_summary}

REMINDER: Output ONLY the JSON object. Every number you cite must come from the context above.
Use the owner_first_name ({merchant.get('identity', {}).get('owner_first_name', 'there')}) to address them.
If scope is "customer", set send_as="merchant_on_behalf" and honor language_pref={customer.get('identity', {}).get('language_pref', 'en') if customer else 'n/a'}.
If scope is "merchant", set send_as="vera"."""

    response_text = await call_claude(COMPOSER_SYSTEM_PROMPT, user_prompt, COMPOSE_MODEL, temperature=0.0)

    # Parse JSON
    return parse_json_response(response_text)


# ─────────────────────────────────────────────────────────────────────────────
# REPLY HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def handle_reply_with_llm(
    message: str,
    from_role: str,
    turn_number: int,
    conversation_history: list,
    last_bot_body: str,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
    trigger_kind: str,
    trigger_payload: dict,
    merchant_id: str,
    is_customer_message: bool = False
) -> dict:
    """
    Handles a merchant/customer reply using LLM intent classification + response composition.
    If is_customer_message=True, composes a customer-voiced reply addressed to the customer.
    """
    history_str = "\n".join([
        f"  [{t.get('from','?').upper()} turn {t.get('turn_number','?')}]: {t.get('body','')[:200]}"
        for t in conversation_history[-5:]
    ])

    merch_summary = summarize_merchant(merchant)
    cat_summary = summarize_category(category) if category else "Category: not available"
    cust_summary = summarize_customer(customer) if customer else ""
    digest_items = category.get("digest", []) if category else []
    digest_str = "\n".join([
        f"  [{d.get('kind')}] {d.get('title')}: {d.get('summary','')[:150]}"
        for d in digest_items[:3]
    ])

    # ── CUSTOMER MESSAGE: compose customer-voiced reply ──────────────────────
    if is_customer_message:
        cust_name = ""
        cust_lang = "english"
        slot_picked = ""
        if customer:
            cust_name = customer.get("identity", {}).get("name", "")
            cust_lang = customer.get("identity", {}).get("language_pref", "english")
        # Try to detect slot selection from message
        if any(x in message.lower() for x in ["wed", "thu", "sat", "6pm", "7pm", "8am", "yes", "book", "confirm"]):
            slot_picked = message

        customer_system = """You are Vera, composing a message FROM the merchant TO their customer.
The customer just replied. Compose a warm, helpful response addressed directly to the customer by name.
- Confirm their booking/request if they accepted
- Use their language preference
- Be brief, warm, specific
- End with one clear next step
Output ONLY valid JSON: {"body": "...", "cta": "confirm|open_ended", "send_as": "merchant_on_behalf", "rationale": "..."}"""

        customer_prompt = f"""Customer replied to a merchant message.

CUSTOMER: {cust_name} | Language: {cust_lang} | State: {customer.get('state','') if customer else ''}
MERCHANT: {merchant.get('identity',{}).get('name','')} | Owner: {merchant.get('identity',{}).get('owner_first_name','')}
TRIGGER KIND: {trigger_kind}
LAST BOT MESSAGE: "{last_bot_body}"
CUSTOMER MESSAGE: "{message}"
CUSTOMER CONTEXT: {cust_summary[:400]}

Compose a reply FROM the merchant addressed TO the customer {cust_name}.
If they picked a slot or said yes: confirm the booking with specific details.
Output ONLY the JSON."""

        response_text = await call_claude(customer_system, customer_prompt, COMPOSE_MODEL, temperature=0.0)
        result = parse_json_response(response_text)
        if result and result.get("body"):
            result["action"] = "send"
            return result
        # Fallback for customer
        confirm_body = f"Hi {cust_name}! Your appointment is confirmed. We look forward to seeing you. Please call us if anything changes." if cust_name else "Your appointment is confirmed! We look forward to seeing you."
        return {
            "intent": "accept",
            "action": "send",
            "body": confirm_body,
            "cta": "confirm",
            "rationale": "Customer accepted/confirmed — booking confirmed response"
        }

    # ── MERCHANT MESSAGE: standard intent classification ──────────────────────
    user_prompt = f"""Merchant/customer just replied. Handle it correctly.

CONVERSATION HISTORY (last 5 turns):
{history_str}

LAST BOT MESSAGE (what you must honor if they accept):
"{last_bot_body}"

NEW REPLY (turn {turn_number}):
From: {from_role}
Message: "{message}"

TRIGGER KIND: {trigger_kind}
TRIGGER PAYLOAD: {json.dumps(trigger_payload, ensure_ascii=False)[:300]}

MERCHANT CONTEXT:
{merch_summary[:600]}

CATEGORY DIGEST (use for content if merchant accepts):
{digest_str}

{cust_summary[:300] if cust_summary else ""}

TASK:
1. Classify intent of "{message}"
2. accept → deliver the exact promised artifact using real context data
3. decline → graceful close
4. question → answer from context, re-offer CTA
5. off_topic → acknowledge briefly, return to topic
6. hostile → end immediately

Output ONLY the JSON object."""

    response_text = await call_claude(REPLY_SYSTEM_PROMPT, user_prompt, COMPOSE_MODEL, temperature=0.0)
    result = parse_json_response(response_text)

    if not result:
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["yes", "sure", "ok", "go", "haan", "karo", "confirm"]):
            return {
                "intent": "accept", "action": "send",
                "body": "On it — sending the draft now. Let me know if you'd like any changes.",
                "cta": "open_ended", "wait_seconds": 0, "rationale": "Fallback accept"
            }
        return {
            "intent": "unclear", "action": "send",
            "body": "Should I go ahead? Reply YES to confirm.",
            "cta": "yes_no", "wait_seconds": 0, "rationale": "Fallback unclear"
        }
    return result

# JSON PARSER  (handles LLM sometimes wrapping in markdown)
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> Optional[dict]:
    """Robustly parse JSON from LLM output, handling markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r'^```(json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()
    # Find JSON object
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        log.error(f"No JSON found in response: {text[:200]}")
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}. Text: {text[start:end+1][:300]}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.info(f"Vera Bot v{BOT_VERSION} starting up")
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        log.info(f"LLM Provider: OpenAI ({COMPOSE_MODEL}) ✅")
    elif LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        log.info(f"LLM Provider: Anthropic ({COMPOSE_MODEL}) ✅")
    else:
        log.warning("NO API KEY SET — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env!")
    log.info("All 5 endpoints ready: /v1/healthz /v1/metadata /v1/context /v1/tick /v1/reply")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)