"""
TECHY AI v3 — Multimodal Smart Assistant
Features: Image Gen • Web Search • YouTube • Video Analysis • TikTok (via Apify)
"""

import io, os, json, time, base64, re
from datetime import datetime
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TECHY AI", page_icon="⚡", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, header, footer, .stDeployButton {visibility:hidden;}
.stApp {background:#0E1117; color:#E2E8F0;}
.stChatInput input {background:#1F2937!important; color:#FFF!important; border-radius:14px!important; border:1px solid #374151!important;}
div[data-testid="stChatMessage-user"] {background:#1F2937; border-radius:16px; padding:14px; margin-bottom:12px;}
div[data-testid="stChatMessage-assistant"] {background:#111827; border-radius:16px; padding:14px; margin-bottom:12px; border:1px solid #1F2937;}
p, li, code {color:#E5E7EB!important; font-size:15px;}
code {background:#0B1220!important; padding:2px 6px; border-radius:6px;}
pre {background:#0B1220!important; border-radius:10px; padding:12px;}
.techy-header {display:flex; align-items:center; gap:10px; padding:14px 18px; margin-bottom:18px; background:linear-gradient(90deg,#1E3A8A,#0F172A); border-radius:16px; border:1px solid #1F2937;}
.techy-header h1 {margin:0; font-size:22px; color:#F9FAFB;}
.techy-header span {color:#93C5FD; font-size:13px;}
section[data-testid="stSidebar"] {background:#0B1220;}
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = "openrouter/auto"
MAX_HISTORY = 30
MAX_PDF_CHARS = 150_000

MODEL_OPTIONS = {
    "Auto (best free)": "openrouter/auto",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Gemini 2.0 Flash (video)": "google/gemini-2.0-flash-001",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "DeepSeek Chat": "deepseek/deepseek-chat",
}

IMAGE_MODEL = "bytedance-seed/seedream-4.5"  # OpenRouter image model [citation:2]

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def get_setting(name):
    try: return st.secrets[name]
    except Exception: pass
    return os.environ.get(name)

API_KEY = get_setting("OPENROUTER_API_KEY")
YOUTUBE_KEY = get_setting("YOUTUBE_API_KEY")
APIFY_TOKEN = get_setting("APIFY_TOKEN")  # optional, for TikTok

if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY"); st.stop()

@st.cache_resource
def get_client(key): return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

client = get_client(API_KEY)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

TECHY_PROMPT = """
You are TECHY, a smart, friendly AI assistant with access to tools.
You can generate images, search the web, search YouTube, and analyze videos.

RULES:
- If the user asks for a picture/image, say you'll generate it and use the image tool.
- If the user asks about current events, recent info, or "search for", use web search.
- If the user asks about a YouTube video or channel, use YouTube search.
- If the user provides a video link or asks to analyze a video, use video analysis.
- If the user asks about a TikTok person, explain you'll try to look them up (may need Apify).
- Cite sources with markdown links when using web search.
- Be honest: if a tool fails, say so clearly.
- Never fabricate facts, sources, or video content.
""".strip()

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def generate_image(prompt: str):
    """Generate an image via OpenRouter Image API [citation:2]"""
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/images",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": IMAGE_MODEL, "prompt": prompt},
            timeout=120,
        )
        if not r.ok:
            return None, f"Image API error: {r.status_code} — {r.text[:200]}"
        data = r.json().get("data", [])
        if not data or not data[0].get("b64_json"):
            return None, "No image data returned."
        img_bytes = base64.b64decode(data[0]["b64_json"])
        return img_bytes, None
    except Exception as e:
        return None, str(e)


def web_search(query: str):
    """Use OpenRouter web plugin for grounded search [citation:8]"""
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "openrouter/auto",
                "messages": [{"role": "user", "content": query}],
                "plugins": [{"id": "web"}],
            },
            timeout=60,
        )
        if not r.ok:
            return None, f"Search error: {r.status_code} — {r.text[:200]}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)


def youtube_search(query: str, max_results=5):
    """Search YouTube via Data API v3 [citation:5]"""
    if not YOUTUBE_KEY:
        return None, "YOUTUBE_API_KEY not set in secrets."
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet", "q": query, "type": "video",
                "maxResults": max_results, "key": YOUTUBE_KEY,
            },
            timeout=15,
        )
        if not r.ok:
            return None, f"YouTube error: {r.status_code} — {r.text[:200]}"
        items = r.json().get("items", [])
        lines = [f"**YouTube results for:** `{query}`\n"]
        for it in items:
            sn = it["snippet"]
            vid = it["id"]["videoId"]
            lines.append(
                f"- **{sn['title']}** — {sn['channelTitle']}  \n"
                f"  https://youtube.com/watch?v={vid}  \n"
                f"  _{sn.get('description','')[:140]}..._"
            )
        return "\n".join(lines), None
    except Exception as e:
        return None, str(e)


def analyze_video(video_url: str, question: str):
    """Analyze a YouTube video or video URL via OpenRouter multimodal [citation:7]"""
    try:
        # OpenRouter video input: use content array with input_video
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "google/gemini-2.0-flash-001",  # video-capable [citation:7]
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }],
            },
            timeout=120,
        )
        if not r.ok:
            return None, f"Video error: {r.status_code} — {r.text[:300]}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)


def tiktok_lookup(username: str):
    """TikTok lookup via Apify (optional, paid) [citation:14]"""
    if not APIFY_TOKEN:
        return None, ("TikTok lookup requires an APIFY_TOKEN. "
                      "TikTok has no free official search API. Add APIFY_TOKEN to secrets.")
    try:
        # Apify actor run (simplified — real actor ID may vary)
        r = requests.post(
            f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs",
            params={"token": APIFY_TOKEN},
            json={"profiles": [username], "resultsPerPage": 5},
            timeout=60,
        )
        if not r.ok:
            return None, f"Apify error: {r.status_code} — {r.text[:200]}"
        run_id = r.json()["data"]["id"]
        # Poll briefly
        for _ in range(10):
            time.sleep(3)
            s = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                params={"token": APIFY_TOKEN}, timeout=15,
            ).json()
            if s["data"]["status"] == "SUCCEEDED":
                dataset_id = s["data"]["defaultDatasetId"]
                items = requests.get(
                    f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                    params={"token": APIFY_TOKEN}, timeout=15,
                ).json()
                if not items:
                    return f"No TikTok data found for `{username}`.", None
                top = items[0]
                return (
                    f"**TikTok: @{top.get('authorMeta',{}).get('name', username)}**\n"
                    f"- Followers: {top.get('authorMeta',{}).get('fans','?')}\n"
                    f"- Bio: {top.get('authorMeta',{}).get('signature','')[:150]}\n"
                    f"- Top video: {top.get('text','')[:120]}\n"
                    f"- URL: {top.get('webVideoUrl','')}"
                ), None
        return None, "Apify run timed out."
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------------------------
# Router — decides which tool to use
# ---------------------------------------------------------------------------

def route_and_execute(user_input: str, chat_history):
    """Detect intent and call the right tool. Returns (tool_name, result, error)."""
    text = user_input.lower()

    # Image generation
    if any(k in text for k in ["generate image", "draw", "create picture", "make an image", "picture of"]):
        prompt = re.sub(r"(generate|create|make|draw).*?(image|picture|of)", "", user_input, flags=re.I).strip()
        prompt = prompt or user_input
        img, err = generate_image(prompt)
        return ("image", img, err)

    # YouTube search
    if "youtube" in text or ("search" in text and "video" in text):
        q = re.sub(r"(search|find|look up).*?(youtube|for)", "", user_input, flags=re.I).strip()
        q = q or user_input
        res, err = youtube_search(q)
        return ("youtube", res, err)

    # TikTok
    if "tiktok" in text:
        match = re.search(r"@?([A-Za-z0-9_.]+)", user_input.split("tiktok")[-1])
        user = match.group(1) if match else ""
        res, err = tiktok_lookup(user)
        return ("tiktok", res, err)

    # Video analysis (if a video URL is present)
    video_url = None
    for pattern in [r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)",
                    r"(https?://youtu\.be/[\w-]+)"]:
        m = re.search(pattern, user_input)
        if m: video_url = m.group(1); break
    if video_url:
        q = user_input.replace(video_url, "").strip() or "Summarize this video."
        res, err = analyze_video(video_url, q)
        return ("video", res, err)

    # Web search
    if any(k in text for k in ["search", "latest", "news", "current", "who is", "what is happening"]):
        res, err = web_search(user_input)
        return ("web", res, err)

    return (None, None, None)  # no tool needed → use LLM directly

# ---------------------------------------------------------------------------
# Stream LLM reply (no tool)
# ---------------------------------------------------------------------------

def stream_reply(messages, model, temperature, max_tokens):
    formatted = [{"role": "system", "content": TECHY_PROMPT}]
    for m in messages[-MAX_HISTORY:]:
        formatted.append({"role": m["role"], "content": m["content"]})
    stream = client.chat.completions.create(
        model=model, messages=formatted, stream=True,
        temperature=temperature, max_tokens=max_tokens,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

for k, v in [("messages", []), ("pdf_text", ""), ("pdf_name", None), ("last_usage", None)]:
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ TECHY Controls")
    st.caption("Multimodal smart assistant")

    st.markdown("### 🧠 Model (chat)")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    selected_model = MODEL_OPTIONS[model_label]

    st.markdown("### 🎛️ Tuning")
    temperature = st.slider("Temperature", 0.0, 1.2, 0.6, 0.1)
    max_tokens = st.slider("Max tokens", 256, 4096, 1024, 128)

    st.markdown("---")
    st.markdown("### 📄 PDF Context")
    up = st.file_uploader("Upload PDF", type="pdf")
    if up:
        try:
            reader = PdfReader(io.BytesIO(up.getvalue()))
            pages = []
            for i, p in enumerate(reader.pages, 1):
                t = (p.extract_text() or "").strip()
                if t: pages.append(f"[Page {i}]\n{t}")
            st.session_state.pdf_text = "\n\n".join(pages)[:MAX_PDF_CHARS]
            st.session_state.pdf_name = up.name
            st.success(f"Loaded: {up.name}")
        except Exception as e:
            st.error(f"PDF error: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Actions")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_usage = None
        st.rerun()

    if st.session_state.last_usage:
        u = st.session_state.last_usage
        st.caption(f"⏱️ {u['elapsed']:.1f}s • {u['tool']}")

# ---------------------------------------------------------------------------
# Header + Chat History
# ---------------------------------------------------------------------------

st.markdown("""
<div class="techy-header">
  <div style="font-size:28px;">⚡</div>
  <div><h1>TECHY v3</h1><span>Image gen • Web • YouTube • Video • TikTok</span></div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="Generated by TECHY")

# ---------------------------------------------------------------------------
# Input & Tool Execution
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask TECHY anything — or say 'draw a cat', 'search the web for...', 'youtube search...'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    start = time.time()
    tool, result, err = route_and_execute(user_input, st.session_state.messages)

    with st.chat_message("assistant", avatar="⚡"):
        if tool == "image" and result:
            st.image(result, caption="Generated image")
            text = "Here's the image I generated 🎨"
            st.session_state.messages.append({"role": "assistant", "content": text, "image": result})
        elif tool and result:
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        elif tool and err:
            st.warning(f"Tool `{tool}` failed: {err}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Tool error: {err}"})
        else:
            # Normal LLM stream
            try:
                stream = stream_reply(st.session_state.messages, selected_model, temperature, max_tokens)
                reply = st.write_stream(stream)
                if reply and reply.strip():
                    st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"LLM error: {e}")

    st.session_state.last_usage = {"elapsed": time.time() - start, "tool": tool or "llm"}
