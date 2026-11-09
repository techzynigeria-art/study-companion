"""
TECHY AI — Clean Chat Assistant (Light + Dark auto, readable input)
"""

import io
import os
import time
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TECHY AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — fixed contrast, clean app feel
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, header, footer, .stDeployButton {visibility: hidden;}

/* App background */
.stApp {
    background-color: #0E1117;
}

/* ---------- Chat input: FIXED — visible text, proper placeholder ---------- */
.stChatInput textarea,
.stChatInput input {
    background-color: #1F2937 !important;
    color: #F9FAFB !important;
    caret-color: #60A5FA !important;
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}
.stChatInput textarea::placeholder,
.stChatInput input::placeholder {
    color: #9CA3AF !important;
    opacity: 1 !important;
}
/* Send button */
.stChatInput button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

/* ---------- Chat bubbles ---------- */
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
div[data-testid="stChatMessage-user"] {
    background-color: #1E293B;
    border: 1px solid #334155;
}
div[data-testid="stChatMessage-assistant"] {
    background-color: #111827;
    border: 1px solid #1F2937;
}

/* ---------- Text inside chat ---------- */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span {
    color: #E5E7EB !important;
    font-size: 15px;
    line-height: 1.55;
}
div[data-testid="stChatMessage"] code {
    background-color: #0B1220 !important;
    color: #93C5FD !important;
    padding: 2px 6px;
    border-radius: 6px;
}
div[data-testid="stChatMessage"] pre {
    background-color: #0B1220 !important;
    border-radius: 10px;
    padding: 12px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #0B1220;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #E5E7EB !important;
}

/* ---------- Header banner ---------- */
.techy-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; margin-bottom: 18px;
    background: linear-gradient(90deg, #1E3A8A, #0F172A);
    border-radius: 14px;
    border: 1px solid #1F2937;
}
.techy-header h1 { margin: 0; font-size: 20px; color: #F9FAFB; }
.techy-header span { color: #93C5FD; font-size: 12.5px; }

/* ---------- Buttons ---------- */
.stButton button {
    background-color: #1F2937 !important;
    color: #E5E7EB !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
}
.stButton button:hover {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openrouter/auto"
MAX_HISTORY = 30
MAX_PDF_CHARS = 120_000

MODEL_OPTIONS = {
    "Auto (recommended)": "openrouter/auto",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "DeepSeek Chat": "deepseek/deepseek-chat",
}

SYSTEM_PROMPT = """
You are TECHY, a smart, warm, and reliable AI assistant.
You help with everyday questions, coding, math, science, writing, and general knowledge.

Style:
- Talk naturally, like a helpful human mentor.
- Keep answers clear and well-structured. Use short paragraphs, bullets, or steps.
- For code: write clean modern snippets and briefly explain the key parts.
- For math/science: show steps and check your reasoning.
- Be honest when unsure. Never invent facts, sources, or links.
- Never say "As an AI language model".
""".strip()

PDF_RULES = """
DOCUMENT MODE:
The user uploaded a document. Answer using ONLY the document below.
If the answer isn't there, say so plainly.
Include page references like [Page 3] where relevant.
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_resource
def get_client(key):
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)[:MAX_PDF_CHARS]


def build_system_prompt(pdf_text):
    if not pdf_text:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\n" + PDF_RULES + \
           "\n\n--- DOCUMENT ---\n" + pdf_text + "\n--- END ---"


def stream_reply(client, system_prompt, messages, model, temperature):
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages[-MAX_HISTORY:]:
        formatted.append({"role": m["role"], "content": m["content"]})

    stream = client.chat.completions.create(
        model=model,
        messages=formatted,
        stream=True,
        temperature=temperature,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def friendly_error(err):
    t = str(err).lower()
    if "429" in t or "rate limit" in t:
        return "⏳ Rate limit reached — wait a few seconds and try again."
    if "401" in t or "auth" in t:
        return "🔑 API key issue. Check your OpenRouter key."
    if "context" in t or "too many tokens" in t:
        return "📚 Chat too long. Try clearing the chat."
    return f"⚠️ Something went wrong: {err}"


def clear_chat():
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pdf_text", "")
st.session_state.setdefault("pdf_name", None)

api_key = get_secret("OPENROUTER_API_KEY")
if not api_key:
    st.error("⚠️ Missing `OPENROUTER_API_KEY` in secrets.")
    st.stop()

client = get_client(api_key)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ TECHY")
    st.caption("Smart chat assistant")

    st.markdown("### 🧠 Model")
    model_label = st.selectbox(
        "Choose model", list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    selected_model = MODEL_OPTIONS[model_label]

    st.markdown("### 🎛️ Creativity")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.1)

    st.markdown("---")
    st.markdown("### 📄 Attach a PDF (optional)")
    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if uploaded:
        try:
            st.session_state.pdf_text = extract_pdf(uploaded.getvalue())
            st.session_state.pdf_name = uploaded.name
            st.success(f"Loaded: {uploaded.name}")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
    elif st.session_state.pdf_name:
        st.info(f"📎 Using: {st.session_state.pdf_name}")
        if st.button("Remove PDF", use_container_width=True):
            st.session_state.pdf_text = ""
            st.session_state.pdf_name = None
            st.rerun()

    st.markdown("---")
    st.button("🧹 Clear Chat", on_click=clear_chat, use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="techy-header">
  <div style="font-size:26px;">⚡</div>
  <div>
    <h1>TECHY</h1>
    <span>Your smart, friendly AI assistant</span>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.pdf_name:
    st.caption(f"📎 Document mode: `{st.session_state.pdf_name}`")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Message TECHY..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚡"):
        try:
            system_prompt = build_system_prompt(st.session_state.pdf_text)
            stream = stream_reply(
                client, system_prompt, st.session_state.messages,
                selected_model, temperature,
            )
            reply = st.write_stream(stream)
            if reply and reply.strip():
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
        except Exception as e:
            st.error(friendly_error(e))
