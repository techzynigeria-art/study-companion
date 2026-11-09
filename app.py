"""
TECHY - A modern AI assistant & study companion built with Streamlit and OpenRouter.

Run locally: streamlit run app.py
"""

import io
import os

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# App Configuration & Settings
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TECHY AI",
    page_icon="⚡",
    layout="centered"
)

# OpenRouter auto-routes to an available free model
DEFAULT_MODEL = "openrouter/auto"

MAX_PDF_CHARS = 150_000      # Keeps big PDFs inside token limits
MAX_HISTORY_MESSAGES = 20    # Recent messages sent to the model

# ---------------------------------------------------------------------------
# Techy's Personality & Instructions
# ---------------------------------------------------------------------------

TECHY_PROMPT = """
You are TECHY, an ultra-smart, friendly, and intuitive AI assistant and mentor.
You excel at explaining complex technical ideas, engineering, coding, math, 
physics, and general knowledge with absolute clarity.

Style rules:
- Speak naturally, warmly, and directly. Never say "As an AI language model" or 
  talk about your restrictions unless specifically asked.
- Use clear real-world analogies to make tricky topics easy to grasp (e.g., 
  explaining voltage like water pressure or code functions like recipes).
- For Math & Engineering: Break down steps clearly, state units, and check calculations.
- For Programming: Write clean, modern, readable code snippets and explain what 
  key lines do.
- Keep responses engaging, well-structured, and concise using short paragraphs.
- If a user's question is unclear, ask a quick clarifying question before answering.
- Finish with an insightful follow-up question or small test when it helps solidify understanding.
""".strip()

PDF_RULES = """
IMPORTANT - DOCUMENT MODE:
The user has uploaded a document. Answer questions using ONLY the text below.
- If the answer is not in the document, state clearly that it isn't covered. Do not guess.
- Include page number references (e.g., [Page 3]) when available.
""".strip()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_setting(name):
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    return value or os.environ.get(name)


@st.cache_resource
def get_client(api_key):
    """Initialize OpenAI client pointed at OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes):
    """Extract text from uploaded PDF with page markers."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {number}]\n{text}")
    return "\n\n".join(pages)


def build_system_prompt(pdf_text):
    if not pdf_text:
        return TECHY_PROMPT
    return (
        TECHY_PROMPT
        + "\n\n"
        + PDF_RULES
        + "\n\n--- START OF DOCUMENT ---\n"
        + pdf_text
        + "\n--- END OF DOCUMENT ---"
    )


def stream_reply(client, system_prompt, messages):
    """Stream AI responses chunk by chunk."""
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for m in messages[-MAX_HISTORY_MESSAGES:]:
        formatted_messages.append({"role": m["role"], "content": m["content"]})

    model = get_setting("OPENROUTER_MODEL") or DEFAULT_MODEL

    stream = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        stream=True,
        temperature=0.6,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def friendly_error(error):
    text = str(error)
    if "API key" in text or "401" in text or "403" in text:
        return "API Key issue detected. Please check your Streamlit secrets setup."
    if "429" in text:
        return "Free limit temporarily reached. Give it a minute and try again."
    return f"Something went wrong: {text}"


def clear_chat():
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# App Initialization & Secrets Check
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = get_setting("OPENROUTER_API_KEY")
if not api_key:
    st.title("⚡ TECHY AI")
    st.error(
        "No API key found. Add `OPENROUTER_API_KEY` to your Streamlit secrets panel."
    )
    st.stop()

client = get_client(api_key)

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ TECHY Settings")
st.sidebar.markdown("---")

uploaded_pdf = st.sidebar.file_uploader("Upload Document (PDF)", type="pdf")
pdf_text = ""

if uploaded_pdf is not None:
    try:
        pdf_text = extract_pdf_text(uploaded_pdf.getvalue())
        read_failed = False
    except Exception:
        pdf_text = ""
        read_failed = True

    if read_failed:
        st.sidebar.error("Could not read this PDF file.")
    elif not pdf_text:
        st.sidebar.warning("No readable text found in this document.")
    else:
        if len(pdf_text) > MAX_PDF_CHARS:
            pdf_text = pdf_text[:MAX_PDF_CHARS]
            st.sidebar.warning("Using the first section of this long PDF.")
        st.sidebar.success(f"Loaded: {uploaded_pdf.name}")
else:
    st.sidebar.caption("No document attached. TECHY is using general knowledge.")

st.sidebar.markdown("---")
st.sidebar.button("Clear Conversation", on_click=clear_chat, use_container_width=True)

# ---------------------------------------------------------------------------
# Main Chat UI
# ---------------------------------------------------------------------------

st.title("⚡ TECHY")
st.caption("Your high-powered AI companion for coding, engineering, and learning.")

# Render existing chat history in clean chat bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bottom sticky chat input box
if user_input := st.chat_input("Ask TECHY anything..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream AI response
    with st.chat_message("assistant"):
        try:
            response_stream = stream_reply(
                client,
                build_system_prompt(pdf_text),
                st.session_state.messages,
            )
            reply_text = st.write_stream(response_stream)
        except Exception as err:
            reply_text = None
            st.error(friendly_error(err))

        if reply_text and str(reply_text).strip():
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
