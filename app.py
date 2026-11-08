"""
Study Companion - a minimal AI tutor built with Streamlit and the Gemini API.

Run it with:   streamlit run app.py
"""

import io
import os

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Settings you may want to change
# ---------------------------------------------------------------------------

# Models are tried in this order. If one is unavailable (retired, or its free
# daily limit is used up), the next one is tried automatically.
# To force a specific model, add  GEMINI_MODEL = "model-name"  to your secrets.
DEFAULT_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]

MAX_PDF_CHARS = 150_000      # keeps big PDFs inside the free-tier token limits
MAX_HISTORY_MESSAGES = 20    # how many recent messages are sent to the model

# ---------------------------------------------------------------------------
# The tutor's personality
# ---------------------------------------------------------------------------

TUTOR_PROMPT = """
You are a friendly, patient and highly logical engineering tutor, talking with
a university student. You speak like a real human mentor who sits next to the
student and wants them to truly understand.

Style rules:
- Never say "As an AI language model" or "As an AI". Never talk about your
  own limitations unless it is truly needed. Just teach, naturally and warmly.
  (If the student sincerely asks whether you are a person or an AI, answer
  honestly.)
- Explain ideas with real-world analogies. For example, compare electric
  current to water flowing in a pipe, voltage to water pressure, and a
  resistor to a narrow section of the pipe.
- Start with the simple intuition, then build up to the formal definition,
  formula or rule. Go one logical step at a time.
- For Math and Physics, show the working step by step, state the units, and
  double-check the arithmetic before giving a final answer.
- For Programming, give short, correct examples and explain what each important
  line does.
- For Islamic Studies, be respectful, accurate and balanced. Mention when
  scholars differ, and for personal religious rulings suggest asking a
  qualified scholar.
- For Arabic, give the Arabic text together with a simple transliteration and
  the meaning, and explain the grammar in plain words.
- Reply in the language the student writes in.
- Keep answers focused and not longer than needed. Use short paragraphs.
  Do not use emojis.
- When it helps, finish with one small question or exercise so the student can
  check their own understanding.
- If a question is unclear, ask one short clarifying question first.
- If you are not sure about something, say so honestly instead of guessing.
""".strip()

PDF_RULES = """
IMPORTANT - DOCUMENT MODE:
The student has uploaded a document. Answer questions using ONLY the document
text below. Do not use outside knowledge for facts.
- If the answer is not in the document, say clearly that the document does not
  cover it. Do not guess.
- When possible, mention the page number (the text contains markers such as
  [Page 3]).
- You may still use analogies and simple wording to explain what the document
  says, but the facts must come from the document.
""".strip()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_setting(name):
    """Read a value from Streamlit secrets first, then from environment variables."""
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    return value or os.environ.get(name)


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes):
    """Return the text of a PDF, with [Page N] markers."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {number}]\n{text}")
    return "\n\n".join(pages)


def build_system_prompt(pdf_text):
    if not pdf_text:
        return TUTOR_PROMPT
    return (
        TUTOR_PROMPT
        + "\n\n"
        + PDF_RULES
        + "\n\n--- START OF DOCUMENT ---\n"
        + pdf_text
        + "\n--- END OF DOCUMENT ---"
    )


def stream_reply(client, system_prompt, messages):
    """Ask Gemini for a reply and yield it piece by piece."""
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages[-MAX_HISTORY_MESSAGES:]
    ]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.6,
    )

    # Build the list of models to try (custom model first, then the last one that worked).
    models = list(DEFAULT_MODELS)
    custom = get_setting("GEMINI_MODEL")
    if custom:
        models.insert(0, custom)
    working = st.session_state.get("working_model")
    if working in models:
        models.remove(working)
        models.insert(0, working)

    last_error = None
    for model in models:
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            st.session_state.working_model = model
            return
        except Exception as error:
            last_error = error
            code = getattr(error, "code", None)
            # 404 = model gone, 429 = free limit reached, 503 = model busy.
            # In those cases (before any text was shown) try the next model.
            if started or code not in (404, 429, 503):
                raise
    raise last_error


def friendly_error(error):
    code = getattr(error, "code", None)
    text = str(error)
    if "API key" in text or code in (401, 403):
        return "The API key was rejected. Check that it is copied correctly in your secrets file."
    if code == 429:
        return "The free usage limit was reached. Wait a minute and try again."
    if code in (404, 503):
        return "No Gemini model is available right now. Try again later, or set GEMINI_MODEL in your secrets."
    return f"Something went wrong: {text}"


def show_message(role, text):
    st.markdown("**You**" if role == "user" else "**Tutor**")
    st.markdown(text)


def clear_chat():
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Study Companion", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = get_setting("GEMINI_API_KEY") or get_setting("GOOGLE_API_KEY")
if not api_key:
    st.title("Study Companion")
    st.error(
        "No API key found. Create the file .streamlit/secrets.toml and add:\n\n"
        'GEMINI_API_KEY = "your-key-here"'
    )
    st.stop()

client = get_client(api_key)

# ---------------------------------------------------------------------------
# Sidebar: PDF upload and Clear Chat
# ---------------------------------------------------------------------------

st.sidebar.header("Study material")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")

pdf_text = ""
if uploaded_pdf is not None:
    try:
        pdf_text = extract_pdf_text(uploaded_pdf.getvalue())
        read_failed = False
    except Exception:
        pdf_text = ""
        read_failed = True

    if read_failed:
        st.sidebar.error("This PDF could not be read. It may be damaged or password-protected.")
    elif not pdf_text:
        st.sidebar.warning(
            "No readable text found. Scanned PDFs (photos of pages) are not supported."
        )
    else:
        if len(pdf_text) > MAX_PDF_CHARS:
            pdf_text = pdf_text[:MAX_PDF_CHARS]
            st.sidebar.warning("The PDF is long, so only the first part is used.")
        st.sidebar.write(f"Answering only from: {uploaded_pdf.name}")
else:
    st.sidebar.caption("No PDF loaded. The tutor answers from general knowledge.")

st.sidebar.button("Clear Chat", on_click=clear_chat)

# ---------------------------------------------------------------------------
# Main area: chat history and input
# ---------------------------------------------------------------------------

st.title("Study Companion")
st.caption("Physics, Math, Islamic Studies, Arabic, Programming and more.")

history_box = st.container(height=520, border=True)

with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_area(
        "Your message",
        height=90,
        placeholder="Type your question here",
        label_visibility="collapsed",
    )
    sent = st.form_submit_button("Send")

with history_box:
    if not st.session_state.messages and not (sent and user_text.strip()):
        st.caption("Ask a question below to begin.")

    for message in st.session_state.messages:
        show_message(message["role"], message["content"])

    if sent and user_text.strip():
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})
        show_message("user", user_text.strip())

        st.markdown("**Tutor**")
        reply = None
        try:
            reply = st.write_stream(
                stream_reply(
                    client,
                    build_system_prompt(pdf_text),
                    st.session_state.messages,
                )
            )
        except Exception as error:
            st.error(friendly_error(error))

        if reply and str(reply).strip():
            st.session_state.messages.append({"role": "assistant", "content": reply})
        elif reply is not None:
            st.warning("No answer came back. Try rephrasing your question.")
