"""
=============================================================================
 Autonomous Data Analyst — Streamlit Web Application
=============================================================================
 Tech Stack : Streamlit · LangChain · Groq (llama-3.3-70b-versatile)
              Pandas · Matplotlib · Seaborn
 Theme      : Professional Indigo / Light Slate
 Standard   : PEP 8
=============================================================================
"""

import os
import io
import re
import logging
import traceback
import contextlib

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ---------------------------------------------------------------------------
# Bootstrap — must happen before any other Streamlit call
# ---------------------------------------------------------------------------

matplotlib.use("Agg")   # Non-interactive backend required for Streamlit

# AgentType is a plain string enum; use the literal to avoid fragile imports
# that shuffle location between LangChain 0.1 / 0.2 / 0.3.
AGENT_TYPE_ZERO_SHOT = "zero-shot-react-description"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHART_TEMP_PATH = "temp_chart.png"
DEFAULT_MODEL   = "llama-3.3-70b-versatile"
APP_TITLE       = "Data Analyst AI"
APP_SUBTITLE    = "Autonomous insights powered by Groq"
APP_ICON        = "📊"

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile" : "Llama 3.3 · 70B  —  Best reasoning",
    "llama-3.1-8b-instant"    : "Llama 3.1 · 8B   —  Fastest / low quota",
    "gemma2-9b-it"            : "Gemma 2 · 9B     —  Google · efficient",
    "mixtral-8x7b-32768"      : "Mixtral · 8×7B   —  32k context window",
}

# Prefix injected into every agent call.
# CRITICAL RULES enforce strict ReAct format so the parser never sees a
# "Final Answer" and an "Action" in the same response turn.
AGENT_PREFIX = """You are an expert senior data analyst with access to a pandas DataFrame called `df`.

STRICT OUTPUT FORMAT RULES — follow these exactly or the system will break:
1. Each response turn must contain EITHER an Action OR a Final Answer — NEVER both.
2. If you need to run code, output ONLY the Action/Action Input block. Wait for the Observation.
3. Only output "Final Answer:" after you have all the information you need. Never include an Action after Final Answer.
4. Do not combine analysis text with an Action in the same turn.

VISUALIZATION RULES — when asked for any chart, graph or plot:
- Use matplotlib or seaborn only.
- Save with: plt.savefig('""" + CHART_TEMP_PATH + """', bbox_inches='tight', dpi=150)
- Never call plt.show().
- Always call plt.close() after saving.
- Do NOT describe the chart in text AND run code in the same turn.

ANSWER RULES:
- Be concise, precise, and structured.
- Use markdown formatting (bold, lists) where helpful.
- State numbers with 2 decimal places where appropriate.
"""

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config  (first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Professional light indigo theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root variables ──────────────────────────────────────────────────────── */
:root {
    --indigo      : #4f46e5;
    --indigo-dark : #3730a3;
    --indigo-light: #eef2ff;
    --indigo-mid  : #c7d2fe;
    --slate-900   : #0f172a;
    --slate-700   : #334155;
    --slate-500   : #64748b;
    --slate-300   : #cbd5e1;
    --slate-200   : #e2e8f0;
    --slate-100   : #f1f5f9;
    --slate-50    : #f8fafc;
    --white       : #ffffff;
    --radius-sm   : 6px;
    --radius-md   : 10px;
    --radius-lg   : 16px;
    --shadow-sm   : 0 1px 3px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.04);
    --shadow-md   : 0 4px 16px rgba(15,23,42,.10), 0 1px 4px rgba(15,23,42,.06);
    --shadow-lg   : 0 8px 32px rgba(15,23,42,.12), 0 2px 8px rgba(15,23,42,.06);
}

/* ── Global reset / base ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--slate-900);
    background-color: var(--slate-50);
}

/* ── App shell ───────────────────────────────────────────────────────────── */
.stApp {
    background: var(--slate-50);
}

/* ── Main content padding ────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1100px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--slate-200) !important;
    box-shadow: 2px 0 12px rgba(15,23,42,.04);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem;
}

/* Sidebar brand strip at top */
[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, var(--indigo), #818cf8);
    position: sticky;
    top: 0;
    z-index: 10;
}

/* Sidebar section labels */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: var(--slate-700) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Sidebar text inputs */
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: var(--slate-50) !important;
    border: 1.5px solid var(--slate-200) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--slate-900) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color .2s;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--slate-50) !important;
    border: 1.5px solid var(--slate-200) !important;
    border-radius: var(--radius-sm) !important;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: var(--indigo-light);
    border: 1.5px dashed var(--indigo-mid);
    border-radius: var(--radius-md);
    padding: 0.5rem;
}

/* ── Sidebar divider ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid var(--slate-200);
    margin: 1.25rem 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HEADER / HERO
═══════════════════════════════════════════════════════════════════════════ */
.app-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.75rem 2rem 1.5rem;
    background: var(--white);
    border-radius: var(--radius-lg);
    border: 1px solid var(--slate-200);
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}

/* Indigo accent bar left edge */
.app-header::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--indigo) 0%, #818cf8 100%);
    border-radius: 4px 0 0 4px;
}

.header-icon {
    font-size: 2.2rem;
    line-height: 1;
    flex-shrink: 0;
    margin-top: 2px;
}

.header-text { flex: 1; }

.header-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--slate-900);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.25rem;
}

.header-title span {
    color: var(--indigo);
}

.header-sub {
    font-size: 0.9rem;
    color: var(--slate-500);
    font-weight: 400;
    margin: 0;
}

/* ── Dataset stat pills ───────────────────────────────────────────────────── */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--white);
    border: 1.5px solid var(--slate-200);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--slate-700);
    box-shadow: var(--shadow-sm);
    transition: border-color .2s, box-shadow .2s;
}
.stat-pill:hover {
    border-color: var(--indigo-mid);
    box-shadow: 0 0 0 3px rgba(79,70,229,.07);
}
.stat-pill .pill-icon { color: var(--indigo); font-size: 0.85rem; }

/* ── Section labels ────────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--slate-500);
    margin-bottom: 0.5rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATA PREVIEW EXPANDER
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 1.25rem;
}

[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--slate-700) !important;
    font-size: 0.9rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════════════════════════════════════════ */
/* Chat container wrapper */
.stChatFloatingInputContainer {
    border-top: 1px solid var(--slate-200) !important;
    background: var(--white) !important;
    padding: 0.75rem 1rem !important;
}

/* Both message types — base card */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-md) !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem 1.2rem !important;
    animation: slideUp 0.25s ease both;
    border: 1px solid transparent;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--indigo-light) !important;
    border-color: var(--indigo-mid) !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--white) !important;
    border-color: var(--slate-200) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: var(--indigo) !important;
    color: white !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--white) !important;
    border: 2px solid var(--indigo-mid) !important;
    color: var(--indigo) !important;
}

/* ── Chat input box ────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border: 2px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    background: var(--white) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    color: var(--slate-900) !important;
}

/* Send button */
[data-testid="stChatInput"] button {
    background: var(--indigo) !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
}
[data-testid="stChatInput"] button:hover {
    background: var(--indigo-dark) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   EMPTY-STATE WELCOME CARD
═══════════════════════════════════════════════════════════════════════════ */
.welcome-card {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    margin: 1.5rem 0;
}
.welcome-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--slate-700);
    margin-bottom: 0.5rem;
}
.welcome-card p {
    font-size: 0.88rem;
    color: var(--slate-500);
    margin: 0;
    line-height: 1.6;
}
.welcome-card .big-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    display: block;
}

/* ── Prompt chips ─────────────────────────────────────────────────────────── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1.25rem;
}
.chip {
    background: var(--indigo-light);
    border: 1px solid var(--indigo-mid);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--indigo-dark);
    cursor: default;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CODE  &  INLINE ELEMENTS
═══════════════════════════════════════════════════════════════════════════ */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--indigo-light) !important;
    color: var(--indigo-dark) !important;
    padding: 0.1em 0.35em !important;
    border-radius: 4px !important;
    font-size: 0.83em !important;
}

pre code {
    background: var(--slate-900) !important;
    color: #e2e8f0 !important;
    display: block;
    padding: 1rem !important;
    border-radius: var(--radius-md) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ALERTS / STATUS MESSAGES
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left-width: 4px !important;
    font-size: 0.88rem;
}

/* ── Generated chart frame ────────────────────────────────────────────────── */
.chart-frame {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-md);
    padding: 0.75rem;
    box-shadow: var(--shadow-sm);
    margin-top: 0.75rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATAFRAME  &  TABLE
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] {
    color: var(--indigo) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--slate-100); }
::-webkit-scrollbar-thumb { background: var(--slate-300); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--indigo-mid); }

/* ── Sidebar how-to steps ─────────────────────────────────────────────────── */
.step-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.step-list li {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    font-size: 0.83rem;
    color: var(--slate-600);
    padding: 0.35rem 0;
    line-height: 1.45;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    min-width: 18px;
    border-radius: 50%;
    background: var(--indigo);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    margin-top: 1px;
}

/* ── Status badge (model info) ───────────────────────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--indigo-light);
    border: 1px solid var(--indigo-mid);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--indigo-dark);
    font-weight: 500;
}
.model-badge::before {
    content: '●';
    color: #22c55e;
    font-size: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Ensure all required session-state keys exist on first run."""
    defaults = {
        "messages"      : [],
        "df"            : None,
        "agent"         : None,
        "groq_api_key"  : "",
        "file_name"     : "",
        "selected_model": DEFAULT_MODEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def build_agent(df: pd.DataFrame, api_key: str, model: str = DEFAULT_MODEL):
    """
    Initialise and return a LangChain Pandas DataFrame Agent.

    Parameters
    ----------
    df      : The pandas DataFrame to analyse.
    api_key : Groq API key supplied by the user.
    model   : Groq model identifier string.

    Returns
    -------
    LangChain agent executor, or None on failure.
    """
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=0,       # Deterministic code generation
            max_tokens=4096,
        )
        def _on_parse_error(error: Exception) -> str:
            """Feed parse errors back to the LLM so it self-corrects."""
            return (
                "Your previous response could not be parsed. "
                "Remember: output ONLY an Action/Action Input block OR a Final Answer "
                "— never both in the same turn. "
                f"The parsing error was: {error}"
            )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type=AGENT_TYPE_ZERO_SHOT,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=_on_parse_error,   # callable gives LLM self-correction hint
            prefix=AGENT_PREFIX,
            return_intermediate_steps=False,
            max_iterations=8,           # prevent infinite loops
            max_execution_time=120,     # 2-minute hard timeout
        )
        logger.info("Agent ready — model: %s", model)
        return agent

    except Exception as exc:
        logger.error("Agent init failed: %s", exc)
        st.error(f"❌ Failed to initialise the AI agent: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, pd.DataFrame | None, str]:
    """Render sidebar and return (api_key, df_or_none, selected_model)."""

    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;'
            f'margin-bottom:1.25rem;">'
            f'<span style="font-size:1.5rem">{APP_ICON}</span>'
            f'<div><div style="font-weight:700;font-size:1rem;'
            f'color:var(--slate-900);letter-spacing:-.02em">{APP_TITLE}</div>'
            f'<div style="font-size:.72rem;color:var(--slate-500);">'
            f'Groq · LangChain · Pandas</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── API Key ─────────────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label">🔑 Groq API Key</p>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            label="groq_key",
            type="password",
            placeholder="gsk_••••••••••••••••",
            help="Free key at console.groq.com",
            value=st.session_state.groq_api_key,
            label_visibility="collapsed",
        )
        if api_key:
            st.markdown(
                '<div style="font-size:.75rem;color:#16a34a;margin-top:-.25rem;'
                'margin-bottom:.5rem;">✓ API key entered</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Model selector ──────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label">🤖 AI Model</p>',
            unsafe_allow_html=True,
        )
        model_keys   = list(AVAILABLE_MODELS.keys())
        model_labels = list(AVAILABLE_MODELS.values())

        current_idx = (
            model_keys.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_keys else 0
        )
        selected_label = st.selectbox(
            label="model_select",
            options=model_labels,
            index=current_idx,
            label_visibility="collapsed",
        )
        selected_model = model_keys[model_labels.index(selected_label)]

        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.agent = None
            st.toast(f"Switched to {selected_model.split('-')[0].title()} model", icon="🔄")

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── File uploader ────────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label">📂 Dataset</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            label="csv_upload",
            type=["csv"],
            help="Upload any CSV file (recommended < 50 MB)",
            label_visibility="collapsed",
        )

        df = None
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.file_name:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df        = df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.agent     = None
                    st.session_state.messages  = []
                    st.toast(f"Loaded {uploaded_file.name}", icon="✅")
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")
            else:
                df = st.session_state.df

            # File meta pill
            if df is not None:
                st.markdown(
                    f'<div style="background:var(--indigo-light);border:1px solid '
                    f'var(--indigo-mid);border-radius:var(--radius-sm);padding:.45rem '
                    f'.7rem;margin-top:.5rem;">'
                    f'<div style="font-size:.75rem;font-weight:600;color:var(--indigo-dark);">'
                    f'📄 {uploaded_file.name}</div>'
                    f'<div style="font-size:.7rem;color:var(--slate-500);margin-top:.1rem;">'
                    f'{df.shape[0]:,} rows · {df.shape[1]} cols</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── How-to guide ─────────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label">📖 How to Use</p>',
            unsafe_allow_html=True,
        )
        steps = [
            "Enter your <b>Groq API key</b> above.",
            "Select an <b>AI model</b> from the dropdown.",
            "Upload a <b>CSV file</b> to analyse.",
            "Type any question in the chat box.",
            "Ask for <b>charts</b> — the AI generates them automatically.",
        ]
        items = "".join(
            f'<li><span class="step-num">{i+1}</span>{s}</li>'
            for i, s in enumerate(steps)
        )
        st.markdown(
            f'<ul class="step-list">{items}</ul>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Footer ────────────────────────────────────────────────────────────
        short_model = selected_model.split("-")[0].title()
        st.markdown(
            f'<div style="text-align:center;">'
            f'<span class="model-badge">{short_model} active</span>'
            f'<div style="font-size:.68rem;color:var(--slate-400);margin-top:.6rem;">'
            f'Powered by Groq · LangChain</div></div>',
            unsafe_allow_html=True,
        )

    return api_key, df, selected_model


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(df: pd.DataFrame | None) -> None:
    """Render the app hero bar and, if data is loaded, dataset stat pills."""

    st.markdown(
        f'<div class="app-header">'
        f'<div class="header-icon">{APP_ICON}</div>'
        f'<div class="header-text">'
        f'<p class="header-title">Autonomous <span>Data Analyst</span></p>'
        f'<p class="header-sub">{APP_SUBTITLE} — upload a CSV, ask anything in plain English</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if df is not None:
        mem_kb   = df.memory_usage(deep=True).sum() / 1024
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        st.markdown(
            f'<div class="stats-row">'
            f'<span class="stat-pill"><span class="pill-icon">⬛</span>{df.shape[0]:,} rows</span>'
            f'<span class="stat-pill"><span class="pill-icon">⬜</span>{df.shape[1]} columns</span>'
            f'<span class="stat-pill"><span class="pill-icon">#</span>{len(num_cols)} numeric</span>'
            f'<span class="stat-pill"><span class="pill-icon">Aa</span>{len(cat_cols)} categorical</span>'
            f'<span class="stat-pill"><span class="pill-icon">💾</span>{mem_kb:.1f} KB</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍  Preview Dataset — first 5 rows", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<p class="section-label" style="margin-top:.5rem;">Column Types</p>',
                    unsafe_allow_html=True,
                )
                dtype_df = (
                    df.dtypes.rename("type")
                    .reset_index()
                    .rename(columns={"index": "column"})
                )
                st.dataframe(dtype_df, use_container_width=True, height=200)
            with c2:
                st.markdown(
                    '<p class="section-label" style="margin-top:.5rem;">Missing Values</p>',
                    unsafe_allow_html=True,
                )
                null_df = (
                    df.isnull().sum().rename("nulls")
                    .reset_index().rename(columns={"index": "column"})
                )
                null_df = null_df[null_df["nulls"] > 0]
                if null_df.empty:
                    st.success("🎉 No missing values found!")
                else:
                    st.dataframe(null_df, use_container_width=True, height=200)


# ---------------------------------------------------------------------------
# Chart Helper
# ---------------------------------------------------------------------------

def display_and_cleanup_chart() -> None:
    """Display the temp chart (if generated) then delete the file."""
    if os.path.exists(CHART_TEMP_PATH):
        st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
        st.image(CHART_TEMP_PATH, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        try:
            os.remove(CHART_TEMP_PATH)
        except OSError as exc:
            logger.warning("Could not remove chart file: %s", exc)


# ---------------------------------------------------------------------------
# Agent Invocation  (with retry back-off)
# ---------------------------------------------------------------------------

def _parse_agent_result(result) -> str:
    """Extract string answer from whatever LangChain returns."""
    if isinstance(result, dict):
        return result.get("output", str(result))
    return str(result)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke_with_retry(agent, question: str) -> str:
    """Agent invocation wrapped with tenacity exponential back-off."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = agent.invoke({"input": question})
    return _parse_agent_result(result)


def invoke_agent(agent, question: str) -> str:
    """Public wrapper — retries up to 3× on transient failures."""
    return _invoke_with_retry(agent, question)


# ---------------------------------------------------------------------------
# Welcome / empty state
# ---------------------------------------------------------------------------

def render_welcome() -> None:
    """Show an onboarding card when no data has been uploaded yet."""
    sample_prompts = [
        "What are the column names?",
        "Show a bar chart of top 10 values",
        "Are there any missing values?",
        "What is the average of each numeric column?",
        "Show a correlation heatmap",
    ]
    chips = "".join(f'<span class="chip">{p}</span>' for p in sample_prompts)

    st.markdown(
        f'<div class="welcome-card">'
        f'<span class="big-icon">🗂️</span>'
        f'<h3>No dataset loaded yet</h3>'
        f'<p>Upload a CSV file from the sidebar to get started.<br>'
        f'Then ask questions like these in the chat below:</p>'
        f'<div class="chip-row">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

def render_chat(df: pd.DataFrame | None, api_key: str, selected_model: str) -> None:
    """Render persistent chat history and handle new user messages."""

    # Empty state card when no data is loaded
    if df is None and not st.session_state.messages:
        render_welcome()

    # Replay existing history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    placeholder = (
        "Ask anything about your data — e.g. 'Show a histogram of Exam_Score'"
        if df is not None
        else "Upload a CSV file to start chatting…"
    )
    user_input = st.chat_input(placeholder, disabled=(df is None))

    if not user_input:
        return

    # Guard: need both API key + data
    if not api_key:
        st.warning("⚠️ Please enter your **Groq API key** in the sidebar.")
        return
    if df is None:
        st.warning("⚠️ Please **upload a CSV file** in the sidebar first.")
        return

    # Echo user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build / reuse agent
    needs_rebuild = (
        st.session_state.agent is None
        or api_key != st.session_state.groq_api_key
        or selected_model != st.session_state.selected_model
    )
    if needs_rebuild:
        st.session_state.groq_api_key   = api_key
        st.session_state.selected_model = selected_model
        with st.spinner("⚙️ Initialising AI agent…"):
            st.session_state.agent = build_agent(df, api_key, selected_model)

    agent = st.session_state.agent
    if agent is None:
        return

    # Run agent and display answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analysing your data…"):
            try:
                answer = invoke_agent(agent, user_input)
                st.markdown(answer)
                display_and_cleanup_chart()
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except ValueError as exc:
                st.error(
                    "🔄 The model produced an unexpected response format. "
                    "Please rephrase your question.\n\n"
                    f"**Detail:** `{exc}`"
                )
                logger.error("ValueError: %s", exc)

            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                if "rate_limit_exceeded" in exc_str or "429" in exc_str:
                    wait_match = re.search(r"try again in ([\d minsec.]+)", exc_str)
                    wait_hint  = (
                        f" Groq says: **try again in {wait_match.group(1)}**."
                        if wait_match else ""
                    )
                    st.warning(
                        f"⏳ **Rate limit reached.**{wait_hint}\n\n"
                        "**Your options:**\n"
                        "- ⏰ Wait for the daily quota to reset (midnight UTC)\n"
                        "- 🔄 Switch to `llama-3.1-8b-instant` in the sidebar "
                        "(separate quota)\n"
                        "- 💳 [Upgrade to Groq Dev Tier]"
                        "(https://console.groq.com/settings/billing)"
                    )
                else:
                    st.error(
                        "❌ Something went wrong.\n\n"
                        f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
                        "Try rephrasing your question or check your API key."
                    )
                logger.error("Exception:\n%s", traceback.format_exc())

            finally:
                if os.path.exists(CHART_TEMP_PATH):
                    os.remove(CHART_TEMP_PATH)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate all app components."""
    init_session_state()
    api_key, df, selected_model = render_sidebar()

    # Fall back to session state if sidebar returns None (same file reloaded)
    if df is None and st.session_state.df is not None:
        df = st.session_state.df

    render_header(df)
    render_chat(df, api_key, selected_model)


if __name__ == "__main__":
    main()