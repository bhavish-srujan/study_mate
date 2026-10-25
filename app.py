"""StudyMate AI — Streamlit entry point.

Phase 1: resource library, PDF/pasted-text ingestion, and indexing.
Phase 2: resource-filtered retrieval and a context preview.
Phase 3: grounded, validated question generation.
Phase 4: interactive quiz taking with answer persistence and a submit flow.
Phase 5: scoring, results summary, and post-submit answer review.
Phase 6: YouTube transcripts, grounded chat, and a pre-quiz chat-or-quiz choice.
"""

from __future__ import annotations

import html
import time

import streamlit as st
from pydantic import ValidationError

from chat.schemas import ChatMessage
from chat.service import ChatService, build_chat_service
from config.settings import Settings, get_settings
from generation.llm_client import GroqChatClient
from generation.question_generator import QuestionGenerator, build_question_generator
from generation.schemas import GenerationRequest
from ingestion.indexer import Indexer
from ingestion.models import LoadedResource
from ingestion.pdf_loader import load_pdf
from ingestion.schemas import ResourceInfo
from ingestion.text_loader import load_pasted_text
from ingestion.youtube_loader import load_youtube_transcript
from quiz.evaluator import QuizEvaluator, build_quiz_evaluator
from quiz.quiz_service import (
    QuizVault,
    answered_count,
    new_attempt,
    to_quiz_view,
    unanswered_question_ids,
)
from quiz.schemas import (
    QuestionResult,
    QuizAttempt,
    QuizQuestionView,
    QuizResult,
    QuizView,
)
from rag.context_builder import build_context
from rag.embeddings import build_embedder
from rag.retriever import Retriever, build_retriever
from rag.vector_store import ChromaVectorStore, build_vector_store
from utils.errors import StudyMateError
from utils.logging_config import configure_logging, get_logger

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")
configure_logging()
logger = get_logger(__name__)

SOURCE_LABELS = {
    "pdf": "PDF",
    "pasted_text": "Pasted text",
    "youtube": "YouTube",
}

SOURCE_ICONS = {
    "pdf": "📄",
    "pasted_text": "📝",
    "youtube": "🎥",
}

QUESTION_TYPE_LABELS = {
    "mcq": "Multiple choice",
    "fill_blank": "Fill in the blank",
    "short_answer": "Short answer",
}

RESULT_STATUS_LABELS = {
    "correct": "Correct",
    "partial": "Partial credit",
    "incorrect": "Incorrect",
    "unanswered": "Unanswered",
    "ungraded": "Not graded",
}

RESULT_STATUS_CLASSES = {
    "correct": "badge-easy",
    "partial": "badge-medium",
    "incorrect": "badge-hard",
    "unanswered": "badge-neutral",
    "ungraded": "badge-type",
}

NAV_LIBRARY = "📚 Library"
NAV_GENERATE = "✨ Generate"
NAV_CHAT = "💬 Chat"
NAV_QUIZ = "📝 Quiz"
NAV_OPTIONS = [NAV_LIBRARY, NAV_GENERATE, NAV_CHAT, NAV_QUIZ]

APP_CSS = """
<style>
:root {
    --sm-bg: #0B1120;
    --sm-surface: #111827;
    --sm-surface-2: #151F32;
    --sm-surface-3: #1E293B;
    --sm-ink: #E5E7EB;
    --sm-muted: #94A3B8;
    --sm-primary: #8B5CF6;
    --sm-primary-dark: #C4B5FD;
    --sm-border: #263247;
    --sm-soft: #0F172A;
    --sm-success: #6EE7B7;
    --sm-warning: #FCD34D;
    --sm-danger: #FDA4AF;
}
html, body, [data-testid="stAppViewContainer"] {
    font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(224, 231, 255, 0.9) 0, transparent 34%),
        radial-gradient(circle at 96% 4%, rgba(207, 250, 254, 0.8) 0, transparent 30%),
        var(--sm-soft);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5F3FF 0%, #EEF2FF 100%);
    border-right: 1px solid #E4E7F2;
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 18px 0;
}
.sidebar-brand .brand-icon {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    font-size: 1.35rem;
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.25);
}
.sidebar-brand .brand-title {
    font-weight: 800;
    font-size: 1.05rem;
    color: var(--sm-ink);
}
.sidebar-brand .brand-sub {
    font-size: 0.76rem;
    color: var(--sm-muted);
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.status-ok { background: #ECFDF3; color: #027A48; }
.status-warn { background: #FFFAEB; color: #B54708; }
.app-kicker {
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.72rem;
    font-weight: 800;
    opacity: 0.82;
    margin-bottom: 8px;
}
[data-testid="stButtonGroup"] {
    background: #EEF2FF;
    padding: 5px;
    border-radius: 14px;
    gap: 4px;
    margin-bottom: 1.1rem;
    width: 100%;
}
[data-testid="stButtonGroup"] button {
    flex: 1;
    border: 0 !important;
    border-radius: 10px !important;
    color: #475467 !important;
    font-weight: 700 !important;
}
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
    background: #FFFFFF !important;
    color: var(--sm-primary-dark) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.14);
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--sm-border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 22px rgba(16, 24, 40, 0.05);
}
[data-testid="stMetricLabel"] { color: var(--sm-muted); }
[data-testid="stMetricValue"] { color: var(--sm-ink); }
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid var(--sm-border);
    border-radius: 20px;
    padding: 18px 20px 8px 20px;
    box-shadow: 0 12px 30px rgba(16, 24, 40, 0.06);
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--sm-border);
    border-radius: 14px;
}
[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button, .stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid #D9DDF0;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border: 0;
    color: white;
    box-shadow: 0 8px 18px rgba(79, 70, 229, 0.22);
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(79, 70, 229, 0.18);
}
.quiz-hero {
    background: linear-gradient(135deg, #111827 0%, #312E81 100%);
    color: white;
    border-radius: 22px;
    padding: 24px 28px;
    margin-bottom: 1rem;
    box-shadow: 0 18px 40px rgba(17, 24, 39, 0.18);
}
.quiz-hero h2 {
    color: white;
    margin: 0 0 4px 0;
    font-size: 1.55rem;
    letter-spacing: -0.02em;
}
.quiz-hero p {
    color: #D1D5DB;
    margin: 0;
    font-size: 0.9rem;
}
.quiz-card {
    background: #FFFFFF;
    border: 1px solid var(--sm-border);
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 12px 30px rgba(16, 24, 40, 0.06);
    margin-bottom: 1rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(16, 24, 40, 0.05);
}
.quiz-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.01em;
}
.badge-easy { background: #ECFDF3; color: #027A48; }
.badge-medium { background: #FFFAEB; color: #B54708; }
.badge-hard { background: #FEF3F2; color: #B42318; }
.badge-type { background: #EEF2FF; color: #4338CA; }
.badge-neutral { background: #F2F4F7; color: #475467; }
.question-text {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--sm-ink);
    line-height: 1.55;
    margin: 4px 0 14px 0;
}
.empty-state {
    text-align: center;
    padding: 48px 24px;
    background: #FFFFFF;
    border: 1px dashed #CBD5E1;
    border-radius: 22px;
    box-shadow: 0 12px 30px rgba(16, 24, 40, 0.04);
}
.empty-state .icon { font-size: 2.2rem; margin-bottom: 10px; }
.empty-state h3 { margin: 0 0 6px 0; color: var(--sm-ink); }
.empty-state p { margin: 0 auto; max-width: 520px; color: var(--sm-muted); }
.answer-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.answer-card .q { font-weight: 700; color: var(--sm-ink); }
.answer-card .a { color: #475467; margin-top: 4px; }
[data-testid="stRadio"] label {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 6px;
    width: 100%;
}
[data-testid="stRadio"] label:hover {
    border-color: #A5B4FC;
    background: #EEF2FF;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    border-radius: 10px;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
}

/* Dark-only theme overrides. */
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(76, 29, 149, 0.32) 0, transparent 36%),
        radial-gradient(circle at 96% 4%, rgba(8, 145, 178, 0.18) 0, transparent 30%),
        var(--sm-bg);
    color: var(--sm-ink);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
    border-right: 1px solid var(--sm-border);
}
.sidebar-brand .brand-icon {
    background: linear-gradient(135deg, #7C3AED, #2563EB);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.35);
}
.status-ok {
    background: rgba(16, 185, 129, 0.14);
    color: var(--sm-success);
}
.status-warn {
    background: rgba(245, 158, 11, 0.14);
    color: var(--sm-warning);
}
[data-testid="stButtonGroup"] {
    background: var(--sm-surface-2);
    border: 1px solid var(--sm-border);
}
[data-testid="stButtonGroup"] button {
    color: var(--sm-muted) !important;
}
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
    background: var(--sm-surface-3) !important;
    color: var(--sm-primary-dark) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28);
}
[data-testid="stMetric"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricLabel"] { color: var(--sm-muted); }
[data-testid="stMetricValue"] { color: var(--sm-ink); }
[data-testid="stForm"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
}
[data-testid="stExpander"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
}
[data-testid="stExpander"] summary { color: var(--sm-ink); }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: var(--sm-surface-2);
    color: var(--sm-ink);
    border: 1px solid var(--sm-border);
}
.stButton > button:hover, .stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--sm-primary);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7C3AED, #2563EB);
    border: 0;
    color: white;
    box-shadow: 0 8px 18px rgba(124, 58, 237, 0.28);
}
.quiz-hero {
    background: linear-gradient(135deg, #111827 0%, #312E81 100%);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.3);
}
.quiz-hero p { color: #CBD5E1; }
.quiz-card {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
}
.badge-easy {
    background: rgba(16, 185, 129, 0.14);
    color: var(--sm-success);
}
.badge-medium {
    background: rgba(245, 158, 11, 0.14);
    color: var(--sm-warning);
}
.badge-hard {
    background: rgba(244, 63, 94, 0.14);
    color: var(--sm-danger);
}
.badge-type {
    background: rgba(139, 92, 246, 0.18);
    color: var(--sm-primary-dark);
}
.badge-neutral {
    background: rgba(148, 163, 184, 0.12);
    color: #CBD5E1;
}
.question-text { color: var(--sm-ink); }
.empty-state {
    background: var(--sm-surface);
    border: 1px dashed #334155;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
}
.empty-state h3 { color: var(--sm-ink); }
.empty-state p { color: var(--sm-muted); }
.answer-card {
    background: var(--sm-soft);
    border: 1px solid var(--sm-border);
}
.answer-card .q { color: var(--sm-ink); }
.answer-card .a { color: var(--sm-muted); }
[data-testid="stRadio"] label {
    background: var(--sm-soft);
    border: 1px solid var(--sm-border);
    color: var(--sm-ink);
}
[data-testid="stRadio"] label:hover {
    border-color: var(--sm-primary);
    background: #1E1B4B;
}
[data-testid="stRadio"] label p { color: var(--sm-ink); }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background: var(--sm-soft) !important;
    color: var(--sm-ink) !important;
    border-color: var(--sm-border) !important;
}
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    background-color: var(--sm-soft) !important;
    border-color: var(--sm-border) !important;
    color: var(--sm-ink) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background-color: var(--sm-surface) !important;
}
[data-baseweb="menu"] li { color: var(--sm-ink) !important; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--sm-soft);
    border: 1px dashed var(--sm-border);
}
[data-testid="stCaptionContainer"] { color: var(--sm-muted); }
[data-testid="stMarkdownContainer"] code {
    background: var(--sm-surface-2);
    color: #E9D5FF;
}
hr { border-color: var(--sm-border); }
a { color: var(--sm-primary-dark); }
::placeholder { color: #64748B !important; }

/* Phase 5 result cards. */
.result-card {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
}
.result-card .result-question {
    color: var(--sm-ink);
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.55;
    margin: 2px 0 12px 0;
}
.result-answer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
@media (max-width: 640px) {
    .result-answer-grid { grid-template-columns: 1fr; }
}
.result-label {
    color: var(--sm-muted);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.result-value {
    color: var(--sm-ink);
    line-height: 1.5;
    margin-bottom: 8px;
    white-space: pre-wrap;
}
.result-value.student-correct { color: var(--sm-success); }
.result-value.student-partial { color: var(--sm-warning); }
.result-value.student-incorrect { color: var(--sm-danger); }
.result-value.student-neutral { color: var(--sm-muted); }
.result-explanation {
    background: var(--sm-soft);
    border-left: 3px solid var(--sm-primary);
    border-radius: 8px;
    color: #CBD5E1;
    line-height: 1.6;
    margin-top: 8px;
    padding: 10px 12px;
}
.result-note {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.22);
    border-radius: 8px;
    color: var(--sm-warning);
    line-height: 1.5;
    margin-top: 8px;
    padding: 8px 10px;
}
.result-keypoints {
    margin-top: 8px;
}
.result-keypoints ul {
    color: var(--sm-muted);
    margin: 4px 0 0 18px;
    padding: 0;
}
.result-sources {
    color: var(--sm-muted);
    font-size: 0.8rem;
    line-height: 1.5;
    margin-top: 8px;
}

/* Phase 6 grounded chat. */
[data-testid="stChatMessage"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 16px;
    padding: 10px 14px;
    margin-bottom: 10px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: var(--sm-ink);
}

/* Organized layout system. */
.block-container {
    max-width: 1220px;
    padding-top: 1.25rem;
}
.page-header {
    padding: 2px 2px 18px 2px;
    border-bottom: 1px solid var(--sm-border);
    margin-bottom: 20px;
}
.page-kicker {
    color: var(--sm-primary-dark);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.page-header h1 {
    color: var(--sm-ink);
    font-size: clamp(1.65rem, 2.6vw, 2.15rem);
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}
.page-header p {
    color: var(--sm-muted);
    line-height: 1.6;
    margin: 0;
    max-width: 780px;
}
h1, h2, h3 { color: var(--sm-ink); }
.summary-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 4px 0 20px 0;
}
.summary-item {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 14px;
    padding: 12px 14px;
}
.summary-item span {
    display: block;
    color: var(--sm-muted);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.summary-item strong {
    color: var(--sm-ink);
    font-size: 1.1rem;
    line-height: 1.2;
    word-break: break-word;
}
.form-section-label {
    color: var(--sm-primary-dark);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 4px 0 8px 0;
}
.sidebar-card {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 16px;
    padding: 12px 14px;
    margin-bottom: 14px;
}
.sidebar-card .status-chip { margin-bottom: 4px; }
.sidebar-model {
    color: var(--sm-muted);
    font-size: 0.72rem;
    line-height: 1.4;
    margin: 0 0 10px 2px;
    word-break: break-word;
}
.sidebar-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}
.sidebar-stat {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 14px;
    padding: 11px 12px;
}
.sidebar-stat span {
    display: block;
    color: var(--sm-muted);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.sidebar-stat strong {
    color: var(--sm-ink);
    font-size: 1.25rem;
}
.sidebar-heading {
    color: var(--sm-ink);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin: 2px 0 6px 2px;
    text-transform: uppercase;
}
.sidebar-steps {
    color: var(--sm-muted);
    font-size: 0.8rem;
    line-height: 1.7;
    margin: 0 0 14px 0;
    padding-left: 18px;
}
.sidebar-steps strong { color: var(--sm-ink); }
.side-panel {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.side-panel ul {
    color: var(--sm-muted);
    line-height: 1.7;
    margin: 0;
    padding-left: 18px;
}
.side-panel li + li { margin-top: 6px; }
.quiz-summary {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.quiz-summary-title {
    color: var(--sm-ink);
    font-size: 1.15rem;
    font-weight: 800;
    line-height: 1.35;
}
.quiz-summary-meta {
    color: var(--sm-muted);
    font-size: 0.82rem;
    line-height: 1.5;
    margin-top: 3px;
}
.quiz-summary .quiz-meta {
    margin-bottom: 0;
    justify-content: flex-end;
}
.chat-empty {
    background: var(--sm-surface);
    border: 1px dashed var(--sm-border);
    border-radius: 20px;
    padding: 42px 28px;
    text-align: center;
}
.chat-empty .icon { font-size: 2rem; margin-bottom: 8px; }
.chat-empty h3 { color: var(--sm-ink); margin: 0 0 6px 0; }
.chat-empty p {
    color: var(--sm-muted);
    margin: 0 auto 14px auto;
    max-width: 520px;
    line-height: 1.6;
}
.chat-suggestions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
}
.chat-suggestions span {
    background: var(--sm-surface-2);
    border: 1px solid var(--sm-border);
    border-radius: 999px;
    color: var(--sm-muted);
    font-size: 0.74rem;
    padding: 5px 10px;
}
[data-baseweb="tab-list"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 14px;
    gap: 4px;
    padding: 5px;
}
[data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--sm-muted);
    font-weight: 700;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: var(--sm-surface-3);
    color: var(--sm-primary-dark);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
[data-testid="stChatInput"] {
    background: var(--sm-surface);
    border: 1px solid var(--sm-border);
    border-radius: 16px;
}
@media (max-width: 760px) {
    .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .quiz-summary { flex-direction: column; }
    .quiz-summary .quiz-meta { justify-content: flex-start; }
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def _render_page_header(kicker: str, title: str, description: str) -> None:
    """Render one consistent page heading across every section."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-kicker">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state(
    icon: str,
    title: str,
    body: str,
    *,
    button_label: str | None = None,
    nav: str | None = None,
    key: str | None = None,
) -> None:
    """Render a centered empty state with an optional navigation button."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">{icon}</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if button_label and nav:
        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
            key=key,
        ):
            st.session_state["_next_nav"] = nav
            st.rerun()


def _render_summary_strip(items: list[tuple[str, str]]) -> None:
    """Render a compact row of labeled values."""
    cells = "".join(
        f'<div class="summary-item"><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value in items
    )
    st.markdown(f'<div class="summary-strip">{cells}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _services() -> tuple[ChromaVectorStore, Indexer, Retriever]:
    """Create the vector store, indexer, and retriever once per process."""
    settings = get_settings()
    store = build_vector_store(settings)
    embedder = build_embedder(settings)
    indexer = Indexer(
        store=store,
        embedder=embedder,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    retriever = build_retriever(settings, store, embedder)
    return store, indexer, retriever


@st.cache_resource(show_spinner=False)
def _llm_client() -> GroqChatClient:
    """Create the Groq client once per server process (no network at startup)."""
    return GroqChatClient(get_settings())


@st.cache_resource(show_spinner=False)
def _question_generator() -> QuestionGenerator:
    """Create the generator lazily; it needs the LLM client and retriever."""
    settings = get_settings()
    _, _, retriever = _services()
    return build_question_generator(retriever, _llm_client(), settings)


@st.cache_resource(show_spinner=False)
def _quiz_vault() -> QuizVault:
    """Process-local answer-key vault; deliberately outside session state."""
    return QuizVault()


@st.cache_resource(show_spinner=False)
def _quiz_evaluator() -> QuizEvaluator:
    """Create the evaluator once; short answers use the LLM when configured."""
    settings = get_settings()
    llm = None
    if settings.llm_configured:
        try:
            llm = _llm_client()
        except StudyMateError as exc:
            logger.warning(
                "Could not create the short-answer grader: %s",
                exc.detail or exc,
            )
    return build_quiz_evaluator(llm)


def _question_count_label(count: int) -> str:
    return f"{count} question" if count == 1 else f"{count} questions"


def _clear_quiz_widget_state() -> None:
    for key in list(st.session_state):
        if key.startswith(("quiz_answer_", "quiz_jump_")):
            st.session_state.pop(key, None)


def _save_answer(question_id: str, widget_key: str) -> None:
    """Persist a widget value into the public QuizAttempt."""
    attempt = st.session_state.get("quiz_attempt")
    if not isinstance(attempt, QuizAttempt):
        return
    value = st.session_state.get(widget_key)
    if isinstance(value, list):
        value = value[0] if value else ""
    st.session_state["quiz_attempt"] = attempt.with_answer(question_id, value)


def _queue_flash(messages: list[tuple[str, str]]) -> None:
    st.session_state.setdefault("_flash_messages", []).extend(messages)


def _render_flash() -> None:
    for kind, message in st.session_state.pop("_flash_messages", []):
        getattr(st, kind)(message)


def _load_resources(store: ChromaVectorStore) -> list[ResourceInfo]:
    try:
        return store.list_resources()
    except StudyMateError as exc:
        logger.warning("Could not list resources: %s", exc.detail or exc)
        st.error(exc.user_message)
        return []
    except Exception:
        logger.exception("Unexpected error while listing resources")
        st.error("The resource library could not be loaded. Check the logs.")
        return []


def _index_resource(
    indexer: Indexer,
    resource: LoadedResource,
    label: str,
) -> tuple[str, str]:
    try:
        result = indexer.index(resource)
    except StudyMateError as exc:
        logger.warning("Indexing failed for %s: %s", label, exc.detail or exc)
        return "error", f"{label}: {exc.user_message}"
    except Exception:
        logger.exception("Unexpected indexing error for %s", label)
        return "error", f"{label}: an unexpected error occurred. Check the logs."

    name = result.resource.source_name
    if result.already_indexed:
        return (
            "info",
            f"'{name}' is already indexed ({result.resource.chunk_count} chunks).",
        )
    return "success", f"Indexed '{name}' — {result.resource.chunk_count} chunks."


def _delete_resource(indexer: Indexer, resource: ResourceInfo) -> None:
    """Delete one resource and leave a flash message for the rerun."""
    try:
        indexer.delete(resource.resource_id)
    except StudyMateError as exc:
        logger.warning(
            "Delete failed for %s: %s",
            resource.resource_id,
            exc.detail or exc,
        )
        _queue_flash([("error", exc.user_message)])
    except Exception:
        logger.exception(
            "Unexpected delete error for %s",
            resource.resource_id,
        )
        _queue_flash([("error", "Could not delete that resource.")])
    else:
        _queue_flash([("success", f"Deleted '{resource.source_name}'.")])
    st.rerun()


def _render_sidebar(
    settings: Settings,
    resources: list[ResourceInfo],
    retriever: Retriever,
) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">📚</div>
                <div>
                    <div class="brand-title">StudyMate AI</div>
                    <div class="brand-sub">Your study copilot</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        llm_ready = settings.llm_configured
        llm_class = "status-ok" if llm_ready else "status-warn"
        llm_label = "Groq ready" if llm_ready else "LLM not configured"
        llm_note = (
            f"Model: {settings.llm_model}"
            if llm_ready
            else "Add LLM_API_KEY and LLM_MODEL to .env."
        )
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="status-chip {llm_class}">● {html.escape(llm_label)}</div>
                <div class="sidebar-model">{html.escape(llm_note)}</div>
                <div class="status-chip status-ok">● Local embeddings</div>
                <div class="sidebar-model">{html.escape(settings.embedding_model)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total_chunks = sum(item.chunk_count for item in resources)
        st.markdown(
            f"""
            <div class="sidebar-stats">
                <div class="sidebar-stat">
                    <span>Resources</span><strong>{len(resources)}</strong>
                </div>
                <div class="sidebar-stat">
                    <span>Chunks</span><strong>{total_chunks}</strong>
                </div>
            </div>
            <div class="sidebar-heading">Quick start</div>
            <ol class="sidebar-steps">
                <li>Add material in <strong>Library</strong>.</li>
                <li>Chat with it or generate a quiz.</li>
                <li>Practice and review in <strong>Quiz</strong>.</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Advanced"):
            st.caption(
                "The first search loads the embedding model (~20s on CPU); "
                "repeated queries are cached."
            )
            if st.button(
                "Warm up search model",
                key="warmup_embedding_model",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Loading embedding model ..."):
                        retriever.warm_up()
                except StudyMateError as exc:
                    logger.warning(
                        "Embedding warm-up failed: %s",
                        exc.detail or exc,
                    )
                    st.error(exc.user_message)
                except Exception:
                    logger.exception("Unexpected embedding warm-up failure")
                    st.error(
                        "The embedding model could not be loaded. "
                        "Check the logs."
                    )
                else:
                    st.success("Search model ready.")

            st.divider()
            st.caption("Indexing settings")
            st.write(f"Chunk size: {settings.chunk_size_tokens} tokens")
            st.write(f"Chunk overlap: {settings.chunk_overlap_tokens} tokens")
            st.write(f"Embedding batch: {settings.embedding_batch_size}")
            st.write(f"Embedding max tokens: {settings.embedding_max_seq_length}")
            st.write(f"Max PDFs per upload: {settings.max_pdfs}")
            st.write(f"Collection: `{settings.chroma_collection}`")


def _render_pdf_upload(settings: Settings, indexer: Indexer) -> None:
    st.caption(
        "PDFs must contain selectable text. "
        f"Up to {settings.max_pdfs} files per upload. "
        "Scanned, image-only PDFs need OCR and are not supported yet."
    )

    with st.form("pdf_upload_form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Select up to {settings.max_pdfs} PDF files.",
        )
        submitted = st.form_submit_button("Index PDFs", type="primary")

    if not submitted:
        return

    if not uploaded_files:
        st.warning("Choose at least one PDF first.")
        return

    if len(uploaded_files) > settings.max_pdfs:
        st.error(
            f"Too many files: {len(uploaded_files)} selected, "
            f"but the limit is {settings.max_pdfs}."
        )
        return

    messages: list[tuple[str, str]] = []
    for uploaded in uploaded_files:
        try:
            resource = load_pdf(uploaded.getvalue(), uploaded.name)
        except StudyMateError as exc:
            logger.warning(
                "PDF extraction failed for %s: %s",
                uploaded.name,
                exc.detail or exc,
            )
            messages.append(("error", f"{uploaded.name}: {exc.user_message}"))
            continue
        except Exception:
            logger.exception("Unexpected PDF error for %s", uploaded.name)
            messages.append(
                ("error", f"{uploaded.name}: an unexpected error occurred.")
            )
            continue

        with st.spinner(f"Indexing {uploaded.name} ..."):
            messages.append(_index_resource(indexer, resource, uploaded.name))

    _queue_flash(messages)
    st.rerun()


def _render_notes_upload(indexer: Indexer) -> None:
    st.caption(
        "Paste lecture notes, summaries, or definitions. They go through the "
        "same chunking, embedding, and storage pipeline as PDFs."
    )

    with st.form("notes_form", clear_on_submit=False):
        title = st.text_input("Resource title", value="Pasted notes")
        notes = st.text_area(
            "Study notes",
            height=220,
            placeholder="Paste lecture notes, summaries, or definitions here ...",
        )
        submitted = st.form_submit_button("Index notes", type="primary")

    if not submitted:
        return

    if not notes.strip():
        st.warning("Paste some notes first.")
        return

    try:
        resource = load_pasted_text(notes, title)
    except StudyMateError as exc:
        st.error(exc.user_message)
        return

    with st.spinner("Indexing notes ..."):
        _queue_flash(
            [_index_resource(indexer, resource, title.strip() or "Pasted notes")]
        )
    st.rerun()


def _render_youtube_upload(indexer: Indexer) -> None:
    st.caption(
        "Fetches the public transcript and keeps timestamped pages, so chat "
        "answers and quiz sources can point back to the lecture. The video "
        "must have captions available."
    )

    with st.form("youtube_form", clear_on_submit=False):
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
        )
        title = st.text_input(
            "Resource title (optional)",
            placeholder="e.g. Lecture 4 — CPU scheduling",
            key="youtube_title_input",
        )
        submitted = st.form_submit_button("Index transcript", type="primary")

    if not submitted:
        return

    if not url.strip():
        st.warning("Paste a YouTube URL first.")
        return

    try:
        with st.spinner("Fetching the transcript ..."):
            resource = load_youtube_transcript(url, title=title)
    except StudyMateError as exc:
        logger.warning(
            "YouTube transcript loading failed: %s",
            exc.detail or exc,
        )
        st.error(exc.user_message)
        return
    except Exception:
        logger.exception("Unexpected YouTube transcript error")
        st.error("The transcript could not be fetched. Check the logs.")
        return

    with st.spinner("Indexing the transcript ..."):
        _queue_flash(
            [_index_resource(indexer, resource, resource.source_name)]
        )
    st.rerun()


def _render_library(
    settings: Settings,
    indexer: Indexer,
    resources: list[ResourceInfo],
) -> None:
    _render_page_header(
        "Workspace / Library",
        "Your study library",
        "Add PDFs, pasted notes, or YouTube lectures. Everything is chunked "
        "and embedded locally; chat and quizzes retrieve only the passages "
        "they need.",
    )

    st.subheader("Add material")
    st.caption("Choose a source type. Each source is indexed once by content.")
    pdf_tab, notes_tab, youtube_tab = st.tabs(
        ["📄 PDFs", "📝 Pasted notes", "🎥 YouTube"]
    )
    with pdf_tab:
        _render_pdf_upload(settings, indexer)
    with notes_tab:
        _render_notes_upload(indexer)
    with youtube_tab:
        _render_youtube_upload(indexer)

    st.divider()
    st.subheader("Indexed resources")

    if not resources:
        _render_empty_state(
            "📚",
            "Your library is empty",
            "Use the tabs above to add your first PDF, set of notes, or "
            "YouTube lecture. Indexed material appears here.",
        )
        return

    total_chunks = sum(resource.chunk_count for resource in resources)
    resource_label = "resource" if len(resources) == 1 else "resources"
    chunk_label = "chunk" if total_chunks == 1 else "chunks"
    st.caption(
        f"{len(resources)} {resource_label} · {total_chunks} {chunk_label}"
    )

    for resource in resources:
        with st.container(border=True):
            name_col, action_col = st.columns([6, 1], gap="small")
            with name_col:
                icon = SOURCE_ICONS.get(resource.source_type, "📄")
                source_label = SOURCE_LABELS.get(
                    resource.source_type,
                    resource.source_type,
                )
                st.markdown(
                    f"**{icon} {html.escape(resource.source_name)}**"
                )
                chunk_word = (
                    "chunk" if resource.chunk_count == 1 else "chunks"
                )
                st.markdown(
                    f'<span class="badge badge-type">'
                    f"{html.escape(source_label)}</span> "
                    f'<span class="badge badge-neutral">'
                    f"{resource.chunk_count} {chunk_word}</span>",
                    unsafe_allow_html=True,
                )
            with action_col:
                with st.popover(
                    "Delete",
                    use_container_width=True,
                    key=f"delete_popover_{resource.resource_id}",
                ):
                    st.caption(
                        f"Delete {resource.source_name}? This removes its "
                        "chunks from the index."
                    )
                    if st.button(
                        "Confirm delete",
                        type="primary",
                        use_container_width=True,
                        key=f"confirm_delete_{resource.resource_id}",
                    ):
                        _delete_resource(indexer, resource)


def _render_retrieval_preview(
    settings: Settings,
    resources: list[ResourceInfo],
    retriever: Retriever,
) -> None:
    """Let the user inspect the passages retrieval would send to the LLM."""
    with st.expander("🔎 Retrieval preview (development tool)", expanded=False):
        st.caption(
            "Search your indexed resources with the same resource-filtered "
            "vector search that generation will use. Nothing is sent to the LLM."
        )

        if not resources:
            st.info("Index at least one resource to preview retrieval.")
            return

        resource_names = {
            resource.resource_id: resource.source_name for resource in resources
        }
        selected = st.multiselect(
            "Resources to search",
            options=list(resource_names),
            default=list(resource_names),
            format_func=lambda resource_id: resource_names.get(
                resource_id, resource_id
            ),
            key="retrieval_resource_ids",
        )
        # Drop IDs that were selected before a resource was deleted.
        selected = [
            resource_id
            for resource_id in selected
            if resource_id in resource_names
        ]
        query = st.text_input(
            "Topic or question",
            placeholder="e.g. CPU scheduling algorithms",
            key="retrieval_query_text",
        )

        if not st.button("Preview retrieval", key="retrieval_preview_button"):
            return

        if not selected:
            st.warning("Select at least one resource.")
            return
        if not query.strip():
            st.warning("Enter a topic or question first.")
            return

        try:
            with st.spinner("Embedding query and searching indexed chunks ..."):
                started = time.perf_counter()
                hits = retriever.retrieve(query, selected)
                context = build_context(hits, settings.max_context_tokens)
                elapsed_ms = (time.perf_counter() - started) * 1000
        except StudyMateError as exc:
            logger.warning("Retrieval preview failed: %s", exc.detail or exc)
            st.error(exc.user_message)
        except Exception:
            logger.exception("Unexpected retrieval preview failure")
            st.error("Retrieval failed. Check the logs for details.")
        else:
            if not hits:
                st.info("No relevant passages were found. Try a broader topic.")
                return

            resource_count = len({hit.resource_id for hit in hits})
            st.success(
                f"Found {len(hits)} passage(s) from {resource_count} "
                f"resource(s)."
            )
            for rank, hit in enumerate(hits, start=1):
                st.markdown(
                    f"**{rank}. {hit.source_label()}** · "
                    f"similarity `{hit.score:.3f}`"
                )
                snippet = hit.text
                if len(snippet) > 700:
                    snippet = snippet[:700] + " …"
                st.write(snippet)

            st.text_area(
                "Formatted context (what the LLM will see)",
                value=context.text,
                height=240,
                disabled=True,
            )
            st.caption(
                f"Retrieved in {elapsed_ms:.0f} ms · "
                f"{context.token_count} tokens · "
                f"truncated: {'yes' if context.truncated else 'no'}"
            )


def _chat_resource_defaults(resources: list[ResourceInfo]) -> list[str]:
    """Prefer the active quiz's resources, then fall back to every resource."""
    available = [resource.resource_id for resource in resources]
    active_quiz = st.session_state.get("active_quiz")
    if isinstance(active_quiz, QuizView):
        preferred = [
            resource_id
            for resource_id in active_quiz.resource_ids
            if resource_id in available
        ]
        if preferred:
            return preferred
    return available


def _chat_messages() -> list[ChatMessage]:
    stored = st.session_state.get("chat_messages", [])
    if not isinstance(stored, list):
        return []
    return [message for message in stored if isinstance(message, ChatMessage)]


def _render_chat_messages(messages: list[ChatMessage]) -> None:
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role != "assistant":
                continue
            if message.sources:
                with st.expander(
                    f"Sources ({len(message.sources)})",
                    expanded=False,
                ):
                    for source in message.sources:
                        st.markdown(f"**{source.source_label}**")
                        if source.score is not None:
                            st.caption(f"Similarity {source.score:.3f}")
                        st.write(source.snippet)
            elif not message.grounded:
                st.caption(
                    "No matching passage was found in the selected resources."
                )


def _render_chat_tab(
    settings: Settings,
    resources: list[ResourceInfo],
    retriever: Retriever,
) -> None:
    _render_page_header(
        "Study / Chat",
        "Chat with your material",
        "Ask questions about the resources you select. Answers are grounded in "
        "retrieved passages and cite their sources, including page numbers or "
        "YouTube timestamps.",
    )

    if not resources:
        _render_empty_state(
            "💬",
            "No material to chat with yet",
            "Index a PDF, pasted notes, or a YouTube transcript in the "
            "Library, then come back to ask questions.",
            button_label="Go to Library",
            nav=NAV_LIBRARY,
            key="go_library_from_chat",
        )
        return

    available_ids = [resource.resource_id for resource in resources]
    resource_names = {
        resource.resource_id: resource.source_name for resource in resources
    }
    stored_ids = st.session_state.get("chat_resource_ids")
    if not isinstance(stored_ids, list):
        st.session_state["chat_resource_ids"] = _chat_resource_defaults(resources)
    else:
        valid_ids = [
            resource_id
            for resource_id in stored_ids
            if resource_id in available_ids
        ]
        if valid_ids != stored_ids:
            # Keep an intentional empty selection, but replace stale IDs.
            st.session_state["chat_resource_ids"] = (
                valid_ids or _chat_resource_defaults(resources)
            )

    messages = _chat_messages()
    chat_col, context_col = st.columns([2.4, 1], gap="large")

    with context_col:
        st.subheader("Context")
        with st.container(border=True):
            selected = st.multiselect(
                "Resources to chat with",
                options=available_ids,
                format_func=lambda resource_id: resource_names.get(
                    resource_id, resource_id
                ),
                key="chat_resource_ids",
                help="New questions search only these resources.",
            )
            st.caption("New questions search only the selected resources.")

            if not selected:
                st.warning("Select at least one resource.")
            if not settings.llm_configured:
                st.warning(
                    "Set `LLM_API_KEY` and `LLM_MODEL` in `.env` to enable chat."
                )

            active_quiz = st.session_state.get("active_quiz")
            attempt = st.session_state.get("quiz_attempt")
            if (
                isinstance(active_quiz, QuizView)
                and isinstance(attempt, QuizAttempt)
                and attempt.quiz_id == active_quiz.quiz_id
                and not attempt.submitted
            ):
                if st.button(
                    "Return to quiz",
                    type="primary",
                    use_container_width=True,
                    key="return_to_quiz_from_chat",
                ):
                    st.session_state["_next_nav"] = NAV_QUIZ
                    st.rerun()

            if st.button(
                "Clear chat",
                use_container_width=True,
                disabled=not messages,
                key="clear_chat",
            ):
                st.session_state["chat_messages"] = []
                st.rerun()

            st.caption(
                "Changing the selection affects new questions. Previous "
                "answers stay in the transcript."
            )

    with chat_col:
        if messages:
            _render_chat_messages(messages)
        else:
            st.markdown(
                """
                <div class="chat-empty">
                    <div class="icon">💬</div>
                    <h3>Start a conversation</h3>
                    <p>Ask about a definition, compare two ideas, or request a
                    summary of the selected material.</p>
                    <div class="chat-suggestions">
                        <span>Summarize the main idea</span>
                        <span>Explain the key terms</span>
                        <span>Compare the main concepts</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not settings.llm_configured or not selected:
        return

    try:
        service: ChatService = build_chat_service(
            retriever,
            _llm_client(),
            settings,
        )
    except StudyMateError as exc:
        logger.warning("Could not create the chat service: %s", exc.detail or exc)
        st.error(exc.user_message)
        return

    prompt = st.chat_input(
        "Ask a question about your selected material ...",
        key="chat_question_input",
    )
    if not prompt or not prompt.strip():
        return

    user_message = ChatMessage(role="user", content=prompt)
    st.session_state["chat_messages"] = [*messages, user_message]
    try:
        with st.spinner("Searching your material and asking StudyMate ..."):
            answer = service.answer(prompt, selected, history=messages)
    except StudyMateError as exc:
        logger.warning("Chat failed: %s", exc.detail or exc)
        _queue_flash([("error", exc.user_message)])
        st.rerun()
    except Exception:
        logger.exception("Unexpected chat failure")
        _queue_flash([("error", "Chat failed. Check the logs for details.")])
        st.rerun()

    assistant_message = ChatMessage(
        role="assistant",
        content=answer.text,
        sources=list(answer.sources),
        grounded=answer.grounded,
    )
    st.session_state["chat_messages"] = [*messages, user_message, assistant_message]
    st.rerun()


def _render_generation_form(
    settings: Settings,
    resources: list[ResourceInfo],
) -> None:
    """Render the quiz configuration form and run generation on submit."""
    if not resources:
        st.warning("Index at least one resource in the Library tab first.")
        return
    if not settings.llm_configured:
        st.warning(
            "Set `LLM_API_KEY` and `LLM_MODEL` in `.env` to enable generation."
        )
        return

    resource_names = {
        resource.resource_id: resource.source_name for resource in resources
    }
    default_total = min(
        settings.default_question_count,
        settings.max_questions,
    )
    default_easy = min(3, default_total)
    default_medium = min(5, default_total - default_easy)
    default_hard = default_total - default_easy - default_medium

    with st.form("generation_form"):
        st.markdown(
            '<div class="form-section-label">1 · Sources</div>',
            unsafe_allow_html=True,
        )
        selected_resources = st.multiselect(
            "Resources",
            options=list(resource_names),
            default=list(resource_names),
            format_func=lambda resource_id: resource_names.get(
                resource_id, resource_id
            ),
            key="generation_resource_ids",
        )

        st.markdown(
            '<div class="form-section-label">2 · Question mix</div>',
            unsafe_allow_html=True,
        )
        selected_types = st.multiselect(
            "Question types",
            options=list(QUESTION_TYPE_LABELS),
            default=["mcq"],
            format_func=lambda value: QUESTION_TYPE_LABELS[value],
            key="generation_question_types",
        )
        total = st.number_input(
            "Total questions",
            min_value=1,
            max_value=settings.max_questions,
            value=default_total,
            step=1,
        )

        st.markdown(
            '<div class="form-section-label">3 · Difficulty mix</div>',
            unsafe_allow_html=True,
        )
        easy_col, medium_col, hard_col = st.columns(3)
        easy = easy_col.number_input(
            "Easy",
            min_value=0,
            max_value=total,
            value=min(default_easy, total),
            step=1,
        )
        medium = medium_col.number_input(
            "Medium",
            min_value=0,
            max_value=total,
            value=min(default_medium, total),
            step=1,
        )
        hard = hard_col.number_input(
            "Hard",
            min_value=0,
            max_value=total,
            value=min(default_hard, total),
            step=1,
        )
        st.caption("Easy + medium + hard must equal the total question count.")

        st.markdown(
            '<div class="form-section-label">4 · Focus</div>',
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "Optional topic or chapter",
            placeholder="e.g. CPU scheduling",
            key="generation_topic",
        )
        submitted = st.form_submit_button(
            "Generate questions",
            type="primary",
        )

    if not submitted:
        return

    selected_resources = [
        resource_id
        for resource_id in selected_resources
        if resource_id in resource_names
    ]
    if not selected_resources:
        st.warning("Select at least one resource.")
        return
    if not selected_types:
        st.warning("Select at least one question type.")
        return

    try:
        request = GenerationRequest(
            resource_ids=selected_resources,
            question_types=selected_types,
            total_questions=int(total),
            easy=int(easy),
            medium=int(medium),
            hard=int(hard),
            topic=topic,
        )
    except ValidationError as exc:
        for error in exc.errors()[:5]:
            st.error(error["msg"])
        return

    try:
        generator = _question_generator()
        with st.spinner("Retrieving context and generating questions ..."):
            quiz = generator.generate(request)
    except StudyMateError as exc:
        logger.warning("Question generation failed: %s", exc.detail or exc)
        st.error(exc.user_message)
    except Exception:
        logger.exception("Unexpected question-generation failure")
        st.error("Question generation failed. Check the logs for details.")
    else:
        vault = _quiz_vault()
        previous_view = st.session_state.get("active_quiz")
        if (
            isinstance(previous_view, QuizView)
            and previous_view.quiz_id != quiz.quiz_id
        ):
            vault.discard(previous_view.quiz_id)

        # The full quiz, including the key, stays in the process-local vault.
        vault.store(quiz)
        view = to_quiz_view(quiz)
        st.session_state["active_quiz"] = view
        st.session_state["quiz_attempt"] = new_attempt(view)
        st.session_state["quiz_index"] = 0
        st.session_state.pop("submit_confirmation", None)
        st.session_state.pop("quiz_result", None)
        st.session_state.pop("quiz_entry_choice", None)
        st.session_state["chat_resource_ids"] = list(view.resource_ids)
        _clear_quiz_widget_state()
        st.session_state["_next_nav"] = NAV_QUIZ
        _queue_flash(
            [
                (
                    "success",
                    f"Quiz ready — {_question_count_label(len(view.questions))}. "
                    "Chat with StudyMate first or start when you are ready.",
                )
            ]
        )
        st.rerun()


def _render_quiz_summary(view: QuizView) -> None:
    type_labels = " · ".join(
        QUESTION_TYPE_LABELS[question_type]
        for question_type in view.question_types
    )
    topic = html.escape(view.topic or "Practice quiz")
    badges = "".join(
        f'<span class="badge badge-{difficulty}">'
        f"{difficulty.title()} {count}</span>"
        for difficulty, count in view.difficulty_distribution.items()
        if count
    )
    st.markdown(
        f"""
        <div class="quiz-summary">
            <div>
                <div class="quiz-summary-title">{topic}</div>
                <div class="quiz-summary-meta">
                    {_question_count_label(len(view.questions))} ·
                    {html.escape(type_labels)}
                </div>
            </div>
            <div class="quiz-meta">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quiz_ready_card(view: QuizView, attempt: QuizAttempt) -> None:
    status = "Submitted" if attempt.submitted else "Quiz ready"
    status_class = "badge-easy" if attempt.submitted else "badge-type"
    topic = html.escape(view.topic or "Practice quiz")
    detail = (
        "Your answers are submitted. Open the quiz to review your score."
        if attempt.submitted
        else "Your answer key stays hidden while you take the quiz."
    )
    st.markdown(
        f"""
        <div class="quiz-card">
            <div class="quiz-meta">
                <span class="badge {status_class}">{status}</span>
                <span class="badge badge-neutral">
                    {_question_count_label(len(view.questions))}
                </span>
            </div>
            <div class="question-text" style="font-size:1.2rem;">{topic}</div>
            <p style="color:#94A3B8;margin:0;">{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Open quiz",
        type="primary",
        use_container_width=True,
        key="open_quiz_from_generate",
    ):
        st.session_state["_next_nav"] = NAV_QUIZ
        st.rerun()
    if not attempt.submitted and st.button(
        "💬 Chat with StudyMate first",
        use_container_width=True,
        key="chat_from_generate",
    ):
        st.session_state["quiz_entry_choice"] = "chat"
        st.session_state["chat_resource_ids"] = list(view.resource_ids)
        st.session_state["_next_nav"] = NAV_CHAT
        st.rerun()


def _render_quiz_empty_state() -> None:
    _render_page_header(
        "Practice / Quiz",
        "Quiz",
        "Generate a quiz from your indexed resources, then come back to "
        "practice. Your answers save automatically.",
    )
    _render_empty_state(
        "📝",
        "No active quiz yet",
        "Generate a quiz from your indexed resources, then come back here "
        "to practice.",
        button_label="Go to Generate",
        nav=NAV_GENERATE,
        key="go_generate_from_quiz",
    )


def _render_quiz_choice(view: QuizView) -> None:
    """Ask the student to chat first or start the quiz."""
    _render_page_header(
        "Practice / Quiz",
        "Ready when you are",
        "Review the material with chat, or jump straight into the quiz. "
        "Answers save automatically once you start.",
    )
    _render_quiz_summary(view)

    chat_col, quiz_col = st.columns(2, gap="large")
    with chat_col:
        with st.container(border=True):
            st.markdown("### 💬 Review with chat")
            st.caption(
                "Ask StudyMate about the same indexed material. Use it to "
                "warm up, then come back when you are ready."
            )
            if st.button(
                "💬 Chat with StudyMate first",
                use_container_width=True,
                key=f"chat_before_quiz_{view.quiz_id}",
            ):
                st.session_state["quiz_entry_choice"] = "chat"
                st.session_state["chat_resource_ids"] = list(view.resource_ids)
                st.session_state["_next_nav"] = NAV_CHAT
                st.rerun()

    with quiz_col:
        with st.container(border=True):
            st.markdown("### 📝 Take the quiz")
            st.caption(
                "Start now and work through the questions one at a time. "
                "The answer key stays hidden until you submit."
            )
            if st.button(
                "📝 Take the quiz now",
                type="primary",
                use_container_width=True,
                key=f"take_quiz_{view.quiz_id}",
            ):
                st.session_state["quiz_entry_choice"] = "quiz"
                st.rerun()

    st.caption("You can open the Chat tab at any time.")


def _render_answer_widget(
    view: QuizView,
    attempt: QuizAttempt,
    question: QuizQuestionView,
) -> None:
    widget_key = f"quiz_answer_{view.quiz_id}_{question.question_id}"
    current = attempt.answers.get(question.question_id)

    if question.question_type == "mcq":
        if not question.options:
            st.warning("This multiple-choice question has no options.")
            return
        options = [str(index) for index in range(len(question.options))]
        selected_index = int(current) if current in options else None
        st.radio(
            "Choose one answer",
            options=options,
            index=selected_index,
            format_func=lambda value: (
                f"{chr(65 + int(value))}. {question.options[int(value)]}"
            ),
            key=widget_key,
            on_change=_save_answer,
            args=(question.question_id, widget_key),
            label_visibility="collapsed",
        )
    elif question.question_type == "fill_blank":
        st.text_input(
            "Your answer",
            value=current or "",
            key=widget_key,
            on_change=_save_answer,
            args=(question.question_id, widget_key),
            placeholder="Type the missing word or phrase",
        )
    else:
        st.text_area(
            "Your answer",
            value=current or "",
            key=widget_key,
            on_change=_save_answer,
            args=(question.question_id, widget_key),
            height=120,
            placeholder="Write a short answer ...",
        )


def _score_attempt(view: QuizView, attempt: QuizAttempt) -> QuizResult | None:
    """Score a submitted attempt, showing a safe error if it cannot be done."""
    full_quiz = _quiz_vault().get(view.quiz_id)
    if full_quiz is None:
        st.error(
            "The answer key for this quiz is no longer available. "
            "Generate a new quiz."
        )
        return None

    try:
        return _quiz_evaluator().evaluate(full_quiz, attempt)
    except StudyMateError as exc:
        logger.warning("Quiz scoring failed: %s", exc.detail or exc)
        st.error(exc.user_message)
    except Exception:
        logger.exception("Unexpected quiz-scoring failure")
        st.error("Quiz scoring failed. Check the logs for details.")
    return None


def _submit_attempt(view: QuizView, attempt: QuizAttempt) -> None:
    submitted = attempt.submit()
    with st.spinner("Scoring your answers ..."):
        result = _score_attempt(view, submitted)
    if result is None:
        return

    st.session_state["quiz_attempt"] = submitted
    st.session_state["quiz_result"] = result
    st.session_state.pop("submit_confirmation", None)
    _queue_flash([("success", "Quiz submitted and scored.")])
    st.rerun()


def _render_submit_controls(view: QuizView, attempt: QuizAttempt) -> None:
    st.divider()
    if not st.session_state.get("submit_confirmation"):
        if st.button(
            "Submit quiz",
            type="primary",
            use_container_width=True,
            key=f"submit_{view.quiz_id}",
        ):
            if unanswered_question_ids(view, attempt):
                st.session_state["submit_confirmation"] = True
                st.rerun()
            _submit_attempt(view, attempt)
        return

    unanswered = unanswered_question_ids(view, attempt)
    if unanswered:
        question_numbers = {
            question.question_id: number
            for number, question in enumerate(view.questions, start=1)
        }
        numbers = ", ".join(
            str(question_numbers[question_id])
            for question_id in unanswered
        )
        st.warning(
            f"{len(unanswered)} question(s) unanswered: {numbers}. "
            "Submit anyway?"
        )
    else:
        st.info("Everything is answered. Ready to submit?")

    keep_col, submit_col = st.columns(2)
    if keep_col.button(
        "Keep working",
        use_container_width=True,
        key=f"keep_{view.quiz_id}",
    ):
        st.session_state.pop("submit_confirmation", None)
        st.rerun()
    if submit_col.button(
        "Submit anyway",
        type="primary",
        use_container_width=True,
        key=f"confirm_{view.quiz_id}",
    ):
        _submit_attempt(view, attempt)


def _format_submitted_answer(
    question: QuizQuestionView,
    answer: str | None,
) -> str:
    if answer is None or not str(answer).strip():
        return "No answer submitted."
    text = str(answer)
    if question.question_type == "mcq" and question.options:
        try:
            option_index = int(text)
        except ValueError:
            return html.escape(text)
        if 0 <= option_index < len(question.options):
            return (
                f"{chr(65 + option_index)}. "
                f"{html.escape(question.options[option_index])}"
            )
    return html.escape(text)


def _render_active_quiz(view: QuizView, attempt: QuizAttempt) -> None:
    total = len(view.questions)
    index = int(st.session_state.get("quiz_index", 0))
    if index < 0 or index >= total:
        index = 0
        st.session_state["quiz_index"] = 0
    question = view.questions[index]
    answered = answered_count(view, attempt)

    _render_page_header(
        "Practice / Quiz",
        "Take your quiz",
        "Answer one question at a time. Answers save automatically as you "
        "move between questions.",
    )
    _render_quiz_summary(view)
    st.progress(
        answered / total,
        text=f"{answered} of {total} questions answered",
    )

    with st.container(border=True):
        badges = (
            f'<span class="badge badge-type">'
            f"{QUESTION_TYPE_LABELS[question.question_type]}</span>"
            f'<span class="badge badge-{question.difficulty}">'
            f"{question.difficulty.title()}</span>"
            f'<span class="badge badge-neutral">'
            f"Question {index + 1} of {total}</span>"
        )
        st.markdown(
            f'<div class="quiz-meta">{badges}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="question-text">'
            f"{html.escape(question.question_text)}</div>",
            unsafe_allow_html=True,
        )
        _render_answer_widget(view, attempt, question)

    prev_col, jump_col, next_col = st.columns([1, 2, 1], gap="small")
    jump_key = f"quiz_jump_{view.quiz_id}"
    if jump_key not in st.session_state:
        st.session_state[jump_key] = index + 1

    if prev_col.button(
        "← Previous",
        disabled=index == 0,
        use_container_width=True,
        key=f"prev_{view.quiz_id}",
    ):
        st.session_state["quiz_index"] = index - 1
        st.session_state[jump_key] = index
        st.session_state.pop("submit_confirmation", None)
        st.rerun()
    if next_col.button(
        "Next →",
        disabled=index == total - 1,
        use_container_width=True,
        key=f"next_{view.quiz_id}",
    ):
        st.session_state["quiz_index"] = index + 1
        st.session_state[jump_key] = index + 2
        st.session_state.pop("submit_confirmation", None)
        st.rerun()

    selected = jump_col.selectbox(
        "Jump to question",
        options=list(range(1, total + 1)),
        key=jump_key,
        format_func=lambda value: f"Question {value} of {total}",
        label_visibility="collapsed",
    )
    if selected - 1 != index:
        st.session_state["quiz_index"] = selected - 1
        st.session_state.pop("submit_confirmation", None)
        st.rerun()

    _render_submit_controls(view, attempt)


def _render_result_card(number: int, result: QuestionResult) -> None:
    status_label = RESULT_STATUS_LABELS[result.status]
    status_class = RESULT_STATUS_CLASSES[result.status]
    student_class = {
        "correct": "student-correct",
        "partial": "student-partial",
        "incorrect": "student-incorrect",
        "unanswered": "student-neutral",
        "ungraded": "student-neutral",
    }[result.status]
    grading_label = (
        "Exact match" if result.exact_match else "AI-graded · approximate"
    )
    points_label = (
        "Excluded from score"
        if result.status == "ungraded"
        else f"{result.points_awarded:.1f} / {result.max_points:.1f} points"
    )

    parts = [
        '<div class="result-card">',
        '<div class="quiz-meta">',
        f'<span class="badge {status_class}">{status_label}</span>',
        f'<span class="badge badge-type">'
        f"{QUESTION_TYPE_LABELS[result.question_type]}</span>",
        f'<span class="badge badge-{result.difficulty}">'
        f"{result.difficulty.title()}</span>",
        f'<span class="badge badge-neutral">{grading_label}</span>',
        f'<span class="badge badge-neutral">{points_label}</span>',
        "</div>",
        f'<div class="result-question">{number}. '
        f"{html.escape(result.question_text)}</div>",
        '<div class="result-answer-grid">',
        "<div>",
        '<div class="result-label">Your answer</div>',
        f'<div class="result-value {student_class}">'
        f"{html.escape(result.student_answer_display)}</div>",
        "</div>",
        "<div>",
        '<div class="result-label">Correct answer</div>',
        f'<div class="result-value">'
        f"{html.escape(result.correct_answer_display)}</div>",
        "</div>",
        "</div>",
    ]

    if result.accepted_answers and result.question_type == "fill_blank":
        alternatives = ", ".join(
            html.escape(answer) for answer in result.accepted_answers
        )
        parts.append(
            f'<div class="result-sources"><strong>Also accepted:</strong> '
            f"{alternatives}</div>"
        )

    if result.key_points and result.question_type == "short_answer":
        points = "".join(
            f"<li>{html.escape(point)}</li>" for point in result.key_points
        )
        parts.append(
            '<div class="result-keypoints">'
            '<div class="result-label">Key points</div>'
            f"<ul>{points}</ul></div>"
        )

    parts.append(
        '<div class="result-explanation">'
        f"<strong>Explanation:</strong> {html.escape(result.explanation)}"
        "</div>"
    )

    if result.grading_note:
        note_label = (
            "AI grading note" if not result.exact_match else "Grading note"
        )
        parts.append(
            f'<div class="result-note"><strong>{note_label}:</strong> '
            f"{html.escape(result.grading_note)}</div>"
        )

    if result.source_references:
        sources = "<br>".join(
            html.escape(reference) for reference in result.source_references
        )
        parts.append(
            '<div class="result-sources"><strong>Sources:</strong><br>'
            f"{sources}</div>"
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_unscored_submitted_quiz(
    view: QuizView,
    attempt: QuizAttempt,
) -> None:
    _render_page_header(
        "Practice / Results",
        "Quiz submitted",
        "Review your saved answers or retry scoring.",
    )
    st.warning(
        "Scoring could not run automatically. Retry scoring or review your "
        "saved answers below."
    )
    if st.button(
        "Retry scoring",
        type="primary",
        use_container_width=True,
        key="retry_scoring",
    ):
        st.session_state.pop("quiz_result", None)
        st.rerun()

    total = len(view.questions)
    unanswered = unanswered_question_ids(view, attempt)
    answered = total - len(unanswered)
    col1, col2, col3 = st.columns(3)
    col1.metric("Answered", answered)
    col2.metric("Unanswered", len(unanswered))
    submitted_at = (
        attempt.submitted_at.strftime("%H:%M UTC")
        if attempt.submitted_at is not None
        else "—"
    )
    col3.metric("Submitted", submitted_at)
    st.progress(answered / total, text=f"{answered} of {total} answered")

    with st.expander("Review your submitted answers", expanded=True):
        for number, question in enumerate(view.questions, start=1):
            answer_html = _format_submitted_answer(
                question,
                attempt.answers.get(question.question_id),
            )
            st.markdown(
                f'<div class="answer-card">'
                f'<div class="q">{number}. '
                f"{html.escape(question.question_text)}</div>"
                f'<div class="a">{answer_html}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    if st.button(
        "Start a new quiz",
        type="primary",
        use_container_width=True,
        key="new_quiz_after_unscored_submit",
    ):
        _reset_quiz()


def _render_submitted_quiz(
    view: QuizView,
    attempt: QuizAttempt,
    result: QuizResult | None,
) -> None:
    if result is None:
        _render_unscored_submitted_quiz(view, attempt)
        return

    _render_page_header(
        "Practice / Results",
        "Quiz results",
        "Your score, explanations, and source references are below.",
    )
    st.markdown(
        f"""
        <div class="quiz-hero">
            <div class="app-kicker">Quiz results</div>
            <h2>{result.percentage:.0f}%</h2>
            <p>{result.score_points:.1f} of {result.max_points:.1f} points ·
            {result.correct_count} correct · {result.partial_count} partial ·
            {result.incorrect_count} incorrect ·
            {result.unanswered_count} unanswered</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Correct", result.correct_count)
    metrics[1].metric("Partial credit", result.partial_count)
    metrics[2].metric("Incorrect", result.incorrect_count)
    metrics[3].metric("Unanswered", result.unanswered_count)

    st.progress(
        min(max(result.percentage / 100.0, 0.0), 1.0),
        text=(
            f"{result.percentage:.0f}% based on "
            f"{_question_count_label(result.graded_count)} graded"
        ),
    )
    st.caption(
        "Multiple-choice and fill-in-the-blank answers are graded by exact "
        "match. Short answers are graded approximately by AI against the "
        "reference answer and key points."
    )

    if result.ungraded_count:
        st.warning(
            f"{result.ungraded_count} answered short-answer question(s) could "
            "not be graded automatically and are excluded from the score. "
            "Review them manually, configure the LLM, or retry grading."
        )

    if result.partial_count:
        st.info(
            "Partial credit means a short answer covered some, but not all, "
            "of the key points."
        )

    if attempt.submitted_at is not None:
        st.caption(
            "Submitted at "
            f"{attempt.submitted_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    with st.expander(
        "Review answers, explanations, and sources",
        expanded=True,
    ):
        for number, question_result in enumerate(result.results, start=1):
            _render_result_card(number, question_result)

    if result.ungraded_count:
        if st.button(
            "Retry short-answer grading",
            use_container_width=True,
            key="retry_short_answer_grading",
        ):
            st.session_state.pop("quiz_result", None)
            st.rerun()

    if st.button(
        "Start a new quiz",
        type="primary",
        use_container_width=True,
        key="new_quiz_after_submit",
    ):
        _reset_quiz()


def _reset_quiz() -> None:
    view = st.session_state.get("active_quiz")
    if isinstance(view, QuizView):
        _quiz_vault().discard(view.quiz_id)
    for key in (
        "active_quiz",
        "quiz_attempt",
        "quiz_index",
        "submit_confirmation",
        "quiz_result",
        "quiz_entry_choice",
    ):
        st.session_state.pop(key, None)
    _clear_quiz_widget_state()
    st.session_state["_next_nav"] = NAV_GENERATE
    st.rerun()


def _render_quiz_tab() -> None:
    view = st.session_state.get("active_quiz")
    attempt = st.session_state.get("quiz_attempt")

    if not isinstance(view, QuizView):
        st.session_state.pop("quiz_result", None)
        _render_quiz_empty_state()
        return

    if _quiz_vault().get(view.quiz_id) is None:
        for key in (
            "active_quiz",
            "quiz_attempt",
            "quiz_index",
            "submit_confirmation",
            "quiz_result",
            "quiz_entry_choice",
        ):
            st.session_state.pop(key, None)
        st.warning(
            "This quiz is no longer available because the app restarted. "
            "Generate a new quiz."
        )
        _render_quiz_empty_state()
        return

    if not isinstance(attempt, QuizAttempt) or attempt.quiz_id != view.quiz_id:
        attempt = new_attempt(view)
        st.session_state["quiz_attempt"] = attempt
        st.session_state.pop("quiz_result", None)
        st.session_state.pop("quiz_entry_choice", None)

    if (
        not attempt.submitted
        and not attempt.answers
        and st.session_state.get("quiz_entry_choice") not in {"chat", "quiz"}
    ):
        st.session_state.pop("quiz_result", None)
        _render_quiz_choice(view)
    elif attempt.submitted:
        result = st.session_state.get("quiz_result")
        if not isinstance(result, QuizResult) or result.quiz_id != view.quiz_id:
            with st.spinner("Scoring your answers ..."):
                result = _score_attempt(view, attempt)
            if result is not None:
                st.session_state["quiz_result"] = result
        _render_submitted_quiz(view, attempt, result)
    else:
        # The answer key and result must never be in session state pre-submit.
        st.session_state.pop("quiz_result", None)
        _render_active_quiz(view, attempt)


def _render_generate_tab(
    settings: Settings,
    resources: list[ResourceInfo],
    retriever: Retriever,
) -> None:
    _render_page_header(
        "Practice / Generate",
        "Build a practice quiz",
        "Choose your sources and question mix. StudyMate retrieves relevant "
        "passages and writes a validated quiz grounded in them.",
    )

    if not resources:
        _render_empty_state(
            "📚",
            "Add material first",
            "Index at least one PDF, set of notes, or YouTube lecture in the "
            "Library before generating a quiz.",
            button_label="Go to Library",
            nav=NAV_LIBRARY,
            key="go_library_from_generate",
        )
        return

    if not settings.llm_configured:
        st.warning(
            "Set `LLM_API_KEY` and `LLM_MODEL` in `.env` to enable generation."
        )
        return

    total_chunks = sum(item.chunk_count for item in resources)
    _render_summary_strip(
        [
            ("Resources", str(len(resources))),
            ("Chunks", str(total_chunks)),
            ("Question limit", str(settings.max_questions)),
            ("Embedding", settings.embedding_model.split("/")[-1]),
        ]
    )

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Quiz settings")
        _render_generation_form(settings, resources)

    with right:
        view = st.session_state.get("active_quiz")
        attempt = st.session_state.get("quiz_attempt")
        if (
            isinstance(view, QuizView)
            and isinstance(attempt, QuizAttempt)
            and attempt.quiz_id == view.quiz_id
        ):
            st.subheader("Current quiz")
            _render_quiz_ready_card(view, attempt)
        else:
            st.subheader("What you'll get")
            st.markdown(
                """
                <div class="side-panel">
                    <ul>
                        <li>Questions grounded in your selected resources.</li>
                        <li>Validated JSON with duplicate and source checks.</li>
                        <li>A hidden answer key until you submit.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Tip: 5–10 questions is a good practice set.")

    st.divider()
    _render_retrieval_preview(settings, resources, retriever)

    with st.expander("LLM connection", expanded=False):
        st.caption(
            f"Groq · `{settings.llm_model}` · "
            f"reasoning `{settings.llm_reasoning_effort}`"
        )
        if st.button("Test LLM connection", type="secondary"):
            try:
                client = _llm_client()
                with st.spinner("Contacting Groq ..."):
                    reply = client.check_connection()
            except StudyMateError as exc:
                logger.warning(
                    "LLM connection test failed: %s",
                    exc.detail or exc,
                )
                st.error(exc.user_message)
            except Exception:
                logger.exception("Unexpected LLM connection test failure")
                st.error("The LLM connection test failed. Check the logs.")
            else:
                st.success("Groq connection works.")
                st.code(reply)


def main() -> None:
    _inject_styles()

    try:
        settings = get_settings()
        settings.ensure_directories()
    except StudyMateError as exc:
        st.title("📚 StudyMate AI")
        st.error(exc.user_message)
        st.stop()

    try:
        store, indexer, retriever = _services()
    except StudyMateError as exc:
        st.title("📚 StudyMate AI")
        st.error(exc.user_message)
        st.stop()
    except Exception:
        logger.exception("Could not initialise application services")
        st.title("📚 StudyMate AI")
        st.error("The application could not start. Check the logs for details.")
        st.stop()

    # Remove the Phase 3 development-preview key if an old session still has it.
    st.session_state.pop("generated_quiz", None)

    resources = _load_resources(store)
    _render_sidebar(settings, resources, retriever)

    requested_nav = st.session_state.pop("_next_nav", None)
    if requested_nav in NAV_OPTIONS:
        st.session_state["main_nav"] = requested_nav

    active_quiz = st.session_state.get("active_quiz")
    if isinstance(active_quiz, QuizView):
        default_nav = NAV_QUIZ
    elif resources:
        default_nav = NAV_GENERATE
    else:
        default_nav = NAV_LIBRARY

    if st.session_state.get("main_nav") not in NAV_OPTIONS:
        st.session_state["main_nav"] = default_nav

    nav = st.segmented_control(
        "Navigate",
        options=NAV_OPTIONS,
        key="main_nav",
        label_visibility="collapsed",
    )
    if nav not in NAV_OPTIONS:
        nav = default_nav

    _render_flash()

    if nav == NAV_LIBRARY:
        _render_library(settings, indexer, resources)
    elif nav == NAV_GENERATE:
        _render_generate_tab(settings, resources, retriever)
    elif nav == NAV_CHAT:
        _render_chat_tab(settings, resources, retriever)
    else:
        _render_quiz_tab()

    st.divider()
    st.caption(
        "StudyMate AI · Local embeddings · Grounded chat and quizzes"
    )


if __name__ == "__main__":
    main()
