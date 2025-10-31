"""Standalone Streamlit page to display PDF sources referenced by the RAG chatbot."""

import base64
from pathlib import Path
from urllib.parse import unquote

import streamlit as st
from streamlit.components.v1 import html as components_html

BASE_DIR = Path(__file__).parent.parent
PAPERS_DIR = (BASE_DIR / "data" / "papers").resolve()

st.set_page_config(
    page_title="Source Viewer",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_pdf_bytes(path_str: str) -> bytes:
    """Read a PDF file once and cache the bytes for subsequent requests."""
    path = Path(path_str)
    return path.read_bytes()


def _get_first(values):
    if not values:
        return None
    if isinstance(values, (list, tuple)):
        return values[0]
    return values


# Get query parameters
try:
    raw_paper = st.query_params.get("paper", None)
    raw_page = st.query_params.get("page", None)
except Exception:
    # Fallback for older Streamlit versions
    query_params = st.experimental_get_query_params()
    raw_paper = _get_first(query_params.get("paper"))
    raw_page = _get_first(query_params.get("page"))

if not raw_paper:
    st.error("Missing 'paper' query parameter.")
    st.stop()

paper_rel_str = unquote(raw_paper).strip()
if not paper_rel_str:
    st.error("Invalid paper path.")
    st.stop()

paper_rel = Path(paper_rel_str)

if paper_rel.is_absolute() or any(part == ".." for part in paper_rel.parts):
    st.error("Invalid paper path.")
    st.stop()

try:
    candidate_path = (PAPERS_DIR / paper_rel).resolve(strict=True)
except FileNotFoundError:
    st.error("Requested paper was not found.")
    st.stop()

if not candidate_path.is_file():
    st.error("Requested path is not a file.")
    st.stop()

if candidate_path.suffix.lower() != ".pdf":
    st.error("Only PDF sources are supported.")
    st.stop()

if PAPERS_DIR not in candidate_path.parents:
    st.error("Requested paper is outside the papers directory.")
    st.stop()

page_number = 1
if raw_page:
    try:
        page_number = max(1, int(raw_page))
    except ValueError:
        st.warning("Unable to parse page number. Defaulting to page 1.")
        page_number = 1

st.title(candidate_path.name)
st.caption(f"Source located at: {candidate_path.relative_to(PAPERS_DIR)} • Viewing page {page_number}")

with st.spinner("Loading PDF source..."):
    pdf_bytes = load_pdf_bytes(str(candidate_path))

# Encode PDF as base64
encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

# Create iframe with PDF viewer
iframe_html = f"""
    <iframe
        src="data:application/pdf;base64,{encoded_pdf}#page={page_number}"
        width="100%"
        height="900"
        style="border: none;"
        type="application/pdf"
    ></iframe>
"""
components_html(iframe_html, height=900, scrolling=True)

# Download button
st.download_button(
    label="📥 Download PDF",
    data=pdf_bytes,
    file_name=candidate_path.name,
    mime="application/pdf",
)

st.markdown("[← Back to RAG Chatbot](./RAG_Chatbot)")
