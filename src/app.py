#!/usr/bin/env python3
"""
app.py

Production-ready Streamlit UI with argument-based configuration:
- Accepts CLI args for paths, models, top-K, and OpenAI API key
- Loads index and metadata accordingly
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(
    description="Streamlit RAG demo with CLI-configurable options"
)
parser.add_argument('--index-path',    default='output/thebatch.faiss', help='Path to FAISS index file')
parser.add_argument('--meta-path',     default='output/meta.json',   help='Path to metadata JSON')
parser.add_argument('--embed-model',   default='text-embedding-ada-002', help='OpenAI embedding model')
parser.add_argument('--llm-model',     default='gpt-4.1-nano',        help='OpenAI chat model')
parser.add_argument('--top-k',         type=int, default=5,           help='Number of top results to retrieve')
parser.add_argument('--openai-key',    default=None,                 help='OpenAI API key (overrides env)')
# parse known args to allow Streamlit flags
args, _ = parser.parse_known_args()

# ---------------------- Configuration & Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

INDEX_PATH = args.index_path
META_PATH  = args.meta_path
EMBED_MODEL = args.embed_model
LLM_MODEL   = args.llm_model
TOP_K       = args.top_k

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not provided via env")
    st = None  # avoid Streamlit load error
    raise RuntimeError("Missing OpenAI API key.")

# Initialize OpenAI client
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Load Index & Metadata ----------------------
try:
    faiss_index = faiss.read_index(INDEX_PATH)
    logger.info("Loaded FAISS index from %s", INDEX_PATH)
except Exception as e:
    logger.error("Failed to load FAISS index: %s", e)
    raise

try:
    metadata = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    logger.info("Loaded metadata from %s (n=%d)", META_PATH, len(metadata))
except Exception as e:
    logger.error("Failed to load metadata: %s", e)
    raise

# ---------------------- Helper Functions ----------------------
def embed_query(text: str) -> np.ndarray:
    resp = oai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")


def retrieve(query: str) -> List[Tuple[dict, float]]:
    q_emb = embed_query(query).reshape(1, -1)
    fetch_k = TOP_K * 3
    D, I = faiss_index.search(q_emb, fetch_k)

    results, seen_urls = [], set()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        url = m.get("url")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append((m, float(score)))
        if len(results) >= TOP_K:
            break
    return results


def generate_answer(snippets: List[str], question: str) -> str:
    context = "\n\n".join(f"Snippet {i+1}: {s}" for i, s in enumerate(snippets))
    prompt = f"{context}\n\nQ: {question}\nA:"
    resp = oai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="The Batch RAG Demo", layout="wide")
st.title("📰 The Batch RAG Demo")

query: Optional[str] = st.text_input("Enter a question about AI news:")
if query:
    with st.spinner("Retrieving relevant snippets..."):
        hits = retrieve(query)

    snippets = [m.get("text","") for m,_ in hits]
    with st.spinner("Generating answer..."):
        answer = generate_answer(snippets, query)

    st.header("Answer")
    st.write(answer)

    st.header("Sources")
    for i,(m,score) in enumerate(hits, start=1):
        title,url,date = m.get("title",""), m.get("url",""), m.get("date","")
        st.markdown(f"**{i}. [{title}]({url})** — {date} *(sim={1-score:.4f})*")
        img = m.get("img_path")
        if img and Path(img).exists(): st.image(img, width=300)
        st.write(m.get("text",""))
        st.markdown("---")
