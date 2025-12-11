import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from functools import lru_cache

from theme import apply_theme

apply_theme()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="🦉 BBO RAG Chatbot", layout="wide")
st.title("🦉 BBO Owl Migration RAG Chatbot")
st.write("Ask anything about the owls, migration patterns, ports, SNR, or tag behavior.")


# ============================================================
# 1. Load Models (cached for speed)
# ============================================================
@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large")


# ============================================================
# 2. Load BBO Data
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/detections_master.csv")

    # Ensure datetime components exist
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    return df

detections_master = load_data()


# ============================================================
# 3. Prepare Documents (Knowledge Base)
# ============================================================
def prepare_documents(df):

    overview = """
    This project analyzes owl migration at Beaverhill Bird Observatory (BBO).
    We study antenna ports, directions, SNR, timestamps, and movement patterns.
    The goal is to understand migration timing, direction, and tag performance.
    """

    summary_text = f"""
    Total detections: {len(df)}
    Unique tags: {df['motusTagID'].nunique()}
    Peak month: {df['month'].mode()[0]}
    Top used port: {df['port'].mode()[0]}
    Peak detection hour: {df['hour'].mode()[0]}
    """

    documents = {
        "overview": overview,
        "summary": summary_text,
    }

    # Owl-specific narratives
    owls = df["motusTagID"].unique()
    for owl in owls:
        owldf = df[df["motusTagID"] == owl]

        owl_text = f"""
        Owl {owl} summary:
        Total detections: {len(owldf)}
        First detection: {owldf['date'].min()}
        Last detection: {owldf['date'].max()}
        Most used port: {owldf['port'].mode()[0]}
        Peak hour: {owldf['hour'].mode()[0]}
        Average SNR: {round(owldf['snr'].mean(),2)}
        """

        documents[f"owl_{owl}"] = owl_text

    return documents

documents = prepare_documents(detections_master)


# ============================================================
# 4. Embed Documents
# ============================================================
def build_embeddings(documents):
    embedder = load_embedder()
    embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }
    return embeddings

embeddings = build_embeddings(documents)


# ============================================================
# 5. Retrieve Context
# ============================================================
def retrieve_context(query, documents, embeddings, top_k=2):
    embedder = load_embedder()
    q_emb = embedder.encode(query, convert_to_tensor=True)

    scores = {
        doc_id: util.pytorch_cos_sim(q_emb, emb).item()
        for doc_id, emb in embeddings.items()
    }

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in sorted_docs[:top_k]]

    context = "\n\n".join(documents[d] for d in top_ids)
    return context


# ============================================================
# 6. Generate Answer
# ============================================================
def generate_answer(query, context):
    generator = load_generator()

    prompt = (
        "You are an assistant for analyzing BBO owl migration.\n"
        "Use the context below and answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    output = generator(prompt, max_new_tokens=150, temperature=0.7)
    return output[0]["generated_text"]


# ============================================================
# 7. Main RAG Chatbot Function
# ============================================================
def rag_chatbot(query):
    ctx = retrieve_context(query, documents, embeddings)
    ans = generate_answer(query, ctx)
    return ans


# ============================================================
# 8. Streamlit UI
# ============================================================

st.subheader("💬 Ask the BBO Chatbot")

preset_questions = [
    "Summarize overall owl migration patterns.",
    "Which direction do owls usually fly?",
    "Which owl had the most detections?",
    "Explain antenna port usage.",
    "What are the peak migration hours?",
    "Summarize owl 80821.",
    "Summarize owl 80805.",
]

selected_question = st.selectbox("Choose a preset question:", [""] + preset_questions)
custom_question = st.text_input("Or type your own question:")

query = custom_question if custom_question else selected_question

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please select or enter a question.")
    else:
        with st.spinner("Analyzing BBO data..."):
            answer = rag_chatbot(query)

        st.write("### 🟩 Answer:")
        st.success(answer)
