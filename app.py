import streamlit as st
import json
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import google.generativeai as genai

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,"mimic-iv-ext-direct-1.0.0","My_Dataset","combined_rag_data.json")
KB_FAISS_PATH = os.path.join(BASE_DIR, "medical_kb_index.faiss")
CASES_FAISS_PATH = os.path.join(BASE_DIR, "patient_cases_index.faiss")
KB_PICKLE = os.path.join(BASE_DIR, "kb_items.pkl")
PATIENT_PICKLE = os.path.join(BASE_DIR, "patient_items.pkl")

# ---------------------------
# Load SentenceTransformer
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU embeddings

# ---------------------------
# Load or Create KB Items and Patient Items
# ---------------------------
if os.path.exists(KB_PICKLE) and os.path.exists(PATIENT_PICKLE):
    with open(KB_PICKLE, "rb") as f:
        kb_items = pickle.load(f)
    with open(PATIENT_PICKLE, "rb") as f:
        patient_items = pickle.load(f)
else:
    st.warning("Pickle files not found, loading from JSON and creating them...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    kb_items = [item for item in data if "medicalKB" in item]
    patient_items = [item for item in data if "patient_case" in item]
    
    # Save pickle files
    with open(KB_PICKLE, "wb") as f:
        pickle.dump(kb_items, f)
    with open(PATIENT_PICKLE, "wb") as f:
        pickle.dump(patient_items, f)

# ---------------------------
# Load or Build FAISS Index for Medical KB
# ---------------------------
if os.path.exists(KB_FAISS_PATH):
    kb_faiss_index = faiss.read_index(KB_FAISS_PATH)
else:
    st.warning("KB FAISS index not found, rebuilding...")
    kb_texts = [item["medicalKB"] for item in kb_items]
    kb_emb = model.encode(kb_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(kb_emb)
    dim = kb_emb.shape[1]
    kb_faiss_index = faiss.IndexFlatIP(dim)
    kb_faiss_index.add(kb_emb)
    faiss.write_index(kb_faiss_index, KB_FAISS_PATH)

# ---------------------------
# Load or Build FAISS Index for Patient Cases
# ---------------------------
if os.path.exists(CASES_FAISS_PATH):
    cases_faiss_index = faiss.read_index(CASES_FAISS_PATH)
else:
    st.warning("Cases FAISS index not found, rebuilding...")
    case_texts = [item["patient_case"] for item in patient_items]
    case_emb = model.encode(case_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(case_emb)
    dim = case_emb.shape[1]
    cases_faiss_index = faiss.IndexFlatIP(dim)
    cases_faiss_index.add(case_emb)
    faiss.write_index(cases_faiss_index, CASES_FAISS_PATH)

# ---------------------------
# Retrieval Function (Retrieve 2 from KB and 2 from Cases)
# ---------------------------
def retrieve(query, k_kb=2, k_cases=2):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    
    results = []
    
    # Retrieve from Medical KB
    actual_k_kb = min(k_kb, kb_faiss_index.ntotal)
    scores_kb, idx_kb = kb_faiss_index.search(q_emb, actual_k_kb)
    for j, i in enumerate(idx_kb[0]):
        if i >= 0 and i < len(kb_items):
            score = float(scores_kb[0][j])
            results.append(("KB", kb_items[i], score))
    
    # Retrieve from Patient Cases
    actual_k_cases = min(k_cases, cases_faiss_index.ntotal)
    scores_cases, idx_cases = cases_faiss_index.search(q_emb, actual_k_cases)
    for j, i in enumerate(idx_cases[0]):
        if i >= 0 and i < len(patient_items):
            score = float(scores_cases[0][j])
            results.append(("CASE", patient_items[i], score))
    
    return results

# ---------------------------
# Gemini API Function
# ---------------------------
def generate_answer(query, api_key, k_kb=2, k_cases=2):
    if not api_key:
        return "Please enter GEMINI_API_KEY in .env", [], 0.0

    genai.configure(api_key=api_key)
    retrieved = retrieve(query, k_kb=k_kb, k_cases=k_cases)

    context_parts = []
    for source_type, item, score in retrieved:
        if source_type == "KB":
            context_parts.append(f"- [KB: {item['id']}] {item['medicalKB']} (Relevance: {score:.2f})")
        else:
            context_parts.append(f"- [CASE: {item['id']}] {item['patient_case']} (Relevance: {score:.2f})")
    
    context_str = "\n".join(context_parts)
    avg_score = sum(score for _, _, score in retrieved)/len(retrieved) if retrieved else 0.0

    prompt = f"""
You are a medical expert.
Use the following knowledge base and similar patient cases to answer the patient query.

{context_str}

Patient Query:
{query}

Structure response:
1. EXTRACTED INFO: summarize key info from KB and cases
2. DIAGNOSIS: give clear diagnosis
3. REASONING: support diagnosis with bullets, reference symptoms
"""
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        response = gemini_model.generate_content(prompt)
        return response.text, retrieved, avg_score
    except Exception as e:
        return f"Error: {str(e)}", [], 0.0

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Medical RAG Assistant")
st.markdown("#### AI-powered Diagnosis using Retrieval-Augmented Generation (RAG)")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

user_input = st.text_area("Patient Case Input", height=200)

if st.button("Diagnose"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            answer, retrieved_items, avg_score = generate_answer(user_input, api_key)
        
        st.markdown("### Retrieved Knowledge & Similar Cases")
        for idx, (source_type, item, score) in enumerate(retrieved_items, 1):
            if source_type == "KB":
                st.markdown(f"**{idx}. [Medical KB] {item['id']}** - Score: {score:.2f}")
                st.write(item['medicalKB'])
            else:
                st.markdown(f"**{idx}. [Patient Case] {item['id']}** - Score: {score:.2f}")
                st.write(item['patient_case'])
        
        st.markdown("### AI Diagnosis")
        st.write(answer)

        st.markdown(f"**Average Relevance Score:** {avg_score:.2f}")
    else:
        st.warning("Enter patient case details.")
