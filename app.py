import streamlit as st
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import google.generativeai as genai
import torch

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "FAISS_File", "medical_kb_index.faiss")
DATA_PATH = os.path.join(BASE_DIR, "mimic-iv-ext-direct-1.0.0/My_dataset", "combined_rag_data.json")

# ---------------------------
# Load KB
# ---------------------------
with open(DATA_PATH, "r") as f:
    data = json.load(f)

kb_items = [item for item in data if "medicalKB" in item]

# ---------------------------
# Load embedding model (CPU)
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# Load FAISS index
# ---------------------------
faiss_index = faiss.read_index(FAISS_PATH)

# ---------------------------
# FAISS retrieval
# ---------------------------
def retrieve(query, k=2):
    q_emb = model.encode([query], convert_to_tensor=True)
    q_emb_np = q_emb.cpu().detach().numpy().astype("float32")
    faiss.normalize_L2(q_emb_np)
    scores, idx = faiss_index.search(q_emb_np, k)
    top_results = [(kb_items[i], float(scores[0][j])) for j, i in enumerate(idx[0])]
    return top_results

# ---------------------------
# Generate answer using Gemini API
# ---------------------------
def generate_answer(query, api_key):
    if not api_key:
        return "Please enter your Google Gemini API Key in the sidebar.", [], 0.0
    
    genai.configure(api_key=api_key)
    retrieved = retrieve(query, k=2)

    context_str = "\n".join([f"- [{item['id']}] {item['medicalKB']} (Relevance: {score:.2f})"
                             for item, score in retrieved])
    
    avg_relevance_score = sum(score for _, score in retrieved) / len(retrieved) if retrieved else 0.0

    system_instruction = """First explain what the disease could be then give all the reasons.
Use headings or bullet points where needed.
IMPORTANT: Do NOT include any patient names or sensitive personal information in your response. Focus only on medical analysis."""

    prompt = f"You are a medical expert.\n{context_str}\nPatient case:\n{query}\n\n{system_instruction}"

    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        response = gemini_model.generate_content(prompt)
        return response.text, retrieved, avg_relevance_score
    except Exception as e:
        return f"Error generating response: {str(e)}", [], 0.0

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Medical Assistance using RAG with FAISS")

# Load .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.subheader("Try an Example Case:")

# Example buttons in a 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1: NSTEMI"):
        st.session_state.input_text = """A 60-year-old male is suffering from sudden chest pain radiating to the back.
Vitals: BP 160/90 mmHg, HR 100 bpm, RR 20/min."""
    if st.button("Example 4: Acute Appendicitis"):
        st.session_state.input_text = """A 25-year-old male complains of periumbilical pain migrating to the right lower quadrant, associated with nausea and low-grade fever.
Vitals: Temp 38.5°C, HR 95 bpm. Labs: WBC 14,000/mm³."""

with col2:
    if st.button("Example 2: Pulmonary Embolism"):
        st.session_state.input_text = """A 55-year-old female presents with sudden onset shortness of breath and pleuritic chest pain. She recently returned from a long-haul flight.
Vitals: O2 sat 88% on room air, HR 110 bpm, BP 110/70 mmHg."""
    if st.button("Example 5: Ischemic Stroke"):
        st.session_state.input_text = """A 70-year-old female presents with sudden right-sided weakness, facial droop, and slurred speech. History of atrial fibrillation.
Vitals: BP 180/100 mmHg, HR 88 bpm (irregular). Symptoms started 2 hours ago."""

with col3:
    if st.button("Example 3: Diabetes Type 2"):
        st.session_state.input_text = """A 45-year-old male presents with increased thirst, frequent urination, and unexplained weight loss.
Labs: Fasting Plasma Glucose 140 mg/dL, HbA1c 7.5%. BMI 32."""
    if st.button("Example 6: COPD Exacerbation"):
        st.session_state.input_text = """A 65-year-old male with a history of smoking presents with worsening dyspnea, increased sputum production, and wheezing.
Vitals: RR 24/min, O2 sat 90% on room air. Lung exam: Diffuse wheezing."""

user_input = st.text_area("Enter patient case:", value=st.session_state.input_text, height=200)

# Privacy statement
st.caption("❕Privacy Notice: Please avoid entering patient names or any sensitive personal information.")

if st.button("Diagnose"):
    if user_input.strip():
        with st.spinner("Analyzing patient case..."):
            answer, retrieved_items, relevance_score = generate_answer(user_input, api_key)
        
        if retrieved_items:
            st.subheader("Retrieved Context (Top-2 RAG)")
            for item, score in retrieved_items:
                with st.expander(f"Relevance: {score:.4f} - {item['id']}"):
                    st.write(item['medicalKB'])
        
        st.subheader("Answer")
        st.write(answer)
        
        st.divider()
        st.metric(label="Overall Relevance Score", value=f"{relevance_score:.2%}")
