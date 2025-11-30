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
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS for better UI
# ---------------------------
st.markdown("""
    <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Title styling */
        h1 {
            color: #1E88E5;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        /* Subheader styling */
        h2, h3 {
            color: #424242;
            margin-top: 1.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            height: 40px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border: 1px solid #e0e0e0;
        }
        
        .stButton > button:hover {
            border-color: #1E88E5;
            color: #1E88E5;
            box-shadow: 0 2px 8px rgba(30, 136, 229, 0.15);
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            font-size: 0.95rem;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #1E88E5;
            box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.1);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f5f5f5;
            border-radius: 6px;
            font-weight: 500;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #1E88E5;
        }
        
        /* Privacy notice */
        .stCaption {
            color: #757575;
            font-size: 0.875rem;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Spacing improvements */
        .element-container {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "FAISS_File", "medical_kb_index.faiss")
DATA_PATH = os.path.join(BASE_DIR, "mimic-iv-ext-direct-1.0.0","My_dataset", "combined_rag_data.json")

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

    # If FAISS index has fewer vectors than k, adjust k
    actual_k = min(k, faiss_index.ntotal)
    scores, idx = faiss_index.search(q_emb_np, actual_k)
    
    top_results = []
    for j, i in enumerate(idx[0]):
        if i < len(kb_items):
            top_results.append((kb_items[i], float(scores[0][j])))
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

    system_instruction = """Structure your response as follows:

1. EXTRACTED KNOWLEDGE BASE INFORMATION:
   - Summarize the key relevant medical information from the provided context

2. DIAGNOSIS: [State the diagnosis clearly, e.g., "Acute Ischemic Stroke"]

3. REASONING:
   - Use the extracted KB information to support your diagnosis
   - Reference specific symptoms and vitals from the patient case
   - Use bullet points for clarity

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
st.markdown("#### AI-powered diagnostic assistant using Retrieval-Augmented Generation")
st.markdown("---")

# Load .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.subheader("Try an Example Case")
st.markdown("")  # Add spacing

# Example buttons in a 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Case 1: NSTEMI with CAD History", use_container_width=True):
        st.session_state.input_text = """Chief Complaint: Chest pain at rest
        
History: Female patient with dull ache in back and chest starting yesterday at rest, different from previous sharp pain episodes. Pain lasted longer than usual, associated with ongoing SOB for several weeks. Multiple SL nitroglycerin at home did not relieve pain.

Past Medical History: Coronary artery disease, Type 2 diabetes (uncontrolled), hypertension, hypothyroidism, asthma, sleep apnea, morbid obesity (BMI 40-44.9).

Family History: Father with MI

Vitals: BP 186/71 mmHg, HR 88 bpm, Temp 98°F, RR 18/min, O2 sat 98% RA

Physical Exam: Distant heart sounds, RRR no murmur, CTAB, soft obese abdomen, 1+ pitting edema bilaterally

Labs: Troponin 0.60, WBC 9.3, Hgb 13.7, Glucose 339, HCO3 24

EKG: Sinus rhythm, rate 62, QTC 456, no new ischemic changes"""

with col2:
    if st.button("Case 2: NSTEMI with 3-Vessel Disease", use_container_width=True):
        st.session_state.input_text = """Chief Complaint: Urinary retention, then profound weakness and hypotension

History: 66-year-old male developed profound weakness, hypotension, and diaphoresis after Foley placement. Intermittent weakness episodes and syncope over past month. Decreased exercise tolerance, becomes SOB with brief activity. No chest pain with this episode.

Past Medical History: Chronic back pain, no medical care for past decade

Family History: Noncontributory

Vitals: BP 157/85 mmHg, HR 98 bpm, Temp 98.4°F, RR 16/min, O2 sat 98%

Physical Exam: Comfortable, clear chest, regular heart rhythm, soft abdomen, no edema, warm and dry skin

Labs: Troponin 0.34, WBC 10.8, Hgb 15.4, Glucose 127, HbA1c 5.5

EKG: T wave inversions in inferolateral leads, <1mm ST elevations in I and aVL with Qs anteriorly and inferiorly

Cardiac Cath: 3 vessel disease with fully occluded RCA and LAD/Cx with diffuse disease

Echo: LVEF 15% with akinetic apex, severe regional LV dysfunction"""

with col3:
    if st.button("Case 3: NSTEMI with DM/AFib", use_container_width=True):
        st.session_state.input_text = """Chief Complaint: Epigastric pain

History: Male with diabetes, atrial fibrillation, and hypertension presenting with one day of epigastric pressure at rest, non-radiating. Pain started while sitting, lasted until sleep. Similar pain occurred morning of admission after taking medications. Complete relief with ASA 325mg at home. No exertional pain, SOB, nausea, or diaphoresis. Also reports diarrhea x1 month.

Past Medical History: Type II diabetes (on oral agents and insulin), atrial fibrillation, hypertension

Family History: Father MI, mother DM

Vitals: BP 140/82 mmHg, HR 74 bpm, Temp 98.4°F, RR 21/min, O2 sat 95% RA

Physical Exam: NAD, clear oropharynx, regular heart rhythm, CTAB, soft non-tender abdomen, no edema

Labs: Troponin 0.27 (later 1.77), WBC 9.5, Hgb 15.0, Glucose 286, INR 1.0

EKG: ST elevation in III, T wave inversions in V4-V6, ST depressions in I and V6, consistent with acute ischemia

Cardiac Cath: Diffuse CAD, heavily calcified RCA with severe ectasia and 60-70% stenoses

Echo: LVEF 50% with apical hypokinesis and focal apical dyskinesis"""

st.markdown("")  # Add spacing
st.markdown("### Enter Patient Case")
user_input = st.text_area("", value=st.session_state.input_text, height=200, placeholder="Describe the patient symptoms, vitals, and relevant medical history...")

# Privacy statement
st.caption("Privacy Notice: Please avoid entering patient names or any sensitive personal information.")

st.markdown("")  # Add spacing
if st.button("Diagnose", use_container_width=True, type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing patient case..."):
            answer, retrieved_items, relevance_score = generate_answer(user_input, api_key)
        
        st.markdown("---")
        
        # Create two columns for better layout
        col_rag, col_score = st.columns([3, 1])
        
        with col_rag:
            if retrieved_items:
                st.subheader("Retrieved Context (Top-2 from Knowledge Base)")
                for item, score in retrieved_items:
                    with st.expander(f"Relevance: {score:.4f} - {item['id']}"):
                        st.write(item['medicalKB'])
        
        with col_score:
            st.metric(label="Overall Relevance Score", value=f"{relevance_score:.2%}")
        
        st.markdown("---")
        st.subheader("Diagnostic Analysis")
        st.write(answer)
    else:
        st.warning("Please enter a patient case to diagnose.")
