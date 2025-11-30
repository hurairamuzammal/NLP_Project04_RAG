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
        st.session_state.input_text = """A female patient presented with dull aching pain in the back and chest that began yesterday while at rest. This pain is different from her usual sharp chest pain episodes and lasted significantly longer. The pain was not relieved by multiple sublingual nitroglycerin tablets taken at home. She reports ongoing shortness of breath for several weeks. She has a significant cardiac history including coronary artery disease, and multiple comorbidities: uncontrolled type 2 diabetes mellitus, hypertension, hypothyroidism, asthma, obstructive sleep apnea, and morbid obesity with BMI between 40-44.9. Her father had a myocardial infarction. On examination, vital signs showed blood pressure 186/71 mmHg, heart rate 88 beats per minute, temperature 98°F, respiratory rate 18 per minute, and oxygen saturation 98% on room air. Physical examination revealed distant heart sounds, regular rate and rhythm without murmurs, clear lungs bilaterally, soft obese abdomen, and 1+ bilateral pitting edema. Laboratory studies showed elevated troponin at 0.60, white blood cell count 9.3, hemoglobin 13.7, markedly elevated glucose at 339, and bicarbonate 24. Electrocardiogram demonstrated sinus rhythm at rate 62, QTc interval 456 milliseconds, with no new ischemic changes."""

with col2:
    if st.button("Case 2: NSTEMI with 3-Vessel Disease", use_container_width=True):
        st.session_state.input_text = """A 66-year-old male initially presented with urinary retention requiring Foley catheter placement, after which he developed profound weakness, hypotension, and diaphoresis. Over the past month, he has experienced intermittent weakness episodes and syncopal events. His exercise tolerance has significantly decreased, becoming short of breath with even brief periods of activity. Notably, he denied chest pain during this acute episode. He has chronic back pain and has not received medical care for the past decade. Vital signs at presentation were blood pressure 157/85 mmHg, heart rate 98 beats per minute, temperature 98.4°F, respiratory rate 16 per minute, and oxygen saturation 98%. Physical examination showed a comfortable-appearing patient with clear chest auscultation, regular cardiac rhythm, soft abdomen, no peripheral edema, and warm dry skin. Laboratory results revealed troponin elevation at 0.34, white blood cell count 10.8, hemoglobin 15.4, glucose 127, and hemoglobin A1c 5.5%. Electrocardiogram showed T wave inversions in the inferolateral leads with less than 1mm ST elevations in leads I and aVL, and Q waves anteriorly and inferiorly. Cardiac catheterization demonstrated severe three-vessel coronary artery disease with complete occlusion of the right coronary artery and diffuse disease in the left anterior descending and circumflex arteries. Echocardiography revealed severely reduced left ventricular ejection fraction of 15% with akinesis of the apex and severe regional left ventricular systolic dysfunction."""

with col3:
    if st.button("Case 3: NSTEMI with DM/AFib", use_container_width=True):
        st.session_state.input_text = """A male patient with known diabetes mellitus, atrial fibrillation, and hypertension presented with one day of epigastric pressure occurring at rest without radiation. The pain began while sitting and persisted until he fell asleep. Similar epigastric discomfort occurred the morning of admission after taking his daily medications. He experienced complete pain relief after taking aspirin 325mg at home. He denied exertional chest pain, shortness of breath, nausea, vomiting, or diaphoresis. Additionally, he reported having diarrhea for approximately one month. He is on both oral agents and insulin for diabetes management. His family history is significant for myocardial infarction in his father and diabetes in his mother. Vital signs on presentation were blood pressure 140/82 mmHg, heart rate 74 beats per minute, temperature 98.4°F, respiratory rate 21 per minute, and oxygen saturation 95% on room air. Physical examination revealed a patient in no acute distress with clear oropharynx, regular cardiac rhythm, clear lung fields bilaterally, soft non-tender abdomen, and no peripheral edema. Initial laboratory studies showed troponin 0.27 which subsequently rose to 1.77, white blood cell count 9.5, hemoglobin 15.0, elevated glucose at 286, and international normalized ratio 1.0. Electrocardiogram demonstrated ST segment elevation in lead III, T wave inversions in leads V4 through V6, and ST segment depressions in leads I and V6, findings consistent with acute myocardial ischemia. Cardiac catheterization revealed diffuse coronary artery disease with a heavily calcified right coronary artery showing severe ectasia and stenoses ranging from 60-70%. Echocardiography showed left ventricular ejection fraction of 50% with apical hypokinesis and focal apical dyskinesis."""

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
                st.subheader("Retrieved Knowledge Base Entries")
                st.caption(f"Showing top {len(retrieved_items)} most relevant medical knowledge entries based on FAISS similarity search")
                for idx, (item, score) in enumerate(retrieved_items, 1):
                    score_color = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                    with st.expander(f"{score_color} Entry {idx}: {item['id']} (Relevance Score: {score:.4f})", expanded=(idx==1)):
                        st.markdown(f"**Similarity Score:** `{score:.4f}` (FAISS cosine similarity)")
                        st.markdown("**Medical Knowledge:**")
                        st.write(item['medicalKB'])
        
        with col_score:
            st.metric(label="Overall Relevance Score", value=f"{relevance_score:.2%}")
        
        st.markdown("---")
        st.subheader("Diagnostic Analysis")
        st.write(answer)
    else:
        st.warning("Please enter a patient case to diagnose.")
