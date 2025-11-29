import streamlit as st
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import numpy as np

import os

# Load KB
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "mimic-iv-ext-direct-1.0.0", "My_dataset", "combined_rag_data.json")

# Fallback for different folder structures (e.g. if running from root vs subfolder)
if not os.path.exists(data_path):
    # Try looking in the current directory if the file was moved
    if os.path.exists("combined_rag_data.json"):
        data_path = "combined_rag_data.json"
    else:
        # Try the original path the user might have intended if the folder structure is different
        data_path = os.path.join("mimic-iv-ext-direct-1.0.0", "My_dataset", "combined_rag_data.json")

with open(data_path, "r") as f:
    data = json.load(f)

# Filter to only include Knowledge Base items
kb_items = [item for item in data if "medicalKB" in item]

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
kb_embeddings = model.encode([f"{item['id']} : {item['medicalKB']}" for item in kb_items], convert_to_tensor=True)

# Load LLM
# LLM is now Gemini API, configured dynamically


def retrieve(query, k=4):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, kb_embeddings)[0].cpu().numpy()
    top_idx = np.argpartition(-scores, range(min(k,len(scores))))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(kb_items[i], float(scores[i])) for i in top_idx]

def generate_answer(query, api_key):
    if not api_key:
        return "Please enter your Google Gemini API Key in the sidebar."
    
    genai.configure(api_key=api_key)
    retrieved = retrieve(query)
    context_str = "\n".join([f"- [{item['id']}] {item['medicalKB']} (Relevance: {score:.2f})"
                             for item, score in retrieved])
    
    # Hidden instructions added to the prompt
    system_instruction = """First explain what the disease could be then give all the reasons.
Use headings or bullet points where needed."""
    
    prompt = f"You are a medical expert.\n{context_str}\nPatient case:\n{query}\n\n{system_instruction}"
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit interface
st.title("Medical Assistance using RAG")

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")


# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.subheader("Try an Example Case:")

# Create a clean grid for examples
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1: NSTEMI", use_container_width=True):
        st.session_state.input_text = """Explain the symptoms of a 60-year-old male with sudden chest pain radiating to the back.
Vitals: BP 160/90 mmHg, HR 100 bpm, RR 20/min."""
    
    if st.button("Example 4: Acute Appendicitis", use_container_width=True):
        st.session_state.input_text = """A 25-year-old male complains of periumbilical pain migrating to the right lower quadrant, associated with nausea and low-grade fever.
Vitals: Temp 38.5°C, HR 95 bpm. Labs: WBC 14,000/mm³."""

with col2:
    if st.button("Example 2: Pulmonary Embolism", use_container_width=True):
        st.session_state.input_text = """A 55-year-old female presents with sudden onset shortness of breath and pleuritic chest pain. She recently returned from a long-haul flight.
Vitals: O2 sat 88% on room air, HR 110 bpm, BP 110/70 mmHg."""

    if st.button("Example 5: Ischemic Stroke", use_container_width=True):
        st.session_state.input_text = """A 70-year-old female presents with sudden right-sided weakness, facial droop, and slurred speech. History of atrial fibrillation.
Vitals: BP 180/100 mmHg, HR 88 bpm (irregular). Symptoms started 2 hours ago."""

with col3:
    if st.button("Example 3: Diabetes Type 2", use_container_width=True):
        st.session_state.input_text = """A 45-year-old male presents with increased thirst, frequent urination, and unexplained weight loss.
Labs: Fasting Plasma Glucose 140 mg/dL, HbA1c 7.5%. BMI 32."""

    if st.button("Example 6: COPD Exacerbation", use_container_width=True):
        st.session_state.input_text = """A 65-year-old male with a history of smoking presents with worsening dyspnea, increased sputum production, and wheezing.
Vitals: RR 24/min, O2 sat 90% on room air. Lung exam: Diffuse wheezing."""

user_input = st.text_area("Enter patient case:", value=st.session_state.input_text, height=200)

if st.button("Diagnose"):
    if user_input.strip():
        answer = generate_answer(user_input, api_key)
        st.subheader("Answer")
        st.write(answer)
