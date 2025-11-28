import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    kb_items = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
kb_embeddings = model.encode([f"{item['id']} : {item['medicalKB']}" for item in kb_items], convert_to_tensor=True)

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", use_fast=False)
llm = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                           torch_dtype=torch.float16, device_map="auto")

def retrieve(query, k=4):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, kb_embeddings)[0].cpu().numpy()
    top_idx = np.argpartition(-scores, range(min(k,len(scores))))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(kb_items[i], float(scores[i])) for i in top_idx]

def generate_answer(query):
    retrieved = retrieve(query)
    context_str = "\n".join([f"- [{item['id']}] {item['medicalKB']} (Relevance: {score:.2f})"
                             for item, score in retrieved])
    prompt = f"You are a medical expert.\n{context_str}\nPatient case:\n{query}\nGive reasoning and answer."
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    output = llm.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit interface
st.title("Medical RAG Assistant")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.subheader("Try an Example Case:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1: NSTEMI"):
        st.session_state.input_text = """Patient endorses right sided chest pain for the last 2 days which worsened today, at which point she started having nausea and vomiting. Chest pain both at rest and on exertion. 
In the ED initial vitals were: 96.7 70 163/78 18 97% RA 
EKG: ST depressions in V2-V4 
Labs/studies notable for: Trop-T: 0.55, lactate 2.9, K 6.0"""

with col2:
    if st.button("Example 2: Pulmonary Embolism"):
        st.session_state.input_text = """Sudden onset of dyspnea and sharp chest pain worsened by deep breaths. Patient has a history of DVT and recent surgery.
Vitals: Tachycardia (110 bpm), Tachypnea (24/min), O2 Sat 92% on RA.
Signs: Swelling and redness in right calf."""

with col3:
    if st.button("Example 3: Diabetes Type 2"):
        st.session_state.input_text = """45-year-old male presents with increased thirst, frequent urination, and unexplained weight loss.
Labs: Fasting Plasma Glucose 140 mg/dL. HbA1c 7.5%.
Patient is obese and has a sedentary lifestyle."""

user_input = st.text_area("Enter patient case:", value=st.session_state.input_text, height=200)

if st.button("Diagnose"):
    if user_input.strip():
        answer = generate_answer(user_input)
        st.subheader("Answer")
        st.write(answer)
