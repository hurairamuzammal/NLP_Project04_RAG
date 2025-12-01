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
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
DATA_PATH = os.path.join(BASE_DIR,"mimic-iv-ext-direct-1.0.0","My_Dataset","combined_rag_data.json")
KB_FAISS_PATH = os.path.join(VECTOR_STORE_DIR, "medical_kb_index.faiss")
CASES_FAISS_PATH = os.path.join(VECTOR_STORE_DIR, "patient_cases_index.faiss")
KB_PICKLE = os.path.join(VECTOR_STORE_DIR, "kb_items.pkl")
PATIENT_PICKLE = os.path.join(VECTOR_STORE_DIR, "patient_items.pkl")

# ---------------------------
# Load SentenceTransformer
# ---------------------------
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# ---------------------------
# Load or Create KB Items and Patient Items
# ---------------------------
if os.path.exists(KB_PICKLE) and os.path.exists(PATIENT_PICKLE):
    try:
        with open(KB_PICKLE, "rb") as f:
            kb_items = pickle.load(f)
        with open(PATIENT_PICKLE, "rb") as f:
            patient_items = pickle.load(f)
        st.success(f"Loaded {len(kb_items)} KB items and {len(patient_items)} patient cases from pickle files")
    except Exception as e:
        st.error(f"Error loading pickle files: {e}")
        # Fall back to loading from JSON
        kb_items = []
        patient_items = []
else:
    kb_items = []
    patient_items = []

if len(kb_items) == 0 or len(patient_items) == 0:
    st.warning("Pickle files not found or empty, loading from JSON and creating them...")
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
        
        # Filter and validate items
        kb_items = [item for item in data if "medicalKB" in item and "id" in item]
        patient_items = [item for item in data if "patient_case" in item and "id" in item]
        
        st.info(f"Loaded {len(kb_items)} KB items and {len(patient_items)} patient cases from JSON")
        
        # Create vector_store directory if it doesn't exist
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        # Save pickle files
        with open(KB_PICKLE, "wb") as f:
            pickle.dump(kb_items, f)
        with open(PATIENT_PICKLE, "wb") as f:
            pickle.dump(patient_items, f)
        
        st.success("Pickle files created successfully")
    except FileNotFoundError:
        st.error(f"JSON file not found at: {DATA_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        st.stop()

# ---------------------------
# Load or Build FAISS Index for Medical KB
# ---------------------------
# ---------------------------
# Load or Build FAISS Index for Medical KB
# ---------------------------
rebuild_kb = True
if os.path.exists(KB_FAISS_PATH):
    try:
        kb_faiss_index = faiss.read_index(KB_FAISS_PATH)
        if kb_faiss_index.d == model.get_sentence_embedding_dimension():
            rebuild_kb = False
        else:
            st.warning(f"KB FAISS index dimension mismatch ({kb_faiss_index.d} vs {model.get_sentence_embedding_dimension()}), rebuilding...")
    except Exception as e:
        st.error(f"Error loading KB FAISS index: {e}, rebuilding...")

if rebuild_kb:
    if not os.path.exists(KB_FAISS_PATH):
        st.warning("KB FAISS index not found, rebuilding...")
    
    kb_texts = [item["medicalKB"] for item in kb_items]
    kb_emb = model.encode(kb_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(kb_emb)
    dim = kb_emb.shape[1]
    kb_faiss_index = faiss.IndexFlatIP(dim)
    kb_faiss_index.add(kb_emb)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(kb_faiss_index, KB_FAISS_PATH)

# ---------------------------
# Load or Build FAISS Index for Patient Cases
# ---------------------------
rebuild_cases = True
if os.path.exists(CASES_FAISS_PATH):
    try:
        cases_faiss_index = faiss.read_index(CASES_FAISS_PATH)
        if cases_faiss_index.d == model.get_sentence_embedding_dimension():
            rebuild_cases = False
        else:
            st.warning(f"Cases FAISS index dimension mismatch ({cases_faiss_index.d} vs {model.get_sentence_embedding_dimension()}), rebuilding...")
    except Exception as e:
        st.error(f"Error loading Cases FAISS index: {e}, rebuilding...")

if rebuild_cases:
    if not os.path.exists(CASES_FAISS_PATH):
        st.warning("Cases FAISS index not found, rebuilding...")
        
    case_texts = [item["patient_case"] for item in patient_items]
    case_emb = model.encode(case_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(case_emb)
    dim = case_emb.shape[1]
    cases_faiss_index = faiss.IndexFlatIP(dim)
    cases_faiss_index.add(case_emb)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
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
    for idx, (source_type, item, score) in enumerate(retrieved):
        try:
            if source_type == "KB":
                item_id = item.get('id', f'KB_{idx}')
                kb_text = item.get('medicalKB', 'No content')
                context_parts.append(f"- [KB: {item_id}] {kb_text} (Relevance: {score:.2f})")
            else:
                # Patient case has nested structure - extract the relevant information
                item_id = item.get('id', f'CASE_{idx}')
                case_data = item.get('patient_case', {})
                # Combine all inputs into a readable format
                inputs = case_data.get('inputs', {})
                inputs_text = " | ".join([f"{k}: {v}" for k, v in inputs.items() if v and v != "None"])
                case_summary = f"Disease: {case_data.get('specific_disease', 'Unknown')} | {inputs_text[:500]}"
                context_parts.append(f"- [CASE: {item_id}] {case_summary} (Relevance: {score:.2f})")
        except Exception as e:
            st.warning(f"Error processing item {idx}: {e}")
            continue
    
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
st.markdown("---")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Example patient cases
st.subheader("Try an Example Case")
st.markdown("")  # Add spacing

# Example buttons in a 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1: NSTEMI with CAD", use_container_width=True):
        st.session_state.input_text = """A female patient presented with dull aching pain in the back and chest that began yesterday while at rest. This pain is different from her usual sharp chest pain episodes and lasted significantly longer. The pain was not relieved by multiple sublingual nitroglycerin tablets taken at home. She reports ongoing shortness of breath for several weeks. She has a significant cardiac history including coronary artery disease, and multiple comorbidities: uncontrolled type 2 diabetes mellitus, hypertension, hypothyroidism, asthma, obstructive sleep apnea, and morbid obesity with BMI between 40-44.9. Her father had a myocardial infarction. On examination, vital signs showed blood pressure 186/71 mmHg, heart rate 88 beats per minute, temperature 98 degrees F, respiratory rate 18 per minute, and oxygen saturation 98% on room air. Physical examination revealed distant heart sounds, regular rate and rhythm without murmurs, clear lungs bilaterally, soft obese abdomen, and 1+ bilateral pitting edema. Laboratory studies showed elevated troponin at 0.60, white blood cell count 9.3, hemoglobin 13.7, markedly elevated glucose at 339, and bicarbonate 24. Electrocardiogram demonstrated sinus rhythm at rate 62, QTc interval 456 milliseconds, with no new ischemic changes."""

with col2:
    if st.button("Example 2: NSTEMI 3-Vessel", use_container_width=True):
        st.session_state.input_text = """A 66-year-old male initially presented with urinary retention requiring Foley catheter placement, after which he developed profound weakness, hypotension, and diaphoresis. Over the past month, he has experienced intermittent weakness episodes and syncopal events. His exercise tolerance has significantly decreased, becoming short of breath with even brief periods of activity. Notably, he denied chest pain during this acute episode. He has chronic back pain and has not received medical care for the past decade. Vital signs at presentation were blood pressure 157/85 mmHg, heart rate 98 beats per minute, temperature 98.4 degrees F, respiratory rate 16 per minute, and oxygen saturation 98%. Physical examination showed a comfortable-appearing patient with clear chest auscultation, regular cardiac rhythm, soft abdomen, no peripheral edema, and warm dry skin. Laboratory results revealed troponin elevation at 0.34, white blood cell count 10.8, hemoglobin 15.4, glucose 127, and hemoglobin A1c 5.5%. Electrocardiogram showed T wave inversions in the inferolateral leads with less than 1mm ST elevations in leads I and aVL, and Q waves anteriorly and inferiorly. Cardiac catheterization demonstrated severe three-vessel coronary artery disease with complete occlusion of the right coronary artery and diffuse disease in the left anterior descending and circumflex arteries."""

with col3:
    if st.button("Example 3: NSTEMI with DM/AFib", use_container_width=True):
        st.session_state.input_text = """A male patient with known diabetes mellitus, atrial fibrillation, and hypertension presented with one day of epigastric pressure occurring at rest without radiation. The pain began while sitting and persisted until he fell asleep. Similar epigastric discomfort occurred the morning of admission after taking his daily medications. He experienced complete pain relief after taking aspirin 325mg at home. He denied exertional chest pain, shortness of breath, nausea, vomiting, or diaphoresis. Additionally, he reported having diarrhea for approximately one month. He is on both oral agents and insulin for diabetes management. His family history is significant for myocardial infarction in his father and diabetes in his mother. Vital signs on presentation were blood pressure 140/82 mmHg, heart rate 74 beats per minute, temperature 98.4 degrees F, respiratory rate 21 per minute, and oxygen saturation 95% on room air. Physical examination revealed a patient in no acute distress with clear oropharynx, regular cardiac rhythm, clear lung fields bilaterally, soft non-tender abdomen, and no peripheral edema. Initial laboratory studies showed troponin 0.27 which subsequently rose to 1.77, white blood cell count 9.5, hemoglobin 15.0, elevated glucose at 286, and international normalized ratio 1.0. Electrocardiogram demonstrated ST segment elevation in lead III, T wave inversions in leads V4 through V6, and ST segment depressions in leads I and V6, findings consistent with acute myocardial ischemia."""

st.markdown("")  # Add spacing
st.markdown("### Enter Patient Case")
user_input = st.text_area(
    "Patient Case Input", 
    value=st.session_state.input_text,
    height=200,
    placeholder="Describe the patient symptoms, vitals, and relevant medical history...",
    label_visibility="collapsed"
)

# Privacy statement
st.caption("⚠️ Privacy Notice: Please avoid entering patient names or any sensitive personal information.")

if st.button("Diagnose"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            answer, retrieved_items, avg_score = generate_answer(user_input, api_key)
        
        st.markdown("### Retrieved Knowledge & Similar Cases")
        for idx, (source_type, item, score) in enumerate(retrieved_items, 1):
            try:
                if source_type == "KB":
                    item_id = item.get('id', f'KB_{idx}')
                    kb_text = item.get('medicalKB', 'No content available')
                    st.markdown(f"**{idx}. [Medical KB] {item_id}** - Score: {score:.2f}")
                    st.write(kb_text)
                else:
                    item_id = item.get('id', f'CASE_{idx}')
                    case_data = item.get('patient_case', {})
                    st.markdown(f"**{idx}. [Patient Case] {item_id}** - Score: {score:.2f}")
                    st.markdown(f"**Disease:** {case_data.get('specific_disease', 'Unknown')} ({case_data.get('disease_group', 'Unknown')})")
                    
                    # Show inputs
                    inputs = case_data.get('inputs', {})
                    if inputs:
                        with st.expander("View Case Details"):
                            for key, value in inputs.items():
                                if value and value != "None":
                                    st.markdown(f"**{key}:**")
                                    st.write(value)
                                    st.markdown("---")
            except Exception as e:
                st.error(f"Error displaying item {idx}: {e}")
        
        st.markdown("### AI Response")
        st.write(answer)

        st.markdown(f"**Average Relevance Score:** {avg_score:.2f}")
    else:
        st.warning("Enter patient case details.")
