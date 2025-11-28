import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load KB
with open("combined_rag_data.json","r") as f:
    kb_items=json.load(f)

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
user_input = st.text_area("Enter patient case:")

if st.button("Diagnose"):
    if user_input.strip():
        answer = generate_answer(user_input)
        st.subheader("Answer")
        st.write(answer)
