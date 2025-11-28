import json
import os
import numpy as np
from typing import List, Dict

# Try importing sentence_transformers, handle if not installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: 'sentence-transformers' library is not installed.")
    print("Please install it using: pip install sentence-transformers")
    exit(1)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2' # Lightweight, fast model
TOP_K = 3 # Number of KB items to retrieve

def load_data(file_path: str):
    """Loads the processed JSON dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_kb_embeddings(model, kb_items: List[Dict]):
    """
    Encodes all Knowledge Base items into vectors.
    Returns a list of (id, content, embedding).
    """
    print("Encoding Knowledge Base items... (this may take a moment)")
    
    # We will encode the 'content' of the KB item. 
    # You could also include 'topic' like f"{item['topic']}: {item['content']}"
    texts = [f"{item['topic']}: {item['content']}" for item in kb_items]
    
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def retrieve_relevant_kb(model, query_text: str, kb_embeddings, kb_items: List[Dict], k=TOP_K):
    """
    Retrieves the top-k most relevant KB items for a given query (patient case).
    """
    from sentence_transformers import util
    
    # Encode the query (patient case input)
    # Note: If the patient case is very long, the model might truncate it.
    # For retrieval, usually a summary or the first chunk is enough, 
    # but MiniLM handles up to 256/512 tokens. 
    # For better retrieval on long cases, you might want to extract key symptoms first.
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    
    # Compute Cosine Similarity
    cos_scores = util.cos_sim(query_embedding, kb_embeddings)[0]
    
    # Get top-k indices
    top_results = np.argpartition(-cos_scores.cpu(), range(k))[:k]
    
    results = []
    for idx in top_results:
        score = cos_scores[idx].item()
        results.append((kb_items[idx], score))
        
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def construct_prompt(patient_case: Dict, retrieved_kb: List[tuple]):
    """
    Constructs the final prompt for the LLM.
    Includes the FULL patient case and the retrieved knowledge.
    """
    
    # Format Retrieved Context
    context_str = "Medical Knowledge:\n"
    for item, score in retrieved_kb:
        context_str += f"- [{item['topic']}] {item['content']} (Relevance: {score:.2f})\n"
    
    # Format Patient Case
    case_str = f"Patient Case Details:\n{patient_case['input_text']}"
    
    # Assemble Prompt
    prompt = f"""You are an expert medical diagnostician. Use the provided Medical Knowledge to analyze the Patient Case.

{context_str}
---
{case_str}
---
Task:
1. Analyze the patient's symptoms and history.
2. Use the Medical Knowledge to support your reasoning.
3. Provide a differential diagnosis and the most likely diagnosis.
4. Explain your reasoning step-by-step.

Diagnosis:"""
    
    return prompt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "processed_rag_dataset.json")
    
    print("1. Loading Data...")
    data = load_data(data_path)
    knowledge_base = data['knowledge_base']
    cases = data['cases']
    
    print(f"Loaded {len(knowledge_base)} KB items and {len(cases)} cases.")
    
    print("2. Initializing Model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("3. Indexing Knowledge Base...")
    kb_embeddings = create_kb_embeddings(model, knowledge_base)
    
    # --- Demonstration with a Sample Case ---
    # Let's pick the first case
    sample_case = cases[0]
    print(f"\n--- Processing Sample Case: {sample_case['id']} ---")
    print(f"Disease Group: {sample_case['disease_group']}")
    
    print("4. Retrieving Relevant Knowledge...")
    # We use the full input text as the query. 
    # In a real system, you might want to summarize it first or extract keywords.
    retrieved_items = retrieve_relevant_kb(model, sample_case['input_text'], kb_embeddings, knowledge_base)
    
    print("5. Constructing Prompt...")
    prompt = construct_prompt(sample_case, retrieved_items)
    
    print("\n" + "="*50)
    print("GENERATED PROMPT")
    print("="*50)
    print(prompt)
    print("="*50)
    
    # Check token length (approximation)
    approx_tokens = len(prompt.split()) * 1.3
    print(f"\nApproximate Prompt Token Count: {int(approx_tokens)}")
    print("Note: Most LLMs (GPT-4, Claude, etc.) have context windows of 8k+ tokens, so this fits easily.")

if __name__ == "__main__":
    main()
