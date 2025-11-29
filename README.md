# NLP_Project04_RAG

# Medical RAG Assistant

This repository contains a **Retrieval-Augmented Generation (RAG) system** for medical case reasoning. The system retrieves relevant medical knowledge from a curated knowledge base (KB) and generates informative explanations for clinical scenarios using a language model.

---

## **Overview**

RAG combines **retrieval** and **generation**:

1. **Retrieval**:  
   - A medical knowledge base (KB) is embedded using **sentence-transformers**.  
   - Queries are encoded and compared with KB embeddings using **cosine similarity**.  
   - The top-K relevant documents are retrieved for reasoning.

2. **Generation**:  
   - Retrieved knowledge is combined with the user query into a **prompt**.  
   - A **language model** (local LLM or **Google Gemini API**) generates the final explanation.  

This allows the system to provide **context-aware, informative answers** for patient cases.

---

## **Features**

- ✅ Retrieve relevant medical knowledge from a structured JSON KB.
- ✅ Generate detailed reasoning using a LLM or Gemini API.
- ✅ Simple Streamlit interface for interactive querying.
- ✅ Example cases included for testing.

---

## **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/medical-rag-assistant.git
cd medical-rag-assistant

# Install dependencies
pip install -r requirements.txt
