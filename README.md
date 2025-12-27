# 🤖 Context-Aware RAG Chatbot

A production-ready, context-aware Retrieval-Augmented Generation (RAG) chatbot built using **LangChain v1.0+**, **OpenAI**, **FAISS**, and **Streamlit**.  
The chatbot retrieves answers from an external knowledge source and maintains conversational context across user interactions.

---

## 🎯 Objective of the Task

The objective of this project is to build a conversational AI system that:

- Retrieves accurate information from an external knowledge base
- Maintains conversational context for follow-up questions
- Reduces hallucinations by grounding responses in retrieved documents
- Uses modern LangChain v1.0+ APIs (LCEL)
- Provides a clean and secure frontend for OpenAI API key usage

---

## 🛠️ Methodology / Approach

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Data Ingestion**
   - Scrapes the *Artificial Intelligence* Wikipedia page using `WebBaseLoader`

2. **Text Processing**
   - Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter`

3. **Vectorization**
   - Converts text chunks into embeddings using OpenAI Embeddings
   - Stores vectors in a FAISS vector database for efficient similarity search

4. **Context-Aware Retrieval**
   - Reformulates user queries into standalone questions when chat history exists
   - Retrieves the most relevant document chunks based on semantic similarity

5. **Answer Generation**
   - Uses OpenAI Chat models to generate concise answers
   - Grounds responses strictly in retrieved context

6. **Frontend & UX**
   - Built with Streamlit
   - Secure API key input via frontend
   - Chat disabled until API key is connected
   - Clear chat history functionality

---

## 📊 Key Results / Observations

- The chatbot successfully answers factual and follow-up questions using retrieved context
- Conversational memory improves multi-turn question understanding
- Retrieval grounding significantly reduces hallucinated responses
- Modern LCEL-based LangChain implementation improves clarity and maintainability
- Clean UI flow enhances usability and security

---

## 🚀 Tech Stack

- **Python 3.10+**
- **Streamlit** (UI)
- **LangChain v1.0+**
- **OpenAI (Chat + Embeddings)**
- **FAISS** (Vector Store)

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/context-aware-rag-chatbot.git
cd context-aware-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
