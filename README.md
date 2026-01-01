# 🩺 MEDICAL RAG CHATBOT  
## 📄 PDF-Based Medical Question Answering System using FAISS, LangChain & Ollama  

---

## 🚀 PROJECT OVERVIEW  

The **Medical RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** based application that answers medical questions **strictly from provided medical documents (PDFs)**.

This system does **NOT hallucinate answers** and does **NOT rely on the internet** for inference.  
All responses are generated using a **local Large Language Model (LLM)** powered by **Ollama** and a **FAISS vector database**.

⚠️ **Disclaimer**  
This project is intended **only for educational and academic purposes** and **must not be used as a replacement for professional medical advice**.

---

## 🧠 SYSTEM ARCHITECTURE & WORKFLOW  

Medical PDF Documents
↓
Text Extraction
↓
Text Chunking
↓
Vector Embeddings
↓
FAISS Vector Store
↓
Retriever (LangChain)
↓
Ollama Local LLM
↓
Final Answer (CLI / Streamlit UI)


---

## 🧱 TECHNOLOGY STACK  

- 🧠 **LLM**: Ollama (Mistral / LLaMA3 / Phi-3)  
- 🔗 **Framework**: LangChain  
- 📦 **Vector Database**: FAISS  
- 🧾 **Embeddings**: Sentence-Transformers  
- 🌐 **Frontend**: Streamlit  
- 🐍 **Programming Language**: Python  

---

## 📁 PROJECT DIRECTORY STRUCTURE  



medical-chatbot/
│
├── app.py # Streamlit Web Application
├── connect_memory.py # CLI-based Chatbot
├── create_memory.py # PDF → FAISS Vector Store Generator
├── requirements.txt # Python Dependencies
├── README.md # Project Documentation
│
├── data/
│ └── *.pdf # Medical PDF Documents
│
└── vectorstore/
└── db_faiss/
├── index.faiss
└── index.pkl


---

## 🧩 STEP 1: CREATE THE VECTOR STORE (MANDATORY STEP)  

This is the **first and most important step**.

### 🔍 What happens in this step?  
- Medical PDF documents are loaded  
- Text is split into meaningful chunks  
- Each chunk is converted into vector embeddings  
- Embeddings are stored in a FAISS vector database  

### ▶️ Command to run  

```bash
python create_memory.py

📂 Input

Place all medical PDFs inside the data/ directory

📦 Output

FAISS vector store generated at:

vectorstore/db_faiss/

🧩 STEP 2: CONNECT VECTOR STORE TO LLM (CLI CHATBOT)

This step connects:

FAISS vector store (memory)

Ollama local LLM

LangChain retrieval pipeline

▶️ Run the CLI chatbot
python connect_memory.py

🧪 Example Interaction
Ask medical question: What is cancer?
Answer: Cancer is a disease in which abnormal cells divide uncontrollably...

🧩 STEP 3: RUN THE STREAMLIT WEB APPLICATION 🌐

This step provides a user-friendly web interface for the chatbot.

▶️ Start Ollama (CPU mode recommended)
set OLLAMA_NO_CUDA=1
set OLLAMA_NUM_GPU=0
ollama serve

▶️ Run the Streamlit app
streamlit run app.py

🌍 Open in browser
http://localhost:8501

🖥️ STREAMLIT APPLICATION FEATURES

✅ Real-time medical chatbot
✅ Session-based chat history
✅ Answers strictly from PDFs
✅ Medical safety disclaimer
✅ Fast FAISS retrieval
✅ Clean and professional UI

🛡️ SAFETY & RELIABILITY

❌ No internet-based inference

❌ No hallucinated medical answers

✅ Fully local execution

✅ Privacy-preserving architecture

📦 INSTALLATION GUIDE
1️⃣ Create a virtual environment (recommended)
conda create -n medi_chat python=3.10
conda activate medi_chat

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install Ollama

👉 https://ollama.com

4️⃣ Pull a local model
ollama pull mistral

🧪 TESTED ENVIRONMENT

Windows 10 / 11

CPU-only systems

Conda environments

Python 3.9 – 3.11
