🩺 Medical Chatbot (LLM + Retrieval-Augmented Generation)

A Medical Chatbot built using LangChain, Hugging Face LLMs, and FAISS vector database.
The chatbot answers medical-related questions strictly from provided documents, ensuring grounded and reliable responses using Retrieval-Augmented Generation (RAG).

🚀 Features

💬 Ask medical questions in natural language

🧠 Uses Mistral-7B-Instruct LLM via Hugging Face

📚 Retrieves answers from a FAISS vector database

❌ Avoids hallucinations by answering only from context

🔍 Returns source documents along with answers

🧩 Custom prompt to control LLM behavior

🛠️ Tech Stack

Python

LangChain

Hugging Face Inference API

Mistral-7B-Instruct

FAISS Vector Store

Sentence Transformers

Environment Variables for Security

📁 Project Structure
medical-chatbot/
│
├── app.py                  # Main chatbot application
├── vectorstore/
│   └── db_faiss/            # FAISS vector database
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .env                     # Environment variables (not committed)

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd medical-chatbot

2️⃣ Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set Hugging Face Token

Create a .env file in the root directory:

HF_TOKEN=your_huggingface_api_token_here


Or set it directly in your terminal:

export HF_TOKEN=your_token_here     # Linux / Mac
set HF_TOKEN=your_token_here        # Windows

▶️ Running the Chatbot
python app.py


You will be prompted with:

Write Query Here:


Enter your medical question and get:

✅ Answer

📄 Source documents used

🧠 How It Works (High Level)

User enters a medical query

Query is converted into embeddings

FAISS retrieves relevant documents

Context + query is sent to Mistral LLM

LLM generates a grounded response

The chatbot does not answer outside the provided context, ensuring safer outputs. 

53085c05-9e01-4552-aa20-07e5405…

⚠️ Disclaimer

This chatbot is not a replacement for professional medical advice.
Always consult a qualified healthcare professional for medical concerns.

📌 Future Improvements

Add Streamlit / Web UI

Support PDF uploads

Multi-document ingestion

Chat history memory

Better prompt tuning

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.

⭐ Acknowledgements

Hugging Face

LangChain

Mistral AI

FAISS



