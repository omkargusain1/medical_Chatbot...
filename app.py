import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------
load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🩺 Medical RAG Chatbot")
st.sidebar.markdown("""
**Technology Stack**
- Ollama (Local LLM)
- FAISS Vector Store
- LangChain
- Streamlit

⚠️ **Disclaimer**  
This chatbot is for **educational purposes only**  
and should **not replace professional medical advice**.
""")


# ---------------------------------------------------------
# CUSTOM PROMPT
# ---------------------------------------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant.
Answer the question using ONLY the information provided in the context.
If the answer is not present in the context, say "I do not know".

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# ---------------------------------------------------------
# LOAD LLM (OLLAMA)
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    return Ollama(
        model="mistral",   # or "llama3" / "phi3"
        temperature=0.5
    )


# ---------------------------------------------------------
# LOAD FAISS
# ---------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


# ---------------------------------------------------------
# LOAD QA CHAIN
# ---------------------------------------------------------
@st.cache_resource
def load_qa_chain():
    db = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )
    return qa_chain


qa_chain = load_qa_chain()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# ---------------------------------------------------------
# CALLBACK FUNCTION
# ---------------------------------------------------------
def submit_question():
    question = st.session_state.user_input.strip()

    if question:
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"query": question})
            answer = response.get("result", "No answer found.")

            st.session_state.chat_history.append((question, answer))

    # clear input box after submit
    st.session_state.user_input = ""


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("🩺 Medical Question Answering System")
st.markdown("Ask medical questions based on uploaded medical documents.")

st.text_input(
    "Enter your medical question:",
    key="user_input",
    on_change=submit_question,
    placeholder="e.g. What is cancer?"
)


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------
st.divider()
st.subheader("Chat History")

for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**Q{i+1}:** {q}")
    st.markdown(f"**A{i+1}:** {a}")
    st.markdown("---")
