from dotenv import load_dotenv, find_dotenv

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------------------------------------------
# ENV
# -------------------------------------------------------------------
load_dotenv(find_dotenv())


# -------------------------------------------------------------------
# CUSTOM PROMPT
# -------------------------------------------------------------------
CUSTOM_PROMPT_TEMPLATE = """
Use only the information provided in the context to answer the user's question.
If the answer is not present in the context, say "I do not know".
Do not add any external information.

Context:
{context}

Question:
{question}

Answer directly:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# -------------------------------------------------------------------
# LOAD OLLAMA LLM
# -------------------------------------------------------------------
def load_llm():
    llm = Ollama(
        model="mistral",   # or "llama3"
        temperature=0.5
    )
    return llm


# -------------------------------------------------------------------
# LOAD FAISS VECTORSTORE
# -------------------------------------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# -------------------------------------------------------------------
# CREATE QA CHAIN
# -------------------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt()
    }
)


# -------------------------------------------------------------------
# CHAT LOOP
# -------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk medical question (or 'exit'): ").strip()

        if user_query.lower() == "exit":
            print("Exiting...")
            break

        response = qa_chain.invoke({"query": user_query})

        print("\nAnswer:")
        print(response.get("result", "No answer generated"))
