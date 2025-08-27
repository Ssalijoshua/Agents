from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Accessing the variables



def setup_rag_chain():
    # Load saved vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )

    # Prompt template
    prompt_template = PromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Chain: Stuff docs into prompt + LLM
    stuff_chain = create_stuff_documents_chain(llm, prompt_template)

    # Full retrieval chain
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    return rag_chain

# Step 3: CLI for Interaction
def ask_question(rag_chain, question):
    response = rag_chain.invoke({"input": question})
    print("Answer:", response["answer"])
    print("\nRetrieved Contexts:")
    for doc in response["context"]:
        print(doc.page_content[:100] + "...")

# Main execution
if __name__ == "__main__":
    # Ingest documents if not already done
    if not os.path.exists("faiss_index"):
        ingest_documents()

    # Set up RAG chain
    rag_chain = setup_rag_chain()

    # CLI loop
    while True:
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        ask_question(rag_chain, question)