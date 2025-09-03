from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import groq
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Step 1: Ingest Documents into Vector Store (for completeness, optional if already done)
def ingest_documents():
    logger.info("Ingesting documents from docs/ directory")
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        raise ValueError("No documents found in 'docs/' directory.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local("faiss_index")
    logger.info("Vector store created and saved to faiss_index")

# Step 2: Set Up Groq LLM and Prompt
def setup_rag_chain():
    logger.info("Setting up RAG chain")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    prompt_template = PromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {input}\nAnswer:"
    )
    stuff_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    return rag_chain, retriever

# Step 3: CLI for Interaction
def ask_question(rag_chain, retriever, question):
    logger.info(f"Processing question: {question}")
    try:
        response = rag_chain.invoke({"input": question})
        print("Answer:", response["answer"])
        print("\nRetrieved Contexts:")
        for doc in response["context"]:
            print(doc.page_content[:100] + "...")
    except groq.APIConnectionError as e:
        logger.error(f"Failed to connect to Groq API: {e}")
        print(f"Error: Failed to connect to Groq API - {e}")
        print("Falling back to raw retrieved documents:")
        try:
            docs = retriever.invoke(question)
            if not docs:
                print("No relevant documents found.")
            else:
                print("\nRetrieved Documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"Document {i}: {doc.page_content[:200]}...")
        except Exception as retriever_error:
            logger.error(f"Retriever error: {retriever_error}")
            print(f"Error in retriever: {retriever_error}")

# Main execution
if __name__ == "__main__":
    # Ingest documents if not already done
    if not os.path.exists("faiss_index"):
        ingest_documents()

    # Set up RAG chain and retriever
    try:
        rag_chain, retriever = setup_rag_chain()
    except Exception as e:
        logger.error(f"Failed to set up RAG chain: {e}")
        print(f"Error setting up RAG chain: {e}")
        exit(1)

    # CLI loop
    while True:
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        ask_question(rag_chain, retriever, question)