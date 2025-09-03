from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Ingest Documents into Vector Store (for completeness, optional if already done)
def ingest_documents():
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        raise ValueError("No documents found in 'docs/' directory.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local("faiss_index")

# Step 2: Set Up Groq LLM and Prompt
def setup_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    prompt_template = PromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {input}\nAnswer:"
    )  # Changed {question} to {input}
    stuff_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    return rag_chain

# Step 3: CLI for Interaction
def ask_question(rag_chain, question):
    response = rag_chain.invoke({"input": question})  # Keep "input"
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