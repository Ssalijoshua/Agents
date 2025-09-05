# 🧠 Modeling and Simulation RAG System

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to provide **accurate and context-aware information** about **modeling and simulation** using **Large Language Models (LLMs)** and **vector-based retrieval**.

It leverages **LangChain**, **FAISS**, and **Groq’s API** to retrieve relevant document chunks from a **user-provided document set** and generate answers to queries such as:  
- “What is simulation?”  
- “What is the role of modeling in Simulations?”

---

## ✨ Core Features
The system is designed to deliver reliable information on **modeling and simulation** through:

- **Document Retrieval** – Efficiently retrieves relevant text from documents using a **FAISS vector store**.
- **Answer Generation** – Combines retrieved context with **Groq’s LLM** to produce coherent, context-specific responses.
- **Fallback Mechanism** – Returns raw document chunks if the **LLM** is unreachable due to network issues.
- **Interactive Interface** – Provides a **command-line interface (CLI)** for users to ask questions and receive answers.

The system is structured to process **user-provided text files** containing modeling and simulation content, ensuring **answers are grounded** in the provided documents.

---

## 🛠️ Built With
This project is powered by:

- **Python (>=3.12)** – Core programming language.
- **LangChain (>=0.2.2)** – Framework for RAG and LLM integration.
- **FAISS** – Vector store for efficient document retrieval.
- **Groq API** – Provides LLM capabilities *(e.g., llama3-8b-8192)*.
- **HuggingFace Embeddings** – For generating document embeddings.
- **python-dotenv** – For environment variable management.

📄 For full dependencies, see **requirements.txt**.

---

## 👥 Target Audience
- **Researchers and Students** – Studying modeling, simulation, or related fields who need quick access to domain-specific information.
- **AI/ML Practitioners** – Developers experimenting with RAG systems for knowledge retrieval in specialized domains.
- **Educators** – Professionals seeking to explore or teach concepts related to modeling and simulation.

---

## 📂 Project Structure
This repository follows a streamlined directory structure designed for clarity and ease of use:

root/
├── docs/ # Text documents about modeling and simulation
├── faiss_index/ # FAISS vector store (generated)
├── .env # Environment variables (create with GROQ_API_KEY)
├── .gitignore # Git ignore file
├── generate_vector_embeddings.py # Script to create FAISS vector store
├── language_model.py # Script for RAG pipeline and CLI
├── README.md # Project documentation
└── requirements.txt # Project dependencies