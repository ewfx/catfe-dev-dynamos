import time
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load embedding model for evaluation
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load test dataset
with open("test_data.json", "r") as f:
    test_cases = json.load(f)

# Initialize Chatbot Components
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

# Function to prepare vector embedding database
def setup_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        java_files_directory = "./java_files"
        java_documents = []

        for root, _, files in os.walk(java_files_directory):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        java_documents.append(Document(page_content=content, metadata={"source": file_path}))

        if not java_documents:
            raise ValueError("No Java files found in the specified directory.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(java_documents)

        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        print("✅ Vector Store DB with Java Code is Ready")

# Call function to initialize vector store
setup_vector_embedding()

# Function to interact with the chatbot
def query_chatbot(query):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    return response.get("answer", "")

# Metrics Storage
retrieval_scores = []
generation_scores = {"BLEU": [], "ROUGE-L": []}
latencies = []

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Initialize Rouge for text similarity evaluation
rouge = Rouge()

# Run tests
for case in test_cases:
    query = case["query"]
    expected_response = case["expected_response"]

    # Measure Response Time
    start_time = time.time()
    chatbot_response = query_chatbot(query)
    end_time = time.time()

    # Compute Latency
    latencies.append(end_time - start_time)

    # Retrieval Accuracy (Cosine Similarity)
    expected_embedding = embedder.encode(expected_response)
    chatbot_embedding = embedder.encode(chatbot_response)
    retrieval_scores.append(cosine_similarity(expected_embedding, chatbot_embedding))

    # Generation Accuracy (BLEU & ROUGE)
    bleu_score = sentence_bleu([expected_response.split()], chatbot_response.split())
    rouge_score = rouge.get_scores(chatbot_response, expected_response)[0]["rouge-l"]["f"]

    generation_scores["BLEU"].append(bleu_score)
    generation_scores["ROUGE-L"].append(rouge_score)

# Print evaluation results
print(f"✅ Average Retrieval Accuracy (Cosine Similarity): {np.mean(retrieval_scores):.4f}")
print(f"✅ Average BLEU Score: {np.mean(generation_scores['BLEU']):.4f}")
print(f"✅ Average ROUGE-L Score: {np.mean(generation_scores['ROUGE-L']):.4f}")
print(f"✅ Average Latency (seconds): {np.mean(latencies):.4f}")
