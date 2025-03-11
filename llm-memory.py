import os
from datetime import datetime

import nltk
import numpy as np
import ollama
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Download VADER lexicon if not already available
nltk.download('vader_lexicon', quiet=True)

# Initialize Pinecone with API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to a Pinecone index
index_name = "human-memory"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embedding_model, text_key="text")

current_user_identity = "unknown"

# Extract user identity explicitly from LLM
def get_user_identity(user_input):
    prompt = f"Given the following input from a user, what's the user's name? " \
             f"If the name isn't explicitly provided, answer 'unknown'.\n\n" \
             f"User Input: {user_input}\nUser Name:"

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    user_identity = response["message"]["content"].strip()

    return user_identity if user_identity else "unknown"

# Exponential decay function for memory degradation
def calculate_decay_factor(timestamp, half_life_days=90):
    now = datetime.now()
    time_diff = (now - datetime.fromisoformat(timestamp)).total_seconds() / (24 * 3600)
    return np.exp(-time_diff / half_life_days)

# Function to determine importance using VADER sentiment analysis
def calculate_importance(user_input, response):
    sid = SentimentIntensityAnalyzer()
    user_sentiment = sid.polarity_scores(user_input)
    response_sentiment = sid.polarity_scores(response)
    importance = (user_sentiment['compound'] + response_sentiment['compound']) / 2
    return importance

# Store conversation with identity metadata
def store_conversation(user_input, response, user_identity="unknown"):
    conversation_text = f"User: {user_input}\nAI: {response}"
    importance = calculate_importance(user_input, response)
    timestamp = datetime.now().isoformat()

    metadata = {
        "text": conversation_text,
        "timestamp": timestamp,
        "importance": importance,
        "user_identity": user_identity
    }

    embedding = embedding_model.embed_documents([conversation_text])[0]
    amplified_embedding = [v * importance for v in embedding]

    vector_store.add_texts(
        texts=[conversation_text],
        embeddings=[amplified_embedding],
        metadatas=[metadata]
    )

# Retrieve context from vector store
def retrieve_context(query, max_results=3, decay_threshold=0.05):
    results = vector_store.similarity_search_with_score(query=query, k=max_results * 2)

    weighted_results = []
    for doc, score in results:
        metadata = doc.metadata
        decay_factor = calculate_decay_factor(metadata["timestamp"])
        weighted_score = score * decay_factor * metadata.get("importance", 1.0)

        if weighted_score > decay_threshold:
            weighted_results.append((doc, weighted_score))

    weighted_results.sort(key=lambda x: x[1], reverse=True)
    top_results = weighted_results[:max_results]

    context = "\n".join([doc.page_content for doc, _ in top_results])
    return context

# Get response with memory and identity tracking
current_user_identity = "unknown"

def get_response_with_memory(user_input):
    global current_user_identity

    if current_user_identity == "unknown":
        detected_identity = get_user_identity(user_input)
        if detected_identity != "unknown":
            current_user_identity = detected_identity

    context = retrieve_context(user_input)
    prompt = f"Previous conversation (context):\n{context}\n\nUser: {user_input}\nAI:"

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_response = response["message"]["content"]
    store_conversation(user_input, ai_response, current_user_identity)
    return ai_response

# Main chat loop
def chat():
    print("Start chatting (type 'exit' to stop):")
    while True:
        user_input = input("\n\n--------\nYou: ")
        if user_input.lower() == "exit":
            break

        response = get_response_with_memory(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    chat()
