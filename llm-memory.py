import os
from datetime import datetime

import numpy as np
import ollama
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

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
        dimension=384,  # Matches 'sentence-transformers/all-MiniLM-L6-v2'
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=pinecone_index, embedding=embedding_model, text_key="text")

# Exponential decay function for memory degradation
def calculate_decay_factor(timestamp, half_life_days=30):
    now = datetime.now()
    time_diff = (now - datetime.fromisoformat(timestamp)).total_seconds() / (24 * 3600)  # Days
    decay = np.exp(-time_diff / half_life_days)  # Exponential decay
    return decay

# Function to determine importance (simplified example)
def calculate_importance(user_input, response):
    # Example: Importance based on keywords or length (could use sentiment analysis or user tagging)
    keywords = ["important", "urgent", "love", "hate", "remember"]
    base_importance = 1.0
    for keyword in keywords:
        if keyword in user_input.lower() or keyword in response.lower():
            base_importance += 1.0
    return min(base_importance, 5.0)  # Cap at 5 for simplicity

# Store conversation with decay and importance
def store_conversation(user_input, response):
    conversation_text = f"User: {user_input}\nAI: {response}"
    importance = calculate_importance(user_input, response)
    timestamp = datetime.now().isoformat()
    
    metadata = {
        "text": conversation_text,
        "timestamp": timestamp,
        "importance": importance
    }
    
    # Embed the text
    embedding = embedding_model.embed_documents([conversation_text])[0]
    # Amplify embedding based on importance (optional, for stronger imprint)
    amplified_embedding = [v * importance for v in embedding]
    
    # Store in Pinecone with unique ID
    vector_store.add_texts(
        texts=[conversation_text],
        embeddings=[amplified_embedding],
        metadatas=[metadata]
    )

# Retrieve context with decay and importance weighting
def retrieve_context(query, max_results=3, decay_threshold=0.1):
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=max_results * 2)  # Get extra results to filter
    
    # Filter and weight results
    weighted_results = []
    for doc, score in results:
        metadata = doc.metadata
        timestamp = metadata["timestamp"]
        importance = metadata.get("importance", 1.0)
        
        # Calculate decay factor
        decay_factor = calculate_decay_factor(timestamp)
        
        # Weight the similarity score by decay and importance
        weighted_score = score * decay_factor * importance
        
        if weighted_score > decay_threshold:  # Filter out highly decayed memories
            weighted_results.append((doc, weighted_score))
    
    # Sort by weighted score and take top results
    weighted_results.sort(key=lambda x: x[1], reverse=True)
    top_results = weighted_results[:max_results]
    
    context = "\n".join([doc.page_content for doc, _ in top_results])
    return context

# Get response with memory
def get_response_with_memory(user_input):
    context = retrieve_context(user_input)
    prompt = f"Previous conversation (faded by time, stronger if important):\n{context}\n\nUser: {user_input}\nAI:"
    
    response = ollama.chat(
        model="llama3.1:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_response = response["message"]["content"]
    store_conversation(user_input, ai_response)
    return ai_response

# Chat loop
def chat():
    print("Start chatting (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        response = get_response_with_memory(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    chat()