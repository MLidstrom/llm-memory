import os
import uuid
from datetime import datetime

import nltk
import numpy as np
import ollama
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pinecone import Pinecone, ServerlessSpec

# Version information
VERSION = "1.1.0"

# Debug flag - Set to True to enable debug output, False to disable
DEBUG = False

# Download required NLTK data packages
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

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
    print(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Using existing Pinecone index: {index_name}")

pinecone_index = pc.Index(index_name)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore(
    index=pinecone_index, 
    embedding=embedding_model, 
    text_key="text"
)

# Current user tracking
current_user_identity = "unknown"

# Function to explicitly get identity from LLM
def get_user_identity(user_input):
    prompt = (
        f"Given the following input from a user, what's the user's name? "
        f"If the name isn't explicitly provided, answer 'unknown'.\n\n"
        f"User input: {user_input}\n\n"
        f"Name:"
    )

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    identity = response["message"]["content"].strip().lower()
    # If the response has multiple words or punctuation, extract the first word
    identity = (identity.split()[0].strip(".,!?") 
                if identity and identity != "unknown" else "unknown")
    return identity

# Exponential decay function for memory degradation
def calculate_decay_factor(timestamp, half_life_days=90):
    now = datetime.now()
    # Days
    time_diff = (now - datetime.fromisoformat(timestamp)).total_seconds() / (24 * 3600)
    decay = np.exp(-time_diff / half_life_days)  # Exponential decay
    return decay

# Function to determine importance using VADER sentiment analysis
def calculate_importance(user_input, response):
    # Use VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    user_sentiment = sid.polarity_scores(user_input)
    response_sentiment = sid.polarity_scores(response)
    
    # Use the compound score which is normalized between -1 and 1
    # Average the sentiment of both user input and response
    importance = (user_sentiment['compound'] + response_sentiment['compound']) / 2
    
    return importance  # Returns value between -1 and 1

# Store conversation with decay and importance
def store_conversation(user_input, response, user_identity="unknown"):
    conversation_text = f"User: {user_input}\nAI: {response}"
    importance = calculate_importance(user_input, response)
    timestamp = datetime.now().isoformat()
    
    # Create metadata for retrieval
    metadata = {
        "text": conversation_text,
        "timestamp": timestamp,
        "importance": importance,
        "user_identity": user_identity,
        "id": str(uuid.uuid4())
    }
    
    # Embed the text
    embedding = embedding_model.embed_documents([conversation_text])[0]
    
    # Amplify embedding based on importance (helps important memories stand out)
    amplified_embedding = [v * (1 + importance) for v in embedding]
    
    # Store in Pinecone with unique ID
    vector_store.add_texts(
        texts=[conversation_text],
        embeddings=[amplified_embedding],
        metadatas=[metadata]
    )
    
    # Only print debug information if DEBUG flag is set
    if DEBUG:
        print(f"Stored conversation with importance: {importance:.2f}")

# Retrieve context with decay and importance
def retrieve_context(query, max_results=3, decay_threshold=0.05):
    # Perform similarity search
    # Get more results than needed to filter
    results = vector_store.similarity_search_with_score(
        query=query, 
        k=max_results*2
    )
    
    # Apply decay factor to each result
    weighted_results = []
    for doc, score in results:
        if (hasattr(doc, 'metadata') and 'timestamp' in doc.metadata 
                and 'importance' in doc.metadata):
            decay = calculate_decay_factor(doc.metadata['timestamp'])
            importance = doc.metadata['importance']
            
            # Apply both decay and importance to the relevance
            # Higher importance = slower decay
            # importance augments the decay rate
            effective_weight = decay * (1 + importance)
            
            # Only include if above threshold
            if effective_weight >= decay_threshold:
                weighted_results.append((doc, effective_weight))
    
    # Sort by effective weight (descending) and take top results
    weighted_results.sort(key=lambda x: x[1], reverse=True)
    weighted_results = weighted_results[:max_results]
    
    # Format context for prompt
    context = "Previous conversation (faded by time, stronger if important):\n\n"
    for doc, weight in weighted_results:
        # Format with decay-based opacity
        context += f"[Memory strength: {weight:.2f}] {doc.page_content}\n\n"
    
    return context

# Get response with memory
def get_response_with_memory(user_input):
    global current_user_identity

    # Detect identity explicitly only if unknown
    if current_user_identity == "unknown":
        detected_identity = get_user_identity(user_input)
        if detected_identity != "unknown":
            current_user_identity = detected_identity

    context = retrieve_context(user_input)

    # Strong explicit instructions to prevent repetitive comments
    prompt = (
        f"You are a helpful and conversational assistant by the name of James. "
        f"Below is context from past interactions."
        f"Use this context implicitly to inform your responses. "
        f"NEVER mention or imply that you're starting again, "
        f"having memory issues, forgetting, repeating, or experiencing loops. "
        f"IF there's insufficient context, simply PROCEED NATURALLY.\n\n"
        f"Context:\n{context}\n\n"
        f"User: {user_input}\nAI:"
    )

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_response = response["message"]["content"]

    store_conversation(user_input, ai_response, current_user_identity)
    return ai_response

# Version information functions
def display_version_info():
    print(f"Human-Like Memory Chatbot v{VERSION}")
    print("==============================")
    print("Using:")
    print(f"- Pinecone Index: {index_name}")
    print(f"- Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"- LLM: llama3.2:latest")
    print(f"- Debug Mode: {'Enabled' if DEBUG else 'Disabled'}")
    print("==============================")

# Main chat loop
def chat():
    display_version_info()
    print("\nStart chatting (type 'exit' to stop):")
    while True:
        user_input = input("\n--------\nYou: ")
        if user_input.lower() == "exit":
            break

        response = get_response_with_memory(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    chat()
