# Human-Like Memory Chatbot with Pinecone and Ollama

This project implements a chatbot with human-like memory characteristics using **Pinecone** as a vector store and **Ollama** with the `llama3.1:latest` model for natural language processing. The system emulates memory degradation over time (based on an exponential forgetting curve) and assigns stronger imprints to important memories, mimicking how humans prioritize significant events.

## Features
- **Memory Degradation**: Older memories fade using an exponential decay function, with a configurable half-life (default: 30 days).
- **Importance Weighting**: Memories deemed important (e.g., based on keywords like "important" or "love") receive a stronger imprint, making them more retrievable.
- **Persistent Storage**: Conversation history is stored in Pinecone, allowing scalable, long-term memory.
- **Contextual Responses**: The chatbot retrieves relevant past interactions to inform its responses, powered by Ollama's `llama3.1:latest`.

## Prerequisites
- **Python 3.8+**
- **Pinecone Account**: Sign up at [Pinecone](https://www.pinecone.io/) and obtain an API key.
- **Ollama**: Installed locally with the `llama3.1:latest` model pulled (`ollama pull llama3.1:latest`).
- **Git**: To clone this repository.

## Installation
1. **Clone the Repository**:
```bash
git clone https://github.com/MLidstrom/llm-memory.git
cd llm-memory
```
2. **Install Dependencies**:
```bash
pip install pinecone-client ollama langchain langchain_pinecone sentence-transformers numpy python-dotenv
```
3. **Set Up Pinecone**:
- Create a `.env` file in the project root with your Pinecone API key:
```text
PINECONE_API_KEY=your-actual-api-key-here
```
4. **Run Ollama Locally**:
- Start the Ollama server with the `llama3.1:latest` model:
```bash
ollama run llama3.1:latest
```
## Usage
1. **Run the Chatbot**:
```bash
python chatbot.py
```
- Type your messages in the prompt (`You: `).
- Type `exit` to stop the chatbot.

2. **Example Interaction**:
```text
You: Hello again
AI: It looks like we're starting from scratch again! Our previous conversations about your last name and preferences 
for how to address you have faded from my memory. But I'm happy to learn more about you now!

So, let's try this again... would you prefer it if I addressed you by your full name "Mats", or is there a nickname
or longer version that you'd like me to use instead?
You: My full name is "Mats Lidström", but please call me "Mats" as it is shorter.
AI: It looks like we're having a bit of déjà vu here! I think I see what's happening. You've already told me that
you prefer to be called just "Mats". There's no need for the full name or any variations.

Let's move on and start fresh, shall we? What would you like to talk about today?
You:
```
Over time (e.g., days or weeks), less important memories fade, while important ones remains prominent.

## How It Works
- **Pinecone Vector Store**: Stores conversation embeddings with metadata (text, timestamp, importance).
- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dimensional vectors.
- **Memory Degradation**: Applies an exponential decay factor based on time elapsed since the memory was stored.
- **Importance Scoring**: Assigns a score (1.0–5.0) to each memory based on keywords, amplifying its embedding and retrieval weight.
- **Retrieval**: Combines similarity, decay, and importance to fetch the most relevant past context for responses.

## Configuration
- **Half-Life**: Adjust `half_life_days` in `calculate_decay_factor` (default: 30 days) to control memory fade rate.
- **Importance Logic**: Modify `calculate_importance` to use custom rules (e.g., sentiment analysis, user tags).
- **Decay Threshold**: Change `decay_threshold` in `retrieve_context` (default: 0.1) to filter faded memories.

## Future Enhancements
- Add sentiment analysis for more accurate importance scoring.
- Implement a repetition boost for frequently mentioned topics.
- Periodically prune highly decayed memories from Pinecone.
- Support multi-user memory with user-specific filtering.

## Dependencies
- `pinecone-client`: Pinecone vector store integration.
- `ollama`: Local LLM inference with `llama3.1:latest`.
- `langchain` & `langchain_pinecone`: Framework for embeddings and vector store management.
- `sentence-transformers`: Embedding generation.
- `numpy`: Numerical operations for decay calculations.
- `python-dotenv`: Load environment variables from `.env` file.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions, bug reports, or enhancements.

## Acknowledgments
- Built with [Pinecone](https://www.pinecone.io/) for vector storage.
- Powered by [Ollama](https://ollama.ai/) and the `llama3.1:latest` model.
- Inspired by human memory research and the forgetting curve.

---
*Last Updated: February 22, 2025*