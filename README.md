# Human-Like Memory Chatbot with Pinecone and Ollama

This project implements a chatbot with human-like memory characteristics using **Pinecone** as a vector store and **Ollama** with the `llama3.2:latest` model for natural language processing. The system emulates memory degradation over time (based on an exponential forgetting curve) and assigns stronger imprints to important memories, mimicking how humans prioritize significant events.

## Features
- **Memory Degradation**: Older memories fade using an exponential decay function, with a configurable half-life (default: 90 days).
- **Importance Weighting**: Memories deemed important (via sentiment analysis) receive a stronger imprint, making them more retrievable. **(Done ✅)**
- **Persistent Storage**: Conversation history is stored in Pinecone, allowing scalable, long-term memory.
- **Contextual Responses**: The chatbot retrieves relevant past interactions to inform its responses implicitly, avoiding explicit mentions of memory loss or repetition. **(Done ✅)**
- **Dynamic User Identity Detection**: The user's identity is detected explicitly by the LLM from the user's input and stored as metadata in Pinecone. **(Done ✅)**

## Prerequisites
- **Python 3.8+**
- **Pinecone Account**: Sign up at [Pinecone](https://www.pinecone.io/) and obtain an API key.
- **Ollama**: Installed locally with the `llama3.2:latest` model pulled (`ollama pull llama3.2:latest`).
- **Git**: To clone this repository.

## Installation
1. **Clone the Repository**:
```bash
git clone https://github.com/MLidstrom/llm-memory.git
cd llm-memory
```
2. **Install Dependencies**:
```bash
pip install pinecone-client ollama langchain langchain_pinecone sentence-transformers numpy python-dotenv nltk
```
3. **Set Up Pinecone**:
- Create a `.env` file in the project root with your Pinecone API key:
```text
PINECONE_API_KEY=your-actual-api-key-here
```
4. **Run Ollama Locally**:
- Start the Ollama server with the `llama3.2:latest` model:
```bash
ollama run llama3.2:latest
```

## Usage
1. **Run the Chatbot**:
```bash
python chatbot.py
```
- Type your messages in the prompt (`You: `).
- Type `exit` to stop the chatbot.

## How It Works
- **Pinecone Vector Store**: Stores conversation embeddings with metadata (text, timestamp, importance, user identity).
- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dimensional vectors.
- **Memory Degradation**: Applies an exponential decay factor based on time elapsed since the memory was stored.
- **Importance Scoring**: Uses sentiment analysis (VADER) to assign an importance score (−1.0 to 1.0) to each memory, amplifying its embedding and retrieval weight.
- **User Identity Detection**: The user's identity is explicitly identified by the LLM based on user disclosures.

## Configuration
- **Half-Life**: Adjust `half_life_days` in `calculate_decay_factor` (default: 90 days) to control memory fade rate.
- **Importance Logic**: Modify `calculate_importance` to use custom rules.
- **Decay Threshold**: Change `decay_threshold` in `retrieve_context` (default: 0.05) to filter faded memories.

## Future Enhancements
- Implement a repetition boost for frequently mentioned topics.
- Periodically prune highly decayed memories from Pinecone.
- Support multi-user memory with user-specific filtering.

## Dependencies
- `pinecone-client`: Pinecone vector store integration.
- `ollama`: Local LLM inference with `llama3.2:latest`.
- `langchain` & `langchain_pinecone`: Framework for embeddings and vector store management.
- `sentence-transformers`: Embedding generation.
- `numpy`: Numerical operations for decay calculations.
- `python-dotenv`: Load environment variables from `.env` file.
- `nltk`: Sentiment analysis for importance scoring.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions, bug reports, or enhancements.

## Acknowledgments
- Built with [Pinecone](https://www.pinecone.io/) for vector storage.
- Powered by [Ollama](https://ollama.ai/) and the `llama3.2:latest` model.
- Inspired by human memory research and the forgetting curve.

---
*Last Updated: March 11, 2025*
