# Personal Agent

A RAG (Retrieval Augmented Generation) system that indexes documents and enables conversational AI with context-aware responses using vector search and LLMs.

## Features

- **Document Indexing**: Index text files (.txt, .md) into Redis with vector embeddings
- **Vector Search**: Semantic search using cosine similarity on document embeddings
- **RAG Agent**: Conversational AI agent that retrieves relevant context before generating responses
- **Smart Indexing**: Skips re-indexing if index already exists
- **Index Updates**: Add new documents to existing indexes without recreating them

## Architecture

The system consists of three main components:

1. **Chunker**: Splits text into manageable chunks for processing
2. **Embedder**: Converts text chunks into vector embeddings using sentence transformers
3. **Indexer**: Stores and searches embeddings in Redis using RedisSearch

## Requirements

- Python 3.8+
- Redis Stack (with RedisSearch module) - running in Docker
- Required Python packages:
  - `redis`
  - `sentence-transformers`
  - `transformers`
  - `torch`
  - `numpy`
  - `tqdm`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PersonalAgent
```

2. Install dependencies:
```bash
pip install redis sentence-transformers transformers torch numpy tqdm
```

3. Start Redis Stack with Docker:
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Verify Redis is running:
```bash
docker exec redis-stack redis-cli PING
# Should return: PONG
```

## Usage

### Index Documents

Index a directory of text files (skips if index already exists):
```bash
python main.py --data_dir test_directory --mode index --index-name my_index
```

### Update Index

Add new documents to an existing index:
```bash
python main.py --data_dir new_files --mode update --index-name my_index
```

### Run Agent

Start the conversational agent with RAG:
```bash
python main.py --index-name my_index --mode agent
```

### Index and Run Agent

Index documents and immediately start the agent:
```bash
python main.py --data_dir test_directory --mode both --index-name my_index
```

### Run Agent Without RAG

Run the agent without any indexed context:
```bash
python main.py --mode agent
```

## Command Line Arguments

- `--data_dir`: Directory path containing text files to index (.txt, .md)
- `--index-name`: Name of the Redis index (default: `test_index`)
- `--mode`: Operation mode
  - `index`: Index documents only (skips if index exists)
  - `agent`: Run agent only (requires existing index)
  - `both`: Index then run agent
  - `update`: Add new documents to existing index

## Project Structure

```
PersonalAgent/
├── main.py          # Entry point - handles initialization and CLI
├── agent.py          # Agent class with RAG and chat loop
├── indexer.py        # Core classes: Chunker, Embedder, Indexer
├── utils.py          # Utility functions for indexing
├── test_directory/   # Sample text files for testing
└── README.md         # This file
```

## How It Works

1. **Indexing**:
   - Text files are chunked into smaller pieces (default: 3000 chars)
   - Each chunk is converted to a vector embedding using `all-MiniLM-L6-v2`
   - Chunks and embeddings are stored in Redis as JSON documents
   - RedisSearch creates a vector index for fast similarity search

2. **Search**:
   - User query is converted to an embedding
   - KNN (K-Nearest Neighbors) search finds most similar document chunks
   - Results are ranked by cosine similarity

3. **RAG Agent**:
   - Retrieves top-k relevant chunks from the index
   - Builds a prompt with context + user question
   - Generates response using LLM (Qwen2-0.5B-Instruct)
   - Maintains conversation history

## Configuration

### Embedding Model

Default: `all-MiniLM-L6-v2` (384 dimensions)

To change, modify `Embedder` class in `indexer.py`:
```python
embedder = Embedder(model_name="your-model-name")
```

### LLM Model

Default: `Qwen/Qwen2-0.5B-Instruct`

To change, modify `Agent` class in `agent.py`:
```python
agent = Agent(model_name="your-model-name", ...)
```

### Chunk Size

Default: 3000 characters

To change:
```python
chunker = Chunker(chunk_size=5000)
```

## Redis Management

### Delete an Index

```bash
docker exec redis-stack redis-cli FT.DROPINDEX index_name
```

### List All Indexes

```bash
docker exec redis-stack redis-cli FT._LIST
```

### Flush Database

```bash
docker exec redis-stack redis-cli FLUSHDB
```

## Troubleshooting

### Redis Connection Error

Make sure Redis Stack is running:
```bash
docker ps | grep redis
```

### Index Already Exists

The system automatically skips indexing if the index exists. Use `--mode update` to add new documents.

### Module Not Found

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

