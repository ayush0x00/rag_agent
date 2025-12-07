from typing import Any
from redis.commands.search.index_definition import IndexDefinition
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import torch
import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from redis.commands.search.query import Query
import numpy as np
import os


class Chunker:
    def __init__(self, chunk_size: int = 3000):
        self.chunk_size = chunk_size

    def chunk_pdf(self):
        pass

    def chunk_text(self, text: str, file_path: str = None):
        import hashlib

        file_hash = (
            hashlib.md5(file_path.encode()).hexdigest()[:8] if file_path else "unknown"
        )
        return [
            {
                "id": f"chunk_{file_hash}_{i:03d}",
                "text": text[i : i + self.chunk_size],
                "embedding": None,
                "metadata": {
                    "source_file": file_path,
                    "chunk_index": i,
                    "timestamp": time.time(),
                },
            }
            for i in range(0, len(text), self.chunk_size)
        ]

    def chunk_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.chunk_text(text, file_path)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Shifting model to {self.device}")
        self.model = SentenceTransformer(model_name).to(device=self.device)
        print(
            f"Loaded model {model_name} with embedding dim as {self.model.get_sentence_embedding_dimension()}"
        )

    def encode(self, text: str):
        return self.model.encode(text, device=self.device)


class Indexer:
    def __init__(self, host, port, index_name: str, vector_dim: int = 384):
        self.client = redis.Redis(host=host, port=port)
        self.index_name = index_name
        self.vector_dim = vector_dim

        if self.client.ping():
            print("Connected to redis ✌🏻")
        else:
            print("Failed to connect to Redis server! 😔")

    def index_exists(self):
        """Check if the index already exists"""
        try:
            info = self.client.ft(self.index_name).info()
            return info.get("index_name") == self.index_name
        except Exception:
            return False

    def create_search_index(self):
        """Create the search index if it doesn't exist"""
        if self.index_exists():
            print(f"Index {self.index_name} already exists")
            return True

        print(f"Creating index {self.index_name}")
        schema = (
            TextField("$.id", as_name="chunk_id"),
            TextField("$.text", as_name="text"),
            TextField("$.metadata.source_file", as_name="source_file"),
            NumericField("$.metadata.chunk_index", as_name="chunk_index"),
            NumericField("$.metadata.timestamp", as_name="timestamp"),
            VectorField(
                "$.embeddings",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": int(self.vector_dim),
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="embeddings",
            ),
        )

        index_definition = IndexDefinition(prefix=["chunk_"], index_type=IndexType.JSON)

        try:
            self.client.ft(self.index_name).create_index(
                fields=schema, definition=index_definition
            )
            print(f"{self.index_name} created ")
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            raise e

    def search_index(self, query_vector, top_k: int = 5):
        query = (
            Query(f"(*) => [KNN {top_k} @embeddings $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields(
                "id", "text", "source_file", "chunk_index", "timestamp", "vector_score"
            )
            .dialect(2)
        )

        try:
            results = self.client.ft(self.index_name).search(
                query,
                {"query_vector": np.array(query_vector, dtype=np.float32).tobytes()},
            )
            return results.docs
        except Exception as e:
            print(f"Error executing the query: {e}")
            raise e

    def delete_index(self):
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=False)
            print(f"{self.index_name} deleted")
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False

    def add_documents(self, chunks, embeddings):
        """
        Add documents to the index.

        Args:
            chunks: List of chunk dictionaries with 'id', 'text', and 'metadata' keys
            embeddings: List of numpy arrays or list of lists representing embeddings
        """
        try:
            pipe = self.client.pipeline()
            for chunk, embedding in zip(chunks, embeddings):
                doc_key = chunk["id"]
                embedding_array = np.array(embedding, dtype=np.float32)
                doc_data = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "embeddings": embedding_array.tolist(),
                    "metadata": {
                        "source_file": chunk["metadata"]["source_file"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "timestamp": chunk["metadata"]["timestamp"],
                    },
                }
                pipe.json().set(doc_key, "$", doc_data)
            pipe.execute()
            print(f"Added {len(chunks)} documents to index")
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e

    def index_directory(self, directory_path: str, chunker=None, embedder=None):
        """Index all text files from a directory"""
        from utils import get_text_files

        if chunker is None:
            chunker = Chunker()
        if embedder is None:
            embedder = Embedder()

        text_files = get_text_files(directory_path)
        if not text_files:
            print("No text files found to index")
            return

        for file in tqdm(text_files, desc="Chunking and indexing files"):
            chunks = chunker.chunk_file(file)
            embeddings = [embedder.encode(chunk["text"]) for chunk in chunks]
            self.add_documents(chunks, embeddings)
            print(f"Indexed {file}")


if __name__ == "__main__":
    import sys
    from utils import initialize_index

    directory_path = sys.argv[1]
    index_name = sys.argv[2]
    initialize_index(directory_path, index_name)
