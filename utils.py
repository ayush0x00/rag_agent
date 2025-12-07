from indexer import Indexer, Chunker, Embedder
from tqdm import tqdm
import os


def get_text_files(directory_path: str = None):
    """Get all text files from a directory"""
    if directory_path:
        text_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(".txt") or file.lower().endswith(".md"):
                    text_files.append(os.path.join(root, file))
        return text_files
    else:
        print("Provide directory to index !!")
        return None


def chunk_and_index_directory(chunker, indexer, embedder, directory_path: str):
    """Chunk and index all files from a directory"""
    text_files = get_text_files(directory_path)
    if not text_files:
        print("No text files found to index")
        return

    for file in tqdm(text_files, desc="Chunking and indexing files"):
        chunks = chunker.chunk_file(file)
        embeddings = [embedder.encode(chunk["text"]) for chunk in chunks]
        indexer.add_documents(chunks, embeddings)
        print(f"Indexed {file}")


def initialize_index(
    directory_path: str,
    index_name: str,
    host: str = "localhost",
    port: str = "6379",
):
    """Initialize and populate an index from a directory of text files.
    If index already exists, skips indexing."""
    available_text_files = get_text_files(directory_path)
    if not available_text_files:
        print("No text files found!")
        return None

    indexer = Indexer(host=host, port=port, index_name=index_name)

    # Check if index already exists
    if indexer.index_exists():
        print(f"Index '{index_name}' already exists. Skipping indexing.")
        print("Use update_index() to add new documents to existing index.")
        return indexer

    # Create index and index files
    indexer.create_search_index()
    embedder = Embedder()
    chunker = Chunker()
    chunk_and_index_directory(chunker, indexer, embedder, directory_path)

    # Test search
    results = indexer.search_index(
        embedder.encode("Tell me about machine learning"), top_k=5
    )
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {doc.vector_score}) ---")
        print(f"Source: {doc.source_file}")
        print(f"Text: {doc.text[:200]}...")

    return indexer


def update_index(
    directory_path: str,
    index_name: str,
    host: str = "localhost",
    port: str = "6379",
):
    """Update an existing index by adding new documents from a directory.
    Creates the index if it doesn't exist."""
    available_text_files = get_text_files(directory_path)
    if not available_text_files:
        print("No text files found!")
        return None

    indexer = Indexer(host=host, port=port, index_name=index_name)

    # Create index if it doesn't exist
    if not indexer.index_exists():
        print(f"Index '{index_name}' does not exist. Creating it...")
        indexer.create_search_index()
    else:
        print(f"Updating existing index '{index_name}'...")

    # Index files
    embedder = Embedder()
    chunker = Chunker()
    chunk_and_index_directory(chunker, indexer, embedder, directory_path)

    print(f"Index '{index_name}' updated successfully!")
    return indexer
