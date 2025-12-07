from utils import initialize_index, update_index
from agent import Agent
from indexer import Indexer, Embedder
import sys
import argparse


def run_agent(index_name: str = None):
    """Initialize and run the agent with optional RAG"""
    print("[Main] Initializing agent...")

    indexer = None
    embedder = None

    if index_name:
        print(f"[Main] Connecting to Redis index: {index_name}...")
        indexer = Indexer(host="localhost", port="6379", index_name=index_name)
        print("[Main] Loading embedding model...")
        embedder = Embedder()
        print(f"[Main] Connected to index: {index_name}")
    else:
        print("[Main] No index provided, running without RAG")

    # Initialize agent
    print("[Main] Loading LLM model...")
    agent = Agent(indexer=indexer, embedder=embedder)
    print("[Main] Agent initialized successfully!\n")

    # Start chat loop
    agent.chat_loop()


def main():
    parser = argparse.ArgumentParser(
        description="Personal Agent - Index documents and chat with RAG"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory path to data for indexing",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="test_index",
        help="Name of the Redis index (default: test_index)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["index", "agent", "both", "update"],
        default="both",
        help="Mode: 'index' (index only, skip if exists), 'agent' (agent only), 'both' (index then agent), 'update' (add to existing index)",
    )

    args = parser.parse_args()

    if args.mode == "index":
        if not args.data_dir:
            print("Error: --data_dir required for index mode")
            return
        initialize_index(args.data_dir, args.index_name)

    elif args.mode == "agent":
        run_agent(args.index_name)

    elif args.mode == "both":
        if not args.data_dir:
            print("Error: --data_dir required for 'both' mode")
            return
        initialize_index(args.data_dir, args.index_name)
        print("\n" + "=" * 50 + "\n")
        run_agent(args.index_name)

    elif args.mode == "update":
        if not args.data_dir:
            print("Error: --data_dir required for update mode")
            return
        update_index(args.data_dir, args.index_name)


if __name__ == "__main__":
    main()
