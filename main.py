from indexer import Embedder

def main():
    print("Hello from personalagent!")
    
    embedder = Embedder()
    embedding = embedder.encode("Hello world")
    print(embedding)


if __name__ == "__main__":
    main()
