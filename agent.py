from transformers import AutoModelForCausalLM, AutoTokenizer
from indexer import Indexer, Embedder
import torch


class Agent:
    def __init__(
        self, model_name="Qwen/Qwen2-0.5B-Instruct", indexer=None, embedder=None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.indexer = indexer
        self.embedder = embedder
        self.conversation_history = []

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context from the index"""
        if not self.indexer or not self.embedder:
            return ""

        query_vector = self.embedder.encode(query)
        results = self.indexer.search_index(query_vector, top_k=top_k)

        context_parts = []
        for doc in results:
            context_parts.append(f"Source: {doc.source_file}\n{doc.text}")

        return "\n\n---\n\n".join(context_parts)

    def generate_response(self, user_query, use_rag=True):
        """Generate response using LLM with optional RAG"""
        # Retrieve context if RAG is enabled
        context = ""
        if use_rag and self.indexer:
            context = self.retrieve_context(user_query)
            if context:
                user_query = f"""Context from knowledge base:
{context}

User question: {user_query}

Please answer the user's question based on the provided context. If the context doesn't contain relevant information, say so."""

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_query})

        # Create prompt with conversation history
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            attention_mask=inputs.attention_mask,
            temperature=0.7,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (remove the prompt)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def chat_loop(self):
        """Main conversation loop"""
        print("Agent ready! Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("Agent: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)
            print()
