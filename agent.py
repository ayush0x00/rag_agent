from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
   model_name = "Qwen/Qwen2-0.5B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   
   messages = [
        {"role": "user", "content": "Hello! Can you introduce yourself?"}
    ]
   inputs = tokenizer.apply_chat_template(messages,return_tensors="pt").to(model.device)
   outputs = model.generate(inputs, max_new_tokens=100)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)

if __name__ == "__main__":
    main()