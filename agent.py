from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
   model_name = "Qwen/Qwen2-0.5B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   
   messages = [
        {"role": "user", "content": "Hello! Can you introduce yourself?"}
    ]
   prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   outputs = model.generate(inputs.input_ids, max_new_tokens=100, attention_mask=inputs.attention_mask)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)

if __name__ == "__main__":
    main()