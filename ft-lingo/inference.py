from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("out", trust_remote_code=True)



# import argparse
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, TaskType, get_peft_model

# def load_model(model_path, use_lora=True):
#     config = transformers.AutoConfig.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

#     if use_lora:
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             target_modules=["qkv_proj"],
#             inference_mode=True,
#             r=8,
#             lora_alpha=16,
#             lora_dropout=0.1,
#         )
#         model = get_peft_model(model, peft_config)

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     return model.to("cuda" if torch.cuda.is_available() else "cpu"), tokenizer

# def generate_response(model, tokenizer, input_text, max_length=512):
#     input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response

# def main():
#     parser = argparse.ArgumentParser(description='Interactive conversation with a trained LoRA model.')
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model directory.')
#     args = parser.parse_args()

#     model, tokenizer = load_model(args.model_path)
    
#     print("Model loaded. Start typing your messages:")
#     while True:
#         input_text = input("You: ")
#         if input_text.lower() in ["quit", "exit"]:
#             break
#         response = generate_response(model, tokenizer, input_text)
#         print(f"AI: {response}")

# if __name__ == '__main__':
#     main()
