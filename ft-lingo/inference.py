from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
import transformers
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora-path", type=str, default=None,
                    help="Path to the LoRA model checkpoint")
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained("deeplang-ai/LingoWhale-8B",trust_remote_code=True)
#model = AutoPeftModelForCausalLM.from_pretrained("out/1001/", trust_remote_code=True).to("cuda")
model = transformers.AutoModel.from_pretrained("deeplang-ai/LingoWhale-8B", trust_remote_code=True, device_map="auto").to("cuda")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    target_modules=['qkv_proj'],
    r=8, lora_alpha=16, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(args.lora_path), strict=False)

while True:
    prompt = input("Prompt: ")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(inputs["input_ids"].shape)
    response = model.generate(input_ids=inputs["input_ids"],
                              max_length=inputs["input_ids"].shape[-1] + 500)
    print(response.shape)
    response = response[0, inputs["input_ids"].shape[-1]:]
    print("Response:", tokenizer.decode(response, skip_special_tokens=True))

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
