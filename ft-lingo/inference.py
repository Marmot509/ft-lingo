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
model = transformers.AutoModelForCausalLM.from_pretrained("deeplang-ai/LingoWhale-8B", trust_remote_code=True, device_map="auto").to("cuda")

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

