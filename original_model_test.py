from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deeplang-ai/LingoWhale-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deeplang-ai/LingoWhale-8B", device_map="auto", trust_remote_code=True)

prompt = input("Prompt: ")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
pred = model.generate(input_ids=inputs["input_ids"], max_new_tokens=500)
pred = pred[0, inputs["input_ids"].shape[-1]:]
print("Response:", tokenizer.decode(pred, skip_special_tokens=True))
