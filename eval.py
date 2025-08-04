import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "XiWangEric/literary-classicist-llama3"  # replace if needed

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

prompt = "Once upon a time, in a quiet village,"  # simple prefix prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Generated Text ===")
print(decoded)