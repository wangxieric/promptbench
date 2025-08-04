import torch
import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "XiWangEric/literary-classicist-llama3"  # replace if needed

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# model.eval()

model = pb.load_model("XiWangEric/literary-classicist-llama3")
prompt = "Once upon a time, in a quiet village,"  # simple prefix prompt
outputs = model(prompt)
print("=== Generated Text ===")
print(outputs)