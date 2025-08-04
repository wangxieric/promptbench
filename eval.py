import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

def generate(prompt, max_new_tokens=30, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # important for avoiding warning
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# Example 1: Plain language continuation
print("=== Story prompt ===")
print(generate("Once upon a time, there was a lonely dragon who"))

# Example 2: Prompt with answer template (no [INST] tags)
print("\n=== Sentiment prompt ===")
print(generate("I love programming. Sentiment:"))

# import torch
# import promptbench as pb
# from transformers import AutoTokenizer, AutoModelForCausalLM

# dataset = pb.DatasetLoader.load_dataset("sst2")
# print("dataset loaded:", dataset[:5])

# model = pb.LLMModel("meta-llama/Meta-Llama-3-8B", max_new_tokens=10, temperature=0.0001, device='cuda')
# prompt = "Once upon a time, in a quiet village,"  # simple prefix prompt
# outputs = model(prompt)

# print("=== Generated Text ===")
# print(outputs)

# prompt = "classify the sentiment of the following sentence: 'I love programming!'"
# outputs = model(prompt)
# print("=== Generated Text ===")
# print(outputs)

# print("=== Generated Text ===")
# prompts = [
#     "<s>[INST] Classify the sentiment of this sentence: 'I love programming!' [/INST]",
#     "<s>[INST] Classify the sentiment of this sentence: 'The movie was terrible.' [/INST]",
#     "<s>[INST] Sentiment of 'She enjoyed every moment of the concert.'? [/INST]",
#     "<s>[INST] Sentiment of 'I hated that book.'? [/INST]"
# ]

# for p in prompts:
#     print("Prompt:", p)
#     print("Output:", model(p))
#     print("------")

# prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
#                      "Determine the emotion of the following sentence as positive or negative: {content}"
#                      ])

# def proj_func(pred):
#     mapping = {
#         "positive": 1,
#         "negative": 0
#     }
#     return mapping.get(pred, -1)

# from tqdm import tqdm
# for prompt in prompts:
#     preds = []
#     labels = []
#     for data in tqdm(dataset):
#         # process input
#         input_text = pb.InputProcess.basic_format(prompt, data)
#         print(f"input_text: {input_text}")
#         label = data['label']
#         raw_pred = model(input_text)
#         # process output
#         pred = pb.OutputProcess.cls(raw_pred, proj_func)
#         preds.append(pred)
#         labels.append(label)
    
#     # evaluate
#     score = pb.Eval.compute_cls_accuracy(preds, labels)
#     print(f"{score:.3f}, {prompt}")