import torch
import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = pb.DatasetLoader.load_dataset("sst2")
print("dataset loaded:", dataset[:5])

model = pb.LLMModel("XiWangEric/literary-classicist-llama3", max_new_tokens=10, temperature=0.0001, device='cuda')
prompt = "Once upon a time, in a quiet village,"  # simple prefix prompt
outputs = model(prompt)

print("=== Generated Text ===")
print(outputs)

prompt = "classify the sentiment of the following sentence: 'I love programming!'"
outputs = model(prompt)
print("=== Generated Text ===")
print(outputs)

print("=== Generated Text ===")
prompts = [
    "I love programming! This means the sentiment is",
    "The movie was terrible. The sentiment is",
    "She enjoyed every moment of the concert. Sentiment:",
    "I hated that book. Sentiment:"
]
for p in prompts:
    print("Prompt:", p)
    print("Output:", model(p))
    print("------")

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