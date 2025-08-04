import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dataset = pb.DatasetLoader.load_dataset("sst2")
print("Dataset loaded:", dataset[:5])

model_name = "XiWangEric/technical_communicator-llama3"
model = pb.LLMModel(model_name, max_new_tokens=10, temperature=0.0001, device='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
                     "Determine the emotion of the following sentence as positive or negative: {content}"
                     ])

from tqdm import tqdm
for prompt in prompts:
    preds = []
    labels = []
    raw_preds = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['label']
        logits = model(input_text, output_logits=True)[0, -1]
        label_ids = [tokenizer(label).input_ids[-1] for label in ["positive", "negative"]]
        probs = torch.nn.functional.softmax(torch.tensor([logits[i] for i in label_ids]), dim=0)
        pred = int(probs[0] > probs[1]) # 0 for negative, 1 for positive
        preds.append(pred)
        labels.append(label)
    # print(raw_preds[:5])
    # print(preds[:5])
    # print(labels[:5])
    # # # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")



# Example 1: Plain language continuation
# print("=== Story prompt ===")
# print(model("Once upon a time, there was a lonely dragon who"))

# # Example 2: Prompt with answer template (no [INST] tags)
# print("\n=== Sentiment prompt ===")
# print(model("If I classify the sentence {content} as positive or negative, the answer will be:"))