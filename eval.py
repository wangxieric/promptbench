import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = pb.DatasetLoader.load_dataset("sst2")
print("Dataset loaded:", dataset[:5])

model = pb.LLMModel("meta-llama/Meta-Llama-3-8B", max_new_tokens=10, temperature=0.0001, device='cuda')

prompts = pb.Prompt([
    "Classify the sentence as either 'positive' or 'negative': {content}\nAnswer:",
    "Label the sentiment of the following sentence. Respond only with 'positive' or 'negative': {content}\nSentiment:",
    "If I classify the sentence {content} as positive or negative, the answer will be:",
    "Decide if the sentiment of this sentence is positive or negative. Just reply 'positive' or 'negative': {content}",
    "Instruction: Respond with only 'positive' or 'negative'. Sentence: {content}\nAnswer:"
])

def proj_func(pred):
    pred = pred.strip().lower()
    if "positive" in pred:
        return 1
    elif "negative" in pred:
        return 0
    else:
        return -1

from tqdm import tqdm
for prompt in prompts:
    preds = []
    labels = []
    raw_preds = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['label']
        raw_pred = model(input_text)
        raw_preds.append(raw_pred)
        # process output
        pred = pb.OutputProcess.cls(raw_pred, proj_func)
        preds.append(pred)
        labels.append(label)
    
    print(raw_preds[:5])
    print(preds[:5])
    print(labels[:5])
    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")



# Example 1: Plain language continuation
# print("=== Story prompt ===")
# print(model("Once upon a time, there was a lonely dragon who"))

# # Example 2: Prompt with answer template (no [INST] tags)
# print("\n=== Sentiment prompt ===")
# print(model("If I classify the sentence {content} as positive or negative, the answer will be:"))