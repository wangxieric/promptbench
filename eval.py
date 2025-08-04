import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = pb.DatasetLoader.load_dataset("sst2")
print("Dataset loaded:", dataset[:5])

model = pb.LLMModel("meta-llama/Meta-Llama-3-8B", max_new_tokens=10, temperature=0.0001, device='cuda')

prompts = pb.Prompt([
    "If I classify the sentence '{content}' as positive or negative, the answer will be:",
    "The sentiment of the sentence '{content}' is clearly",
    "Sentence: '{content}'\nSentiment:",
    "Consider the sentence: '{content}' The sentiment is",
    "When evaluating the sentence '{content}', one would say it is",
    "Reading the sentence '{content}', I would label the sentiment as",
    "After reading: '{content}', the sentiment that comes to mind is",
    "'{content}' — this sentence expresses a",
    "Given this sentence: '{content}', its sentiment can be described as",
    "Let us classify the following sentence: '{content}'. It is"
])

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

from tqdm import tqdm
for prompt in prompts:
    preds = []
    labels = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['label']
        raw_pred = model(input_text)
        # process output
        pred = pb.OutputProcess.cls(raw_pred, proj_func)
        preds.append(pred)
        labels.append(label)
    
    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")



# Example 1: Plain language continuation
# print("=== Story prompt ===")
# print(model("Once upon a time, there was a lonely dragon who"))

# # Example 2: Prompt with answer template (no [INST] tags)
# print("\n=== Sentiment prompt ===")
# print(model("If I classify the sentence {content} as positive or negative, the answer will be:"))