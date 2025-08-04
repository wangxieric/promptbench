import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = pb.DatasetLoader.load_dataset("sst2")
print("Dataset loaded:", dataset[:5])

model = pb.LLMModel("meta-llama/Meta-Llama-3-8B", max_new_tokens=10, temperature=0.0001, device='cuda')

prompts = pb.Prompt([
    "Classify the sentiment of the sentence '{content}' as either positive or negative: ",
    "If I classify the sentence '{content}' as positive or negative, the answer will be: ",
    "The sentiment of the sentence '{content}' is (positive or negative): ",
    "Sentence: '{content}'\nSentiment (positive/negative): ",
    "Label the following sentence as positive or negative.\nSentence: '{content}'\nLabel: ",
    "Is the sentiment of the sentence '{content}' positive or negative? Answer: ",
    "Given the sentence '{content}', the correct sentiment label is: ",
    "Decide if the sentiment of '{content}' is positive or negative. It is: ",
    "Choose the sentiment of the sentence '{content}' (positive or negative): ",
    "The following sentence expresses a [positive/negative] sentiment: '{content}'\nAnswer: "
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