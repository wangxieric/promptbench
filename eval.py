import promptbench as pb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dataset = pb.DatasetLoader.load_dataset("sst2")
print("Dataset loaded:", dataset[:5])

model_names = ['meta-llama/Meta-Llama-3-8B', 'XiWangEric/literary-classicist-llama3', 'XiWangEric/inventive_technologist-llama3', 'XiWangEric/patent_strategist-llama3', 
    'XiWangEric/cultural_scholar-llama3', 'XiWangEric/technical_communicator-llama3', 'XiWangEric/business_advisor-llama3', 'XiWangEric/health_advisor-llama3', 
    'XiWangEric/scientific_scholar-llama3', 'XiWangEric/scientific_mathematician-llama3', 'XiWangEric/legal_analyst-llama3', 'XiWangEric/biomedical_expert-llama3']

for model_name in model_names:
    print(f"Evaluating model: {model_name}")
    model = pb.LLMModel(model_name, max_new_tokens=10, temperature=0.0001, device='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
    #                      "Determine the emotion of the following sentence as positive or negative: {content}"
    #                      ])
    prompts = pb.Prompt([
        "Sentence: '{content}'\nSentiment:",
        "Text: '{content}'\nLabel (positive/negative):",
        "The sentence '{content}' expresses a sentiment that is",
        "Based on the sentence '{content}', the sentiment is",
        "'{content}'\nThis sentence is",
        "Read the sentence: '{content}'\nSentiment:",
        "Given the sentence: '{content}'\nClass:",
        "Review the sentence: '{content}'\nLabel:",
        "Consider: '{content}'\nThe sentiment is either positive or negative.\nAnswer:",
        "Input: {content}\nSentiment (positive or negative):"
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

     # 🧹 Free memory before loading the next model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()

# Example 1: Plain language continuation
# print("=== Story prompt ===")
# print(model("Once upon a time, there was a lonely dragon who"))

# # Example 2: Prompt with answer template (no [INST] tags)
# print("\n=== Sentiment prompt ===")
# print(model("If I classify the sentence {content} as positive or negative, the answer will be:"))