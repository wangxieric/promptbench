import promptbench as pb
print('All supported datasets: ')
print(pb.SUPPORTED_DATASETS)
dataset = pb.DatasetLoader.load_dataset("sst2")
print('dataset samples: ', dataset[:5])

# print all supported models in promptbench
print('All supported models: ')
print(pb.SUPPORTED_MODELS)

# load a model, e.g., literary-classicist-llama3 with improved settings
model = pb.LLMModel(
    model='XiWangEric/literary-classicist-llama3',
    max_new_tokens=50,
    temperature=0.0001,
    system_prompt="You are a helpful assistant."
)

# Use correct placeholder for SST-2 data
prompts = pb.Prompt(["Classify the sentiment of the following sentence as positive or negative: {context}"])

print("test prompt output: ", model("What is the sentiment of the sentence: 'It was a great movie.'"))

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred.strip().lower(), -1)

from tqdm import tqdm
for prompt in prompts:
    preds = []
    labels = []
    raw_preds = []
    input_texts = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        print("input_text:", input_text)
        label = data['label']
        raw_pred = model(input_text)
        raw_preds.append(raw_pred)
        # process output
        pred = pb.OutputProcess.cls(raw_pred, proj_func)
        preds.append(pred)
        labels.append(label)

    print("predictions:", preds)
    print("labels:", labels)
    print("raw predictions:", raw_preds)

    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")
