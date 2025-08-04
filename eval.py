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
    temperature=0.1
    )

# Use correct placeholder for SST-2 data
# prompts = pb.Prompt(["Classify the sentiment of the following sentence as positive or negative: {content}"])

print("test prompt output: ", model("What is the sentiment of the sentence: 'It was a great movie.'"))

# def proj_func(pred):
#     mapping = {
#         "positive": 1,
#         "negative": 0
#     }
#     return mapping.get(pred.strip().lower(), -1)

# from tqdm import tqdm
# for prompt in prompts:
#     preds = []
#     labels = []
#     raw_preds = []
#     input_texts = []
#     for data in tqdm(dataset):
#         # process input
#         input_text = pb.InputProcess.basic_format(prompt, data)
#         print("input_text:", input_text)
#         label = data['label']
#         input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids.to(model.model.device)
#         logits = model.model(input_ids=input_ids).logits[0, -1]  # logits of last token

#         # Get the token IDs for 'positive' and 'negative'
#         positive_id = model.tokenizer("positive").input_ids[-1]
#         negative_id = model.tokenizer("negative").input_ids[-1]

#         # Compute softmax over those two logits
#         probs = torch.nn.functional.softmax(torch.tensor([logits[positive_id], logits[negative_id]]), dim=0)
#         pred = int(torch.argmax(probs).item())  # 0 = negative, 1 = positive

#         # raw_pred = f"positive: {probs[1].item():.3f}, negative: {probs[0].item():.3f}"
#         # # raw_pred = model(input_text)
#         # raw_preds.append(raw_pred)
#         # # process output
#         # pred = pb.OutputProcess.cls(raw_pred, proj_func)
#         preds.append(pred)
#         labels.append(label)

#     print("predictions:", preds)
#     print("labels:", labels)
#     print("raw predictions:", raw_preds)

#     # evaluate
#     score = pb.Eval.compute_cls_accuracy(preds, labels)
#     print(f"{score:.3f}, {prompt}")
