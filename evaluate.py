import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader

from modeling_bloom import BloomForTokenClassification


def setup():
    dataset = load_dataset("conll2003")
    feature = dataset["train"].features["ner_tags"].feature
    return (dataset["test"],
            {feature.int2str(i): i for i in range(feature.num_classes)},
            {i: feature.int2str(i) for i in range(feature.num_classes)})


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=32, padding=False)
    tokenized["labels"] = [
        [l[w] if w is not None and w != p else -100
        for w, p in zip(tokenized.word_ids(i), [None] + tokenized.word_ids(i)[:-1])]
        for i, l in enumerate(examples["ner_tags"])
    ]
    return tokenized


def predict(dataloader, model, id2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            preds = torch.argmax(outputs.logits, dim=2).cpu()
            for pred, label in zip(preds, batch["labels"]):
                all_preds.append([id2label[p.item()] for p, l in zip(pred, label) if l != -100])
                all_labels.append([id2label[l.item()] for l in label if l != -100])
    return all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name, batch_size = "results/checkpoint-110", 128
    test_dataset, label2id, id2label = setup()

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenized_dataset = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    # model = BloomForTokenClassification.from_pretrained(model_name).to(device)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

    dataloader = DataLoader(
        tokenized_dataset.select_columns(['input_ids', 'labels', 'attention_mask']),
        batch_size=batch_size,
        collate_fn=DataCollatorForTokenClassification(tokenizer))

    pred_labels, true_labels = predict(dataloader, model, id2label)

    print("True labels (first 2):", true_labels[:2], "\nPredicted labels (first 2):", pred_labels[:2])
    print("\nClassification Report:\n", classification_report(true_labels, pred_labels, zero_division=0))


if __name__ == "__main__":
    main()
