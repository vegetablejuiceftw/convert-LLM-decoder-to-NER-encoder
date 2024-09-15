"""
The overall goal is to create a state-of-the-art NER model using modern transformer architectures. We're leveraging the power of pre-trained language models (BERT in this case) and fine-tuning them on a specific task (NER) using a high-quality dataset (OntoNotes). This approach typically yields better results than traditional machine learning methods or even some earlier deep learning approaches.

The code demonstrates the full pipeline from data preparation to model training and inference, making it a comprehensive example of how to tackle NER tasks using current best practices in NLP.
"""

from datasets import load_dataset

dataset = load_dataset("conll2012_ontonotes", "english_v4")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

id2label = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-NORP", 4: "I-NORP", ...}  # Define all labels
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

from seqeval.metrics import classification_report
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": results["micro avg"]["f1-score"],
        "accuracy": results["accuracy"],
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()
trainer.save_model("./ner_model")


from transformers import pipeline

nlp = pipeline("ner", model="./ner_model", tokenizer=tokenizer)
text = "John Smith works at Microsoft in Seattle."
results = nlp(text)
print(results)
