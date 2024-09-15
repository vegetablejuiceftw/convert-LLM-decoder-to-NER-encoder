"""
Train a state-of-the-art NER model using BERT and the OntoNotes dataset.

This script demonstrates the full pipeline from data preparation to model training
and inference for Named Entity Recognition (NER) using current best practices in NLP.
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from seqeval.metrics import classification_report

# Constants and configurations
MODEL_NAME = "bert-base-cased"
DATASET_NAME = "conll2012_ontonotes"
DATASET_CONFIG = "english_v4"
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./ner_model"

# Data loading and preprocessing
def load_and_preprocess_data():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
    return tokenized_datasets, tokenizer

def tokenize_and_align_labels(examples, tokenizer):
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

# Model setup
def setup_model(num_labels, id2label, label2id):
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model

# Training setup
def setup_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

# Evaluation and metrics
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


def main():
    # Load and preprocess data
    tokenized_datasets, tokenizer = load_and_preprocess_data()

    # Setup model
    id2label = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-NORP", 4: "I-NORP", ...}  # Define all labels
    label2id = {v: k for k, v in id2label.items()}
    model = setup_model(len(id2label), id2label, label2id)

    # Setup training arguments
    training_args = setup_training_args()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()

    # Save the model
    trainer.save_model(FINAL_MODEL_DIR)

    # Test the model
    nlp = pipeline("ner", model=FINAL_MODEL_DIR, tokenizer=tokenizer)
    text = "John Smith works at Microsoft in Seattle."
    results = nlp(text)
    print(results)

if __name__ == "__main__":
    main()
