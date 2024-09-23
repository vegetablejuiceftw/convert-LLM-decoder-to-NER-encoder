import numpy as np
from datasets import load_dataset, load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from transformers import DataCollatorForTokenClassification
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_ner_dataset(model_name: str, max_length=40):
    # NER_DS = load_dataset("tartuNLP/EstNER", "estner-reannotated", columns=["tokens", "ner_tags"])
    # NER_DS = load_from_disk("./EstNER")
    # NER_DS = load_dataset("wnut_17")
    NER_DS = load_dataset("conll2003")
    NER_DS['dev'] = NER_DS.pop('validation')

    label_list = NER_DS["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True, max_length=max_length, add_prefix_space=True, padding_side='right')

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length, add_special_tokens=False)#, padding=True)
        # print(len(tokenized_inputs.tokens(0)), tokenized_inputs.tokens(0), len(examples["tokens"][0]), examples["tokens"][0])

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


    # Apply tokenization to the dataset
    tokenized_datasets = NER_DS.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets = tokenized_datasets.select_columns(['input_ids', 'labels', 'attention_mask'])
    print(tokenized_datasets)
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metrics function for evaluation
    def compute_metrics(p):
        predictions, labels = p
        print(predictions.shape)
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        for i in range(2):
            print(true_predictions[i])
            print(true_labels[i])
            print()

        results = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        return {
            "precision": results["micro avg"]["precision"],
            "recall": results["micro avg"]["recall"],
            "f1": results["micro avg"]["f1-score"],
        }

    return data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics
