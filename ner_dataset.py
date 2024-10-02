import numpy as np
from datasets import load_dataset, load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, \
    TrainerCallback
from transformers import DataCollatorForTokenClassification
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_ner_dataset(model_name: str, max_length=48):
    # NER_DS = load_dataset("tartuNLP/EstNER", "estner-reannotated", columns=["tokens", "ner_tags"])
    # NER_DS = load_from_disk("./EstNER")
    # NER_DS = load_dataset("wnut_17")
    NER_DS = load_dataset("conll2003")
    NER_DS['dev'] = NER_DS.pop('validation', None) or NER_DS['dev']

    feature = NER_DS["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}

    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True, max_length=max_length, add_prefix_space=True, padding_side='right')

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length, padding=False)#, add_special_tokens=False)#, padding=True)
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
    tokenized_datasets = NER_DS.map(tokenize_and_align_labels, batched=True, batch_size=8)
    tokenized_datasets = tokenized_datasets
    print(tokenized_datasets)
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metrics function for evaluation
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

        print(classification_report(true_labels, true_predictions, zero_division=0))

        results = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        return {
            # "precision": results["micro avg"]["precision"],
            # "recall": results["micro avg"]["recall"],
            "f1": results["micro avg"]["f1-score"],
        }

    return data_collator, tokenized_datasets, tokenizer, id2label, compute_metrics



class RoundMetricsCallback(TrainerCallback):
    def __init__(self, decimal_places=4):
        self.decimal_places = decimal_places

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, self.decimal_places)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            value[sub_key] = round(sub_value, self.decimal_places)