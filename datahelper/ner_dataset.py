import numpy as np
from datasets import load_dataset as load_dataset_web, load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import os
from dataclasses import dataclass
from typing import Dict, Callable, Any, List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_token_classification_metrics(id2label: Dict[int, str]) -> Callable:
    """
    Create a metrics function for token classification
    
    Args:
        id2label: Mapping from label IDs to label names
        
    Returns:
        Metrics computation function
    """
    def compute_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute evaluation metrics for token classification
        
        Args:
            p: tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Extract true predictions and labels, filtering out padding tokens (-100)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Print detailed classification report for debugging
        print(classification_report(true_labels, true_predictions, zero_division=0))

        # Get detailed metrics as dictionary
        results = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        
        # Return comprehensive metrics
        return {
            "precision": results["micro avg"]["precision"],
            "recall": results["micro avg"]["recall"],
            "f1": results["micro avg"]["f1-score"],
            "accuracy": results["accuracy"] if "accuracy" in results else results["micro avg"]["precision"],
        }
    
    return compute_metrics


def load_dataset(key: str):
    try:
        ds = load_from_disk(key)
    except:  # noqa
        ds = load_dataset_web(key)
    ds["dev"] = ds.pop("validation", None) or ds["dev"]
    return ds


@dataclass
class NERDataset:
    """Class to hold all NER dataset components"""
    data_collator: DataCollatorForTokenClassification
    tokenized_datasets: Any
    tokenizer: Any
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    compute_metrics: Callable
    
    @property
    def num_labels(self) -> int:
        """Return the number of labels"""
        return len(self.label2id)


def load_ner_dataset(model_name: str, max_length=48) -> NERDataset:
    """
    Load and prepare NER dataset
    
    Args:
        model_name: Name of the model to use for tokenization
        max_length: Maximum sequence length
        
    Returns:
        NERDataset object containing all necessary components
    """
    NER_DS = load_dataset(".dataset/EstNER")
    # NER_DS = load_dataset("wnut_17")
    # NER_DS = load_dataset("conll2003")

    feature = NER_DS["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        clean_up_tokenization_spaces=True,
        max_length=max_length,
        add_prefix_space=True,
        padding_side="right",
    )

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding=False,
        )  # , add_special_tokens=False)#, padding=True)
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
    print(tokenized_datasets)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create a compute_metrics function with the id2label mapping
    metrics_fn = create_token_classification_metrics(id2label)

    return NERDataset(
        data_collator=data_collator,
        tokenized_datasets=tokenized_datasets,
        tokenizer=tokenizer,
        id2label=id2label,
        label2id=label2id,
        compute_metrics=metrics_fn
    )
