import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from datasets import load_dataset, load_from_disk

# NER_DS = load_dataset("tartuNLP/EstNER", "estner-reannotated", columns=["tokens", "ner_tags"])
NER_DS = load_from_disk("./EstNER")
# NER_DS = load_dataset("wnut_17")


"""
DatasetDict({
    train: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 9965
    })
    dev: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 2415
    })
    test: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 1907
    })
})
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define label list
# label_list = sorted(set(e for arr in NER_DS["train"]["ner_tags"] for e in arr))
label_list = NER_DS["train"].features[f"ner_tags"].feature.names
# print(label_list, len(label_list))
# print(NER_DS["train"].features[f"ner_tags"].feature.id)
# print(NER_DS["train"].features[f"ner_tags"].feature.str2id(lab) for lab in label_list)
# print(dir( NER_DS["train"].features[f"ner_tags"].feature))

# Load pretrained model and tokenizer
# model_name = "FacebookAI/xlm-roberta-small"  # You can change this to any other suitable pretrained model
model_name = "FacebookAI/xlm-roberta-base"  # You can change this to any other suitable pretrained model
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"  # You can change this to any other suitable pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True, max_length=40)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=40)#, padding=True)
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


# Metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(true_labels, true_predictions, output_dict=True, zero_division="precision")
    return {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": results["micro avg"]["f1-score"],
    }

# Apply tokenization to the dataset
tokenized_datasets = NER_DS.map(tokenize_and_align_labels, batched=True)
tokenized_datasets= tokenized_datasets.remove_columns(['ner_tags', 'tokens'])
print(tokenized_datasets)
# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))
print(config.attention_type if hasattr(config, 'attention_type') else "Standard attention")

# config.attention_type = 'efficient_attention'  # Check if your model supports this
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
# model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)#.half()
# model = torch.compile(model)
# from optimum.bettertransformer import BetterTransformer
#
# model = BetterTransformer.transform(model)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    report_to="none",
    logging_strategy='no',
    save_strategy="no",

    learning_rate=8e-5,
    num_train_epochs=16,
    weight_decay=0.01,
    max_grad_norm=0.5,

    gradient_accumulation_steps=2,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=128,

    # tf32=True,
    bf16=True,
    bf16_full_eval=True,

    # fp16=True,  # 50%, 50s
    # fp16_full_eval=True,

    # half_precision_backend="amp",
    # fp16_opt_level="O1",  # Optimization level for FP16

    dataloader_num_workers=16,  # Adjust based on your CPU cores
    dataloader_pin_memory=True,
    # remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
# eval_results = trainer.evaluate()
# print("eval ", eval_results)
#
# # Save the model
# # trainer.save_model("./ner_model")
#
# # Test the model
test_results = trainer.predict(tokenized_datasets["test"])
print("test", test_results.metrics)

# test {'test_loss': 0.109, 'test_precision': 0.809, 'test_recall': 0.854, 'test_f1': 0.831, 'test_runtime': 1.6295, 'test_samples_per_second': 1170.326, 'test_steps_per_second': 9.206}
