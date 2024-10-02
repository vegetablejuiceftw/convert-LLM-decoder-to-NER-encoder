# -*- coding: utf-8 -*-

import argparse

import numpy as np
import evaluate
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForTokenClassification, MistralForTokenClassification, OpenELMForTokenClassification, \
    Qwen2ForTokenClassification

"""
dotenv run -- python billm_ner.py \
--model_name_or_path mistralai/Mistral-7B-v0.3 \
--dataset_name_or_path conll2003

"""
"""
dotenv run -- python billm_ner.py \
--model_name_or_path Qwen/Qwen2-7B \
--dataset_name_or_path EstNER

"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='NousResearch/Llama-2-7b-hf',
                    help='Specify model_name_or_path to set transformer backbone. Default is NousResearch/Llama-2-7b-hf')
parser.add_argument('--dataset_name_or_path', type=str, default='conll2003',
                    help='Specify huggingface dataset name or local file path. Default is conll2003.')
parser.add_argument('--epochs', type=int, default=3, help='Specify number of epochs, default 10')
parser.add_argument('--batch_size', type=int, default=16, help='Specify number of batch size, default 8')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify learning rate, default 1e-4')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=48, help='Specify max length, default 64')
parser.add_argument('--lora_r', type=int, default=32, help='Specify lora r, default 12')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora alpha, default 0.1')
# configure hub
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify push_to_hub_model_id, default None, format like organization/model_id')
args = parser.parse_args()
print(f'Args: {args}')


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if 'mistral' in args.model_name_or_path.lower():
    tokenizer.add_special_tokens({'pad_token': '<unk>'})

seqeval = evaluate.load("seqeval")
if args.dataset_name_or_path == 'wnut_17':
    ds = load_dataset("wnut_17")
    label2id = { "O": 0, "B-corporation": 1, "I-corporation": 2, "B-creative-work": 3, "I-creative-work": 4, "B-group": 5, "I-group": 6, "B-location": 7, "I-location": 8, "B-person": 9, "I-person": 10, "B-product": 11, "I-product": 12, }
elif args.dataset_name_or_path == 'conll2003':
    ds = load_dataset("conll2003")
    label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
else:
    # raise NotImplementedError
    ds = load_from_disk(args.dataset_name_or_path)
    feature = ds["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    print(label2id)


ds['dev'] = ds.pop('validation', None) or ds['dev']


id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())
if 'mistral' in args.model_name_or_path.lower():
    MODEL = MistralForTokenClassification
elif 'llama' in args.model_name_or_path.lower():
    MODEL = LlamaForTokenClassification
elif 'qwen' in args.model_name_or_path.lower():
    MODEL = Qwen2ForTokenClassification
else:
    print(args.model_name_or_path)
    raise NotImplementedError
model = MODEL.from_pretrained(
    args.model_name_or_path, num_labels=len(label2id), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16, device_map="auto",
)
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                         inference_mode=False,
                         r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=args.max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


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

    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir=f"results/billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_ckpt_2",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 2,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# print(trainer.evaluate())
trainer.train()
test_results = trainer.predict(tokenized_ds["test"])
print("test", test_results.metrics)


# push the best model to the hub
# if args.push_to_hub:
#     trainer.push_to_hub()

# {'eval_loss': 0.12563484907150269, 'eval_precision': 0.9250177683013504, 'eval_recall': 0.9373424558876485, 'eval_f1': 0.9311393310677876, 'eval_accuracy': 0.9865259033521985, 'eval_runtime': 65.6237, 'eval_samples_per_second': 52.618, 'eval_steps_per_second': 6.583, 'epoch': 3.0}