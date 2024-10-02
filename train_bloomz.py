import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import Trainer, TrainingArguments, BloomForTokenClassification as OriginalBloomForTokenClassification
from modeling_bloom import BloomBlock, BloomAttention, BloomConfig, BloomModel, BloomForTokenClassification

from ner_dataset import load_ner_dataset, RoundMetricsCallback

model_name = "bigscience/bloomz-560m"
# model_name = "bigscience/bloomz-1b1"


data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics = load_ner_dataset(model_name, max_length=32)
num_labels = len(label_list)

# model = BloomForTokenClassification.from_pretrained(model_name, num_labels=num_labels)#, torch_dtype=torch.bfloat16)
model = OriginalBloomForTokenClassification.from_pretrained(model_name, num_labels=num_labels)#, torch_dtype=torch.bfloat16)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=32,

    report_to="none",
    logging_strategy='no',
    # save_strategy="no",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,

    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=0.5,

    num_train_epochs=2,
    warmup_steps=32,
    # gradient_accumulation_steps=2,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,

    bf16=True,
    bf16_full_eval=True,

    dataloader_num_workers=16,  # Adjust based on your CPU cores
    dataloader_pin_memory=True,
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
    callbacks=[RoundMetricsCallback(decimal_places=2)],
)

# OG:   {'eval_loss': 0.11, 'eval_precision': 0.78, 'eval_recall': 0.83, 'eval_f1': 0.80, 'eval_runtime': 3.2586, 'eval_samples_per_second': 997.361, 'eval_steps_per_second': 7.979, 'epoch': 5.0}
# LOCAL {'eval_loss': 0.18, 'eval_precision': 0.70, 'eval_recall': 0.80, 'eval_f1': 0.75, 'eval_runtime': 3.3512, 'eval_samples_per_second': 969.816, 'eval_steps_per_second': 7.759, 'epoch': 5.0}

trainer.train()

test_results = trainer.predict(tokenized_datasets["test"])
print("test", test_results.metrics)


"""
              precision    recall  f1-score   support

         LOC       0.72      0.82      0.77      1565
        MISC       0.62      0.69      0.65       655
         ORG       0.59      0.69      0.64      1566
         PER       0.84      0.83      0.83      1474

   micro avg       0.70      0.77      0.73      5260
   macro avg       0.69      0.76      0.72      5260
weighted avg       0.70      0.77      0.73      5260

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [00:02<00:00,  5.50it/s]
test {'test_loss': 0.15662416815757751, 'test_f1': 0.7299058653149892, 'test_runtime': 3.0311, 'test_samples_per_second': 1139.195, 'test_steps_per_second': 4.619}

"""