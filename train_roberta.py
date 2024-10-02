import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from ner_dataset import load_ner_dataset, RoundMetricsCallback

# Load pretrained model and tokenizer
# model_name = "FacebookAI/xlm-roberta-small"  # You can change this to any other suitable pretrained model
model_name = "FacebookAI/xlm-roberta-large"  # You can change this to any other suitable pretrained model
# model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"  # You can change this to any other suitable pretrained model


data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics = load_ner_dataset(model_name)
num_labels = len(label_list)

config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))
print(config.attention_type if hasattr(config, 'attention_type') else "Standard attention")

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config,
                                                        torch_dtype=torch.bfloat16,
                                                        )

training_args = TrainingArguments(
    output_dir="./results/roberta/",
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

trainer.train()

test_results = trainer.predict(tokenized_datasets["test"])
print("test", test_results.metrics)


"""
              precision    recall  f1-score   support

         LOC       0.87      0.88      0.88      1612
        MISC       0.66      0.74      0.70       684
         ORG       0.82      0.84      0.83      1626
         PER       0.96      0.96      0.96      1527

   micro avg       0.85      0.87      0.86      5449
   macro avg       0.83      0.86      0.84      5449
weighted avg       0.85      0.87      0.86      5449

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [00:02<00:00,  4.90it/s]
test {'test_loss': 0.10330799967050552, 'test_f1': 0.8621562952243127, 'test_runtime': 3.3174, 'test_samples_per_second': 1040.89, 'test_steps_per_second': 4.22}
"""