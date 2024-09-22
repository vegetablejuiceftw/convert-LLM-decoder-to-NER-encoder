import torch

from ner_dataset import load_ner_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig


# Load pretrained model and tokenizer
# model_name = "FacebookAI/xlm-roberta-small"  # You can change this to any other suitable pretrained model
model_name = "FacebookAI/xlm-roberta-large"  # You can change this to any other suitable pretrained model
# model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"  # You can change this to any other suitable pretrained model


data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics = load_ner_dataset(model_name)
num_labels = len(label_list)

config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))
print(config.attention_type if hasattr(config, 'attention_type') else "Standard attention")

# config.attention_type = 'efficient_attention'  # Check if your model supports this
# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
                                                        )
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

    learning_rate=3e-5,
    num_train_epochs=16,
    weight_decay=0.01,
    # max_grad_norm=0.5,

    # gradient_accumulation_steps=2,
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
print(trainer.evaluate())
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
# {'eval_loss': 0.070, 'eval_model_preparation_time': 0.324, 'eval_precision': 0.898, 'eval_recall': 0.893, 'eval_f1': 0.896, 'eval_runtime': 2.2226, 'eval_samples_per_second': 1462.223, 'eval_steps_per_second': 11.698, 'epoch': 9.0}
# test {'test_loss': 0.118, 'test_model_preparation_time': 0.324, 'test_precision': 0.859, 'test_recall': 0.869, 'test_f1': 0.864, 'test_runtime': 2.3121, 'test_samples_per_second': 1493.423, 'test_steps_per_second': 11.678}
