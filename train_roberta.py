import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from datahelper.ner_dataset import prepare_ner_dataset, DATASETS
from datahelper.utils import RoundMetricsCallback, get_trainable_linear_modules
from datahelper.datasampler import ResamplingDataset

# Load pretrained model and tokenizer
# model_name = "FacebookAI/xlm-roberta-base"  # You can change this to any other suitable pretrained model
model_name = "FacebookAI/xlm-roberta-large"  # You can change this to any other suitable pretrained model
# model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"  # You can change this to any other suitable pretrained model

torch_dtype = torch.bfloat16
# torch_dtype = torch.float32

ner_dataset = prepare_ner_dataset(DATASETS.CONLL, model_name)


config = AutoConfig.from_pretrained(model_name, num_labels=ner_dataset.num_labels)

# model = AutoModelForTokenClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config,
    # torch_dtype=torch_dtype,  # causes learning effectiveness to drop?
    ignore_mismatched_sizes=True,

    # attn_implementation="flash_attention_2",
    # use_cache=False,
    trust_remote_code=True,
    device_map="auto",
)
print(model)

trainable_layers = 8
for name, param in model.roberta.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

for layer in model.roberta.encoder.layer[-trainable_layers:]:
    for param in layer.parameters():
        param.requires_grad = True

trainable_linears = get_trainable_linear_modules(model)
print(f"Found {len(trainable_linears)} trainable linear layers", trainable_linears)

from peft import get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                       # Rank of the update matrices
    lora_alpha=32,              # Parameter for scaling
    lora_dropout=0.1,           # Dropout probability for LoRA layers
    target_modules=trainable_linears,  # Which modules to apply LoRA to
    bias="none",
    modules_to_save=["classifier"],  # Save the classifier if fine-tuning for classification
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Shows % of trainable parameters


print(config.name_or_path, config.attention_type if hasattr(config, "attention_type") else "Standard attention")

training_args = TrainingArguments(
    output_dir="./results/roberta/",
    # eval_strategy="no",
    # save_strategy="no",
    eval_strategy="epoch",
    save_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=32,
    report_to="none",
    logging_strategy="no",
    load_best_model_at_end=True,
    save_total_limit=1,
    learning_rate=2e-4,
    # weight_decay=0.01,
    max_grad_norm=0.5,
    num_train_epochs=7,
    warmup_steps=64,
    # gradient_accumulation_steps=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    bf16=torch_dtype == torch.bfloat16,
    bf16_full_eval=torch_dtype == torch.bfloat16,
    dataloader_num_workers=8,  # Adjust based on your CPU cores
    dataloader_pin_memory=True,
    # optim="sgd",

    # optim="adamw_torch",  # Use efficient optimizer
    # torch_compile=True,  # Enable torch.compile for PyTorch 2.0+ (significant speedup)
    # gradient_checkpointing=False,  # Disable since we want max speed & have enough memory
)

dataset_train = ResamplingDataset(
    ner_dataset.dataset["train"],
    target_fraction=0.7,
    # target_fraction=None,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=ner_dataset.dataset["dev"],
    data_collator=ner_dataset.data_collator,
    compute_metrics=ner_dataset.compute_metrics,
    callbacks=[RoundMetricsCallback(decimal_places=3), dataset_train.callback()],
)

trainer.train()

print("TEST")
test_results = trainer.predict(ner_dataset.dataset["dev"])
print("dev", test_results.metrics)

test_results = trainer.predict(ner_dataset.dataset["test"])
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

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:02<00:00, 24.51it/s]              precision    recall  f1-score   support

        DATE       0.72      0.60      0.65       178
       EVENT       0.17      0.06      0.09        17
         GPE       0.87      0.94      0.90       461
         LOC       0.74      0.85      0.79        61
       MONEY       0.81      0.60      0.69       141
         ORG       0.76      0.83      0.80       515
         PER       0.91      0.93      0.92       687
     PERCENT       0.94      0.57      0.71        53
        PROD       0.56      0.65      0.60        62
        TIME       0.45      0.61      0.52        28
       TITLE       0.78      0.83      0.80       189

   micro avg       0.82      0.83      0.82      2392
   macro avg       0.70      0.68      0.68      2392
weighted avg       0.82      0.83      0.82      2392

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:02<00:00, 23.35it/s]
test {'test_loss': 0.10385819524526596, 'test_precision': 0.8177319587628866, 'test_recall': 0.8290133779264214, 'test_f1': 0.8233340253269669, 'test_accuracy': 0.8177319587628866, 'test_runtime': 2.8387, 'test_samples_per_second': 671.776, 'test_steps_per_second': 21.136}

"""
