import torch
from torch.utils.data import Dataset

from datahelper.datasampler import report_tag_distribution, create_balanced_sample

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, TrainerCallback
from datahelper.ner_dataset import prepare_ner_dataset, DATASETS, NERDataset
from datahelper.utils import RoundMetricsCallback

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
)
print(model)

# 2. Configure LoRA
from peft import get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                       # Rank of the update matrices
    lora_alpha=32,              # Parameter for scaling
    lora_dropout=0.1,           # Dropout probability for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Which modules to apply LoRA to
    bias="none",
    modules_to_save=["classifier"],  # Save the classifier if fine-tuning for classification
)
# 3. Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Shows % of trainable parameters


print(config.name_or_path, config.attention_type if hasattr(config, "attention_type") else "Standard attention")



class ResamplingDataset(Dataset):
    def __init__(self, dataset, target_fraction: float | None =0.8):
        self.dataset = dataset
        self.target_fraction = target_fraction
        self.current_data = None
        self.resample()

    def resample(self):
        if self.target_fraction is None:
            self.current_data = self.dataset
            return
        self.current_data = create_balanced_sample(self.dataset, target_fraction=self.target_fraction)
        print("Data resampled for next epoch")

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        return self.current_data[idx]

    def callback(self):
        # Define a custom callback to resample data after each epoch
        class ResamplingCallback(TrainerCallback):
            def __init__(self, dataset):
                self.dataset = dataset

            def on_epoch_begin(self, args, state, control, **kwargs):
                self.dataset.resample()

        return ResamplingCallback(self)


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
    num_train_epochs=4,
    warmup_steps=32,
    # gradient_accumulation_steps=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    bf16=torch_dtype == torch.bfloat16,
    bf16_full_eval=torch_dtype == torch.bfloat16,
    dataloader_num_workers=8,  # Adjust based on your CPU cores
    dataloader_pin_memory=True,
    # optim="sgd",

    # optim="adamw_torch",  # Use efficient optimizer
    torch_compile=True,  # Enable torch.compile for PyTorch 2.0+ (significant speedup)
    gradient_checkpointing=False,  # Disable since we want max speed & have enough memory
)

dataset_train = ResamplingDataset(
    ner_dataset.dataset["train"],
    # target_fraction=0.8,
    target_fraction=None,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=ner_dataset.dataset["dev"],
    tokenizer=ner_dataset.tokenizer,
    data_collator=ner_dataset.data_collator,
    compute_metrics=ner_dataset.compute_metrics,
    callbacks=[RoundMetricsCallback(decimal_places=2), dataset_train.callback()],
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
"""
