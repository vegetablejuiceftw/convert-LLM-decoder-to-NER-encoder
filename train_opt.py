import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import Trainer, TrainingArguments, OPTModel
from torch import nn

from datahelper.ner_dataset import prepare_ner_dataset, DATASETS
from datahelper.datasampler import ResamplingDataset
from datahelper.utils import RoundMetricsCallback, get_trainable_linear_modules

from modeling_opt import OPTDecoderLayer, OPTAttention, OPTModel, OPTConfig


class OPTForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels, disable_casual=False,  trainable_layers=8, **kwargs):
        super().__init__()
        if kwargs.get("torch_dtype") is None:
            kwargs.pop("torch_dtype", None)
        self.num_labels = num_labels

        self.config = OPTConfig.from_pretrained(model_name)
        self.config.is_decoder = not disable_casual

        # self.opt = OPTModel(self.config)
        self.opt = OPTModel.from_pretrained(
            model_name,
            config=self.config,
            **kwargs
        )
        # self.opt.init_weights()

        # Disable causal masking in all attention layers
        if disable_casual:
            for param in self.opt.parameters():
                param.requires_grad = False

            print("self.opt.decoder.layers", len(self.opt.decoder.layers))
            for layer in self.opt.decoder.layers[-trainable_layers:]:
                self_attn = layer.self_attn
                layer: OPTDecoderLayer
                self_attn: OPTAttention
                self_attn.is_causal = False
                self_attn.is_decoder = False
                self_attn.causal_mask = None

                for param in layer.parameters():
                    param.requires_grad = True

        for param in self.opt.parameters():
            param.requires_grad = False
        for layer in self.opt.decoder.layers[-trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable_linears = get_trainable_linear_modules(self.opt)
        print(f"Found {len(trainable_linears)} trainable linear layers", trainable_linears)
        # from peft import get_peft_model, LoraConfig, TaskType
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS,
        #     r=32,  # Rank of the update matrices
        #     lora_alpha=32,  # Parameter for scaling
        #     lora_dropout=0.1,  # Dropout probability for LoRA layers
        #     target_modules=trainable_linears,  # Which modules to apply LoRA to
        #     bias="none",
        #     modules_to_save=["classifier"],  # Save the classifier if fine-tuning for classification
        # )
        #
        # self.opt = get_peft_model(self.opt, peft_config)
        # self.opt.print_trainable_parameters()  # Shows % of trainable parameters

        self.classifier = nn.Linear(self.config.word_embed_proj_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # We won't modify the attention mask here
        outputs = self.opt(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # Important: set this to False for token classification
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)

                # Create a boolean mask for active (non-padded) tokens
                active_loss = attention_mask.view(-1).bool()  # [256*40]

                # Only select the logits and labels for active tokens
                active_logits = active_logits[active_loss]  # [num_active_tokens, 23]
                active_labels = active_labels[active_loss]  # [num_active_tokens]

                loss = F.cross_entropy(active_logits, active_labels)
            else:
                print("attention mask is none")

        return {"loss": loss, "logits": logits}

torch_dtype = torch.bfloat16
# torch_dtype = torch.float32

# model_name = "facebook/opt-350m"
# model_name = "facebook/opt-1.3b"
model_name = "facebook/opt-2.7b"

ner_dataset = prepare_ner_dataset(DATASETS.CONLL, model_name)

model = OPTForTokenClassification(
    model_name,
    ner_dataset.num_labels,
    disable_casual=True,
    # torch_dtype = torch_dtype,
    torch_dtype =  torch.float32,
    # device_map="auto",
    attn_implementation="flash_attention_2",
)


training_args = TrainingArguments(
    output_dir="./results/opt",
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
    num_train_epochs=5,
    weight_decay=0.01,
    max_grad_norm=0.5,
    warmup_steps=64,

    # gradient_accumulation_steps=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    bf16=torch_dtype == torch.bfloat16,
    bf16_full_eval=torch_dtype == torch.bfloat16,

    # half_precision_backend="amp",
    # fp16_opt_level="O1",  # Optimization level for FP16
    dataloader_num_workers=8,  # Adjust based on your CPU cores
    dataloader_pin_memory=True,
    # remove_unused_columns=False,

    # torch_compile=True,  # Enable torch.compile for PyTorch 2.0+ (significant speedup)
    gradient_checkpointing=False,  # Disable since we want max speed & have enough memory
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

# print(trainer.evaluate())

trainer.train()

print("TEST")
test_results = trainer.predict(ner_dataset.dataset["dev"])
print("dev", test_results.metrics)

test_results = trainer.predict(ner_dataset.dataset["test"])
print("test", test_results.metrics)

# {'eval_loss': 0.071, 'eval_model_preparation_time': 0.003 'eval_precision': 0.841, 'eval_recall': 0.885, 'eval_f1': 0.863, 'eval_runtime': 6.39, 'eval_samples_per_second': 508.604, 'eval_steps_per_second': 4.069, 'epoch': 2.0}
# {'eval_loss': 0.093, 'eval_model_preparation_time': 0.002, 'eval_precision': 0.891, 'eval_recall': 0.922, 'eval_f1': 0.906, 'eval_runtime': 6.5161, 'eval_samples_per_second': 498.763, 'eval_steps_per_second': 3.99, 'epoch': 16.0
# test {'test_loss': 0.113, 'test_model_preparation_time': 0.003 'test_precision': 0.806, 'test_recall': 0.850, 'test_f1': 0.828, 'test_runtime': 6.665, 'test_samples_per_second': 518.079, 'test_steps_per_second': 4.051}
# test {'test_loss': 0.178, 'test_model_preparation_time': 0.002, 'test_precision': 0.832, 'test_recall': 0.872, 'test_f1': 0.852, 'test_runtime': 6.8165, 'test_samples_per_second': 506.568, 'test_steps_per_second': 3.961}
