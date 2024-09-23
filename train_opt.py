import torch
from transformers import Trainer, TrainingArguments
from torch import nn

from modeling_opt import OPTDecoderLayer, OPTAttention, OPTModel, OPTConfig

from ner_dataset import load_ner_dataset
import torch.nn.functional as F

# model_name = "facebook/opt-350m"
model_name = "facebook/opt-1.3b"


# model_name = "facebook/opt-2.7b"

class OPTForTokenClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.config = OPTConfig.from_pretrained(model_name)
        self.config.is_decoder = False

        # self.opt = OPTModel(self.config)
        self.opt = OPTModel.from_pretrained(
            model_name,
            config=self.config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # self.opt.init_weights()

        # Disable causal masking in all attention layers
        for layer in self.opt.decoder.layers:
            self_attn = layer.self_attn
            layer: OPTDecoderLayer
            self_attn: OPTAttention
            self_attn.is_causal = False
            self_attn.is_decoder = False
            self_attn.causal_mask = None

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

        return {"loss": loss, "logits": logits}


data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics = load_ner_dataset(model_name, max_length=64)
num_labels = len(label_list)

model = OPTForTokenClassification(num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=32,

    report_to="none",
    logging_strategy='no',
    save_strategy="no",

    learning_rate=3e-5,
    num_train_epochs=16,
    weight_decay=0.01,
    max_grad_norm=0.5,

    warmup_steps=32,

    # gradient_accumulation_steps=2,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,

    # tf32=True,
    bf16=True,
    bf16_full_eval=True,

    # fp16=True,
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
# trainer.evaluate()

# # Test the model
test_results = trainer.predict(tokenized_datasets["test"])
print("test", test_results.metrics)

# {'eval_loss': 0.071, 'eval_model_preparation_time': 0.003 'eval_precision': 0.841, 'eval_recall': 0.885, 'eval_f1': 0.863, 'eval_runtime': 6.39, 'eval_samples_per_second': 508.604, 'eval_steps_per_second': 4.069, 'epoch': 2.0}
# {'eval_loss': 0.093, 'eval_model_preparation_time': 0.002, 'eval_precision': 0.891, 'eval_recall': 0.922, 'eval_f1': 0.906, 'eval_runtime': 6.5161, 'eval_samples_per_second': 498.763, 'eval_steps_per_second': 3.99, 'epoch': 16.0
# test {'test_loss': 0.113, 'test_model_preparation_time': 0.003 'test_precision': 0.806, 'test_recall': 0.850, 'test_f1': 0.828, 'test_runtime': 6.665, 'test_samples_per_second': 518.079, 'test_steps_per_second': 4.051}
# test {'test_loss': 0.178, 'test_model_preparation_time': 0.002, 'test_precision': 0.832, 'test_recall': 0.872, 'test_f1': 0.852, 'test_runtime': 6.8165, 'test_samples_per_second': 506.568, 'test_steps_per_second': 3.961}