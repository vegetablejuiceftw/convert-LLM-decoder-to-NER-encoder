import torch
from transformers import Trainer, TrainingArguments, BloomForTokenClassification as OriginalBloomForTokenClassification
from torch import nn
from modeling_bloom import BloomBlock, BloomAttention, BloomConfig, BloomModel, BloomForTokenClassification

from ner_dataset import load_ner_dataset
import torch.nn.functional as F

model_name = "bigscience/bloomz-560m"
# model_name = "bigscience/bloomz-1b1"


class OPTForTokenClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.config = BloomConfig.from_pretrained(model_name, is_decoder=False)
        self.config.is_decoder = False
        print(self.config._attn_implementation)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.llm: BloomModel = BloomModel.from_pretrained( # noqa
            model_name,
            config=self.config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            # attn_implementation="sdpa",
        )

        # Disable causal masking in all attention layers
        for layer in self.llm.h:
            self_attn = layer.self_attention
            layer: BloomBlock
            self_attn: BloomAttention
            self_attn.is_causal = False
            self_attn.is_decoder = False
            self_attn.causal_mask = None


    def forward(self, input_ids, attention_mask=None, labels=None):
        # We won't modify the attention mask here
        outputs = self.llm(
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


data_collator, tokenized_datasets, tokenizer, label_list, compute_metrics = load_ner_dataset(model_name, max_length=40)
num_labels = len(label_list)

# model = OPTForTokenClassification(num_labels)
# model = BloomForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
model = OriginalBloomForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=32,

    report_to="none",
    logging_strategy='no',
    save_strategy="no",

    learning_rate=9e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    max_grad_norm=0.5,

    warmup_steps=32,

    # gradient_accumulation_steps=2,
    per_device_train_batch_size=256,
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

# OG: {'eval_loss': 0.119, 'eval_precision': 0.783, 'eval_recall': 0.836, 'eval_f1': 0.809, 'eval_runtime': 3.2586, 'eval_samples_per_second': 997.361, 'eval_steps_per_second': 7.979, 'epoch': 5.0}
# LOCAL {'eval_loss': 0.186, 'eval_precision': 0.708, 'eval_recall': 0.800, 'eval_f1': 0.751, 'eval_runtime': 3.3512, 'eval_samples_per_second': 969.816, 'eval_steps_per_second': 7.759, 'epoch': 5.0}

# Train the model
# print(trainer.evaluate())
trainer.train()
# trainer.evaluate()

# # Test the model
# test_results = trainer.predict(tokenized_datasets["test"])
# print("test", test_results.metrics)
# {'eval_loss': 0.367, 'eval_model_preparation_time': 0.0024, 'eval_precision': 0.6662567255956956, 'eval_recall': 0.7606177606177607, 'eval_f1': 0.7103171351307055, 'eval_runtime': 3.1393, 'eval_samples_per_second': 1035.259, 'eval_steps_per_second': 8.282, 'epoch': 16.0}
# test {'test_loss': 0.5411821603775024, 'test_model_preparation_time': 0.0024, 'test_precision': 0.5942724458204335, 'test_recall': 0.7035000916254353, 'test_f1': 0.6442896702190148, 'test_runtime': 3.158, 'test_samples_per_second': 1093.429, 'test_steps_per_second': 8.55}

# {'eval_loss': 0.1333947628736496, 'eval_model_preparation_time': 0.9656, 'eval_precision': 0.7943568464730291, 'eval_recall': 0.8399438399438399, 'eval_f1': 0.8165145440586881, 'eval_runtime': 4.0688, 'eval_samples_per_second': 798.769, 'eval_steps_per_second': 6.39, 'epoch': 16.0}
# test {'test_loss': 0.21687284111976624, 'test_model_preparation_time': 0.9656, 'test_precision': 0.7170608108108109, 'test_recall': 0.7778999450247389, 'test_f1': 0.7462424189153556, 'test_runtime': 4.2322, 'test_samples_per_second': 815.897, 'test_steps_per_second': 6.38}

# local
# {'eval_loss': 0.29025998711586, 'eval_model_preparation_time': 0.9677, 'eval_precision': 0.7009505999688328, 'eval_recall': 0.7893997893997894, 'eval_f1': 0.7425505571605447, 'eval_runtime': 4.1152, 'eval_samples_per_second': 789.757, 'eval_steps_per_second': 6.318, 'epoch': 16.0}
