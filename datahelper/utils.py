from torch import nn
from transformers import TrainerCallback


class RoundMetricsCallback(TrainerCallback):
    def __init__(self, decimal_places=4):
        self.decimal_places = decimal_places

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = float(f'%.{self.decimal_places}g' % value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            value[sub_key] =  float(f'%.{self.decimal_places}g' % sub_value)


def get_trainable_linear_modules(model, parent_name=""):
    trainable_modules = []

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # Check if it's a Linear layer with trainable parameters
        if isinstance(module, nn.Linear) and any(p.requires_grad for p in module.parameters()):
            trainable_modules.append(full_name)

        # Recursively check child modules
        child_modules = get_trainable_linear_modules(module, full_name)
        trainable_modules.extend(child_modules)

    return trainable_modules
