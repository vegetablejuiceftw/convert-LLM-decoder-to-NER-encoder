from transformers import TrainerCallback


class RoundMetricsCallback(TrainerCallback):
    def __init__(self, decimal_places=4):
        self.decimal_places = decimal_places

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, self.decimal_places)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            value[sub_key] = round(sub_value, self.decimal_places)
