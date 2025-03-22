


```commandline
aider --read CONVENTIONS.md
```


```python
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    config=config,
    # torch_dtype=torch_dtype,  # causes learning effectiveness to drop?
    ignore_mismatched_sizes=True,
)
```