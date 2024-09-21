from datasets import load_dataset, Sequence, ClassLabel

# Load the dataset
NER_DS = load_dataset("tartuNLP/EstNER", "estner-reannotated", columns=["tokens", "ner_tags"])

# Get unique NER tags
unique_tags = set()
for example in NER_DS["train"]:
    unique_tags.update(example["ner_tags"])

# Create tag-to-id and id-to-tag mappings
tag2id = {tag: i for i, tag in enumerate(sorted(unique_tags))}
id2tag = {i: tag for tag, i in tag2id.items()}


# Define conversion function
def convert_tags_to_ids(example):
    return {"ner_tags": [tag2id[tag] for tag in example["ner_tags"]]}


# Apply conversion
NER_DS = NER_DS.map(convert_tags_to_ids)

# Create a Sequence of ClassLabel feature
ner_feature = Sequence(ClassLabel(num_classes=len(tag2id), names=list(tag2id.keys())))

# Update the dataset's features
NER_DS = NER_DS.cast_column("ner_tags", ner_feature)

# Verify the conversion
print(NER_DS["train"].features["ner_tags"])
print(NER_DS["train"][0]["ner_tags"])
print(NER_DS)

NER_DS.save_to_disk("EstNER")
