from datasets import load_dataset
import random
from collections import Counter

# Load the dataset
NER_DS = load_dataset("conll2003")

# Get the proper named labels
feature = NER_DS["train"].features["ner_tags"].feature
label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
id2label = {v: k for k, v in label2id.items()}

# Check the distribution of entity types in the original dataset
all_tags_original = [tag for example in NER_DS["train"] for tag in example["ner_tags"] if tag != 0]
tag_distribution_original = Counter(all_tags_original)

print("Entity tag distribution in original dataset:")
for tag_id, count in tag_distribution_original.most_common():
    print(f"  {id2label[tag_id]}: {count}")

print(f"\nTotal entity tags: {sum(tag_distribution_original.values())}")
print(f"Total examples: {len(NER_DS['train'])}")


# Function to get the entity types in a sample
def get_entity_types(example):
    entity_types = set()
    for tag in example["ner_tags"]:
        if tag != 0:  # Not 'O'
            entity_types.add(tag)
    return entity_types


# Group examples by their entity types
entity_to_examples = {}
for i, example in enumerate(NER_DS["train"]):
    entity_types = get_entity_types(example)

    # Skip examples with no entities
    if not entity_types:
        continue

    # Create a key based on the set of entity types
    key = tuple(sorted(entity_types))

    if key not in entity_to_examples:
        entity_to_examples[key] = []

    entity_to_examples[key].append(i)

# Now sample from each group to create a balanced dataset
sampled_indices = []
target_count = 1000

# Calculate how many examples to sample from each group
total_groups = len(entity_to_examples)
samples_per_group = target_count // total_groups

print(f"\nSampling approximately {samples_per_group} examples from each of {total_groups} entity type combinations")

# Ensure we get at least one sample from each group, if possible
for key, indices in entity_to_examples.items():
    # Take the minimum between available examples and desired count
    count = min(len(indices), samples_per_group)
    sampled_indices.extend(random.sample(indices, count))

# If we haven't reached our target, sample more randomly
if len(sampled_indices) < target_count:
    remaining = target_count - len(sampled_indices)

    # Get all indices not yet sampled
    all_indices = set(range(len(NER_DS["train"])))
    remaining_indices = list(all_indices - set(sampled_indices))

    # Sample randomly from remaining examples
    additional_samples = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
    sampled_indices.extend(additional_samples)

# Create the balanced dataset
balanced_dataset = NER_DS["train"].select(sampled_indices[:1000])

# Check the distribution of entity types in our balanced sample
all_tags_balanced = [tag for example in balanced_dataset for tag in example["ner_tags"] if tag != 0]
tag_distribution_balanced = Counter(all_tags_balanced)

print("\nEntity tag distribution in sampled dataset:")
for tag_id, count in tag_distribution_balanced.most_common():
    print(f"  {id2label[tag_id]}: {count}")

print(f"\nTotal entity tags in sampled dataset: {sum(tag_distribution_balanced.values())}")
print(f"Total examples in sampled dataset: {len(balanced_dataset)}")

# Comparison of distributions (percentage change)
print("\nComparison of tag distributions (original vs. sampled):")
for tag_id in set(tag_distribution_original.keys()) | set(tag_distribution_balanced.keys()):
    orig_count = tag_distribution_original.get(tag_id, 0)
    bal_count = tag_distribution_balanced.get(tag_id, 0)

    orig_percent = 100 * orig_count / sum(tag_distribution_original.values())
    bal_percent = 100 * bal_count / sum(tag_distribution_balanced.values())

    change = bal_percent - orig_percent

    print(f"  {id2label[tag_id]}: {orig_percent:.2f}% → {bal_percent:.2f}% ({change:+.2f}%)")