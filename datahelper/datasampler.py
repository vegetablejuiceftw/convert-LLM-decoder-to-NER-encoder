from datasets import load_dataset
import random
from collections import Counter
from typing import Dict, List, Set, Tuple, Counter as CounterType

def load_ner_dataset(dataset_name: str = "conll2003"):
    NER_DS = load_dataset(dataset_name)
    
    # Get the proper named labels
    feature = NER_DS["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    id2label = {v: k for k, v in label2id.items()}
    
    return NER_DS, label2id, id2label

def report_tag_distribution(dataset, id2label: Dict[int, str], split: str = "train", tag_field: str = "ner_tags"):
    all_tags = [tag for example in dataset[split] for tag in example[tag_field] if tag != 0]
    tag_distribution = Counter(all_tags)
    
    print(f"Entity tag distribution in {split} dataset:")
    for tag_id, count in tag_distribution.most_common():
        print(f"  {id2label[tag_id]}: {count}")
    
    print(f"\nTotal entity tags: {sum(tag_distribution.values())}")
    print(f"Total examples: {len(dataset[split])}")
    
    return tag_distribution

# Load the dataset
NER_DS, label2id, id2label = load_ner_dataset()

# Check the distribution of entity types in the original dataset
tag_distribution_original = report_tag_distribution(NER_DS, id2label)


def get_entity_types(example: Dict, tag_field: str = "ner_tags") -> Set[int]:
    entity_types = set()
    for tag in example[tag_field]:
        if tag != 0:  # Not 'O'
            entity_types.add(tag)
    return entity_types

def create_balanced_sample(dataset, split: str = "train", target_count: int = 1000, tag_field: str = "ner_tags") -> List[int]:
    # Group examples by their entity types
    entity_to_examples = {}
    for i, example in enumerate(dataset[split]):
        entity_types = get_entity_types(example, tag_field)

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
        all_indices = set(range(len(dataset[split])))
        remaining_indices = list(all_indices - set(sampled_indices))

        # Sample randomly from remaining examples
        additional_samples = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
        sampled_indices.extend(additional_samples)

    return sampled_indices[:target_count]

# Create the balanced dataset
sampled_indices = create_balanced_sample(NER_DS)
balanced_dataset = NER_DS["train"].select(sampled_indices)

def compare_distributions(original_dist: CounterType, balanced_dist: CounterType, id2label: Dict[int, str]):
    print("\nComparison of tag distributions (original vs. sampled):")
    for tag_id in set(original_dist.keys()) | set(balanced_dist.keys()):
        orig_count = original_dist.get(tag_id, 0)
        bal_count = balanced_dist.get(tag_id, 0)

        orig_percent = 100 * orig_count / sum(original_dist.values()) if sum(original_dist.values()) > 0 else 0
        bal_percent = 100 * bal_count / sum(balanced_dist.values()) if sum(balanced_dist.values()) > 0 else 0

        change = bal_percent - orig_percent

        print(f"  {id2label[tag_id]}: {orig_percent:.2f}% → {bal_percent:.2f}% ({change:+.2f}%)")

# Check the distribution of entity types in our balanced sample
tag_distribution_balanced = report_tag_distribution({"train": balanced_dataset}, id2label, split="train")

# Compare the distributions
compare_distributions(tag_distribution_original, tag_distribution_balanced, id2label)
