from datasets import load_dataset, Dataset, DatasetDict
import random
from collections import Counter
from typing import Dict, List, Set, Tuple, Counter as CounterType
from dataclasses import dataclass


@dataclass
class NERDataset:
    dataset: DatasetDict
    label2id: Dict[str, int]
    id2label: Dict[int, str]

    @property
    def num_labels(self) -> int:
        return len(self.label2id)


def get_entity_types(example: Dict, tag_field: str = "ner_tags") -> Set[int]:
    entity_types = set()
    for tag in example[tag_field]:
        if tag != 0:  # Not 'O'
            entity_types.add(tag)
    if not entity_types:
        return {-1}
    return entity_types


def load_ner_dataset(dataset_name: str = "conll2003") -> NERDataset:
    dataset = load_dataset(dataset_name)

    # Get the proper named labels
    feature = dataset["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    id2label = {v: k for k, v in label2id.items()}

    return NERDataset(dataset=dataset, label2id=label2id, id2label=id2label)


def report_tag_distribution(ner_dataset: NERDataset, split: str = "train", tag_field: str = "ner_tags"):
    all_tags = [tag for example in ner_dataset.dataset[split] for tag in get_entity_types(example, tag_field) if
                tag != 0]
    tag_distribution = Counter(all_tags)

    print(f"Entity tag distribution in {split} dataset:")
    for tag_id, count in tag_distribution.most_common():
        print(f"  {ner_dataset.id2label.get(tag_id, tag_id)}: {count}")

    print(f"\nTotal entity tags: {sum(tag_distribution.values())}")
    print(f"Total examples: {len(ner_dataset.dataset[split])}")

    return tag_distribution


# Load the dataset
ner_data = load_ner_dataset()

# Check the distribution of entity types in the original dataset
tag_distribution_original = report_tag_distribution(ner_data)


def create_balanced_sample(ner_dataset: NERDataset, split: str = "train", target_count: int = 4000,
                           tag_field: str = "ner_tags") -> List[int]:
    # Group examples by their entity types
    entity_to_examples = {}
    for i, example in enumerate(ner_dataset.dataset[split]):
        entity_types = get_entity_types(example, tag_field)

        # Skip examples with no entities
        if not entity_types:
            print("No entities", i)
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
        all_indices = set(range(len(ner_dataset.dataset[split])))
        remaining_indices = list(all_indices - set(sampled_indices))

        # Sample randomly from remaining examples
        additional_samples = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
        sampled_indices.extend(additional_samples)

    return sampled_indices[:target_count]


# Create the balanced dataset
sampled_indices = create_balanced_sample(ner_data, split="train")
balanced_dataset = ner_data.dataset["train"].select(sampled_indices)


def compare_distributions(original_dist: CounterType, balanced_dist: CounterType, ner_dataset: NERDataset):
    print("\nComparison of tag distributions (original vs. sampled):")
    for tag_id in set(original_dist.keys()) | set(balanced_dist.keys()):
        orig_count = original_dist.get(tag_id, 0)
        bal_count = balanced_dist.get(tag_id, 0)

        orig_percent = 100 * orig_count / sum(original_dist.values()) if sum(original_dist.values()) > 0 else 0
        bal_percent = 100 * bal_count / sum(balanced_dist.values()) if sum(balanced_dist.values()) > 0 else 0

        change = bal_percent - orig_percent

        print(f"  {ner_dataset.id2label[tag_id]}: {orig_percent:.2f}% → {bal_percent:.2f}% ({change:+.2f}%)")


# Create a new NERDataset with the balanced dataset for reporting
balanced_ner_data = NERDataset(
    dataset={"train": balanced_dataset},
    label2id=ner_data.label2id,
    id2label=ner_data.id2label
)

# Check the distribution of entity types in our balanced sample
tag_distribution_balanced = report_tag_distribution(balanced_ner_data, split="train")

# Compare the distributions
compare_distributions(tag_distribution_original, tag_distribution_balanced, ner_data)
