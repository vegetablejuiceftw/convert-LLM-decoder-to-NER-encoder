from random import sample

from datasets import load_dataset, DatasetDict, Dataset
from collections import Counter
from typing import Dict, List, Set, Counter as CounterType
from dataclasses import dataclass


@dataclass
class NERDataset:
    dataset: DatasetDict
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    ner_labels: List[str]

    @property
    def num_labels(self) -> int:
        return len(self.label2id)
        
    def update(self, **kwargs) -> 'NERDataset':
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"NERDataset has no attribute '{key}'")
        return self


def get_entity_types(example: Dict, tag_field: str = "ner_tags") -> Set[int]:
    entity_types = set()
    for tag in example[tag_field]:
        if tag != 0:  # Not 'O'
            entity_types.add(tag)
    if not entity_types:
        return {-1}  # Special marker for examples with no entities
    return entity_types


def load_ner_dataset(dataset_name: str = "conll2003") -> NERDataset:
    dataset = load_dataset(dataset_name)

    # Get the proper named labels
    feature = dataset["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    id2label = {v: k for k, v in label2id.items()}
    
    # Extract the list of NER labels
    ner_labels = [feature.int2str(i) for i in range(feature.num_classes)]

    return NERDataset(dataset=dataset, label2id=label2id, id2label=id2label, ner_labels=ner_labels)


def report_tag_distribution(ner_dataset: NERDataset, split: str = "train", tag_field: str = "ner_tags"):
    # Count all entity tags, including the special -1 tag for no entities
    all_tags = []
    no_entity = 0

    for example in ner_dataset.dataset[split]:
        entity_types = get_entity_types(example, tag_field)
        if -1 in entity_types:
            no_entity += 1
            continue
        all_tags.extend(entity_types)

    tag_distribution = Counter(all_tags)
    total = sum(tag_distribution.values())

    print(f"Entity tag distribution in {split} dataset:")
    print(f"  NO_ENTITY: {no_entity}")
    for tag_id, count in tag_distribution.most_common():
        print(f"  {ner_dataset.id2label.get(tag_id, tag_id)}: {count / total * 100:.1f}% {count}")

    print(f"\nTotal entity tags: {sum(tag_distribution.values()) - tag_distribution.get(-1, 0)}")
    print(f"Total examples: {len(ner_dataset.dataset[split])}")

    entity_max, entity_min = max(tag_distribution.values()), min(tag_distribution.values())
    print(f"Max/Min/Diff: {entity_max}, {entity_min}, {(entity_max - entity_min) / total * 100:.1f}%")


def create_balanced_sample(dataset: Dataset,
                           tag_field: str = "ner_tags", target_fraction: float = 0.0, id2label: Dict[int, str] | None = None) -> Dataset:
    id2label = id2label or {}
    # First, index examples by individual entity type
    entity_type_to_examples: Dict[int, List[int]] = {}
    example_to_entity_types: Dict[int, Set[int]] = {}
    no_entity_examples: List[int] = []

    # Count total occurrences of each entity type
    entity_type_counts: CounterType[int] = Counter()

    for i, example in enumerate(dataset):
        entity_types = get_entity_types(example, tag_field)

        # Handle examples with no entities separately
        if -1 in entity_types:
            no_entity_examples.append(i)
            continue

        example_to_entity_types[i] = entity_types

        # Add this example to each entity type's list
        for entity_type in entity_types:
            if entity_type not in entity_type_to_examples:
                entity_type_to_examples[entity_type] = []
            entity_type_to_examples[entity_type].append(i)
            entity_type_counts[entity_type] += 1

    # Calculate target count based on dataset size
    candidate_indices = list(set(example_to_entity_types.keys()))
    total_candidates = len(candidate_indices)
    target_count = int(total_candidates * target_fraction)
    # print(target_count, total_candidates)

    # Track how many examples we have for each entity type
    current_entity_counts = entity_type_counts.copy()
    removed = []

    # Calculate target counts for each entity type to achieve balance
    total_entity_types = len(entity_type_to_examples)

    # Calculate ideal count per entity type
    ideal_count_per_type = total_candidates // total_entity_types
    print(entity_type_counts, ideal_count_per_type)
    rare_entities = {et for et, c in entity_type_counts.items() if c < ideal_count_per_type}
    print([id2label.get(et, et) for et in rare_entities])

    def loss_fn(vs):
        avg = sum(vs) / len(vs)
        return sum((avg - v for v in vs if v < avg), start=0)

    # Iteratively remove examples from over-represented entity types
    while len(candidate_indices) > target_count:
        values = current_entity_counts.values()
        average = sum(values) / len(values)
        rare_entities = {et for et, c in entity_type_counts.items() if c < average}
        best_loss = loss_fn(values)

        # print(f"{best_loss * 100:.1f} [{max(values)} - {min(values)}]")
        for idx in sample(candidate_indices, len(candidate_indices) // 10):
            entities = example_to_entity_types[idx]
            if entities & rare_entities:
                continue
            counts = current_entity_counts.copy()
            for et in entities:
                counts[et] -= 1

            loss_local = loss_fn(counts.values())
            if loss_local < best_loss:
                candidate_indices.remove(idx)
                current_entity_counts = counts
                removed.extend(entities)
                break
        else:
            break

    for et, c in Counter(removed).most_common():
        print(id2label.get(et, et), c)

    # Add examples with no entities
    if no_entity_examples:
        print(f"Added {len(no_entity_examples)} examples with no entities")

    sampled_indices = list(candidate_indices) + no_entity_examples

    # Print sampling statistics
    print(f"\nSampled {len(sampled_indices)} examples from {len(dataset)} total examples")

    # Return the sampled indices
    return dataset.select(sampled_indices)


if __name__ == '__main__':
    # Load the dataset
    ner_data = load_ner_dataset()

    # Check the distribution of entity types in the original dataset
    report_tag_distribution(ner_data)

    # Get the sampled indices
    balanced_dataset = create_balanced_sample(
        ner_data.dataset["train"], 
        id2label = ner_data.id2label
    )
    
    # Update the NERDataset with the balanced dataset
    balanced_ner_data = ner_data.update(
        dataset=DatasetDict({**ner_data.dataset, "train": balanced_dataset})
    )

    # Check the distribution of entity types in our balanced sample
    report_tag_distribution(balanced_ner_data, split="train")
