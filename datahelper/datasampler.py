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
    # First, index examples by individual entity type
    entity_type_to_examples: Dict[int, List[int]] = {}
    example_to_entity_types: Dict[int, Set[int]] = {}
    
    # Count total occurrences of each entity type
    entity_type_counts: CounterType[int] = Counter()
    
    for i, example in enumerate(ner_dataset.dataset[split]):
        entity_types = get_entity_types(example, tag_field)
        
        # Skip examples with no entities
        if -1 in entity_types:
            continue
            
        example_to_entity_types[i] = entity_types
        
        # Add this example to each entity type's list
        for entity_type in entity_types:
            if entity_type not in entity_type_to_examples:
                entity_type_to_examples[entity_type] = []
            entity_type_to_examples[entity_type].append(i)
            entity_type_counts[entity_type] += 1
    
    # Calculate target counts for each entity type to achieve balance
    total_entity_types = len(entity_type_to_examples)
    target_per_entity = max(50, target_count // total_entity_types)
    
    print(f"\nBalancing {total_entity_types} entity types with target ~{target_per_entity} examples each")
    
    # First pass: ensure representation of rare entity types
    sampled_indices = set()
    examples_by_rarity = sorted(entity_type_to_examples.items(), 
                               key=lambda x: len(x[1]))
    
    # Start with rarest entity types
    for entity_type, indices in examples_by_rarity:
        # Calculate how many more examples we need for this entity type
        current_count = sum(1 for idx in sampled_indices 
                           if entity_type in example_to_entity_types.get(idx, set()))
        needed = min(target_per_entity - current_count, len(indices))
        
        if needed <= 0:
            continue
            
        # Get candidate indices not yet sampled
        candidates = [idx for idx in indices if idx not in sampled_indices]
        if not candidates:
            continue
            
        # Sample from candidates
        new_samples = random.sample(candidates, min(needed, len(candidates)))
        sampled_indices.update(new_samples)
    
    # Second pass: fill up to target count with stratified sampling
    if len(sampled_indices) < target_count:
        remaining = target_count - len(sampled_indices)
        
        # Get all indices not yet sampled that have entities
        remaining_indices = [i for i in example_to_entity_types.keys() 
                            if i not in sampled_indices]
        
        # Weight examples by inverse frequency of their entity types
        weights = []
        for idx in remaining_indices:
            # Calculate weight based on rarity of entity types in this example
            entity_weights = [1.0 / max(1, entity_type_counts[et]) 
                             for et in example_to_entity_types[idx]]
            weights.append(sum(entity_weights))
        
        # Normalize weights
        if weights and sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
            
            # Sample remaining examples with weights
            additional_samples = random.choices(
                remaining_indices, 
                weights=weights, 
                k=min(remaining, len(remaining_indices))
            )
            sampled_indices.update(additional_samples)
    
    return list(sampled_indices)[:target_count]


# Create the balanced dataset
sampled_indices = create_balanced_sample(ner_data, split="train")
balanced_dataset = ner_data.dataset["train"].select(sampled_indices)

# Print sampling statistics
print(f"\nSampled {len(sampled_indices)} examples from {len(ner_data.dataset['train'])} total examples")


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
