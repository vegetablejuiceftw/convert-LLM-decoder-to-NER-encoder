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
        return {-1}  # Special marker for examples with no entities
    return entity_types


def load_ner_dataset(dataset_name: str = "conll2003") -> NERDataset:
    dataset = load_dataset(dataset_name)

    # Get the proper named labels
    feature = dataset["train"].features["ner_tags"].feature
    label2id = {feature.int2str(i): i for i in range(feature.num_classes)}
    id2label = {v: k for k, v in label2id.items()}

    return NERDataset(dataset=dataset, label2id=label2id, id2label=id2label)


def report_tag_distribution(ner_dataset: NERDataset, split: str = "train", tag_field: str = "ner_tags"):
    # Count all entity tags, including the special -1 tag for no entities
    all_tags = []
    no_entity_count = 0
    
    for example in ner_dataset.dataset[split]:
        entity_types = get_entity_types(example, tag_field)
        if -1 in entity_types:
            no_entity_count += 1
        else:
            all_tags.extend(entity_types)
    
    tag_distribution = Counter(all_tags)
    
    # Add the no-entity count to the distribution
    if no_entity_count > 0:
        tag_distribution[-1] = no_entity_count

    print(f"Entity tag distribution in {split} dataset:")
    for tag_id, count in tag_distribution.most_common():
        if tag_id == -1:
            print(f"  NO_ENTITY: {count}")
        else:
            print(f"  {ner_dataset.id2label.get(tag_id, tag_id)}: {count}")

    print(f"\nTotal entity tags: {sum(tag_distribution.values()) - tag_distribution.get(-1, 0)}")
    print(f"Examples with no entities: {tag_distribution.get(-1, 0)}")
    print(f"Total examples: {len(ner_dataset.dataset[split])}")

    return tag_distribution


# Load the dataset
ner_data = load_ner_dataset()

# Check the distribution of entity types in the original dataset
tag_distribution_original = report_tag_distribution(ner_data)


def create_balanced_sample(ner_dataset: NERDataset, split: str = "train", target_fraction: float = 0.9,
                           tag_field: str = "ner_tags", no_entity_ratio: float = 0.15) -> List[int]:
    """
    Create a balanced sample by removing examples with abundant entity types while preserving rare ones.
    
    This approach:
    1. Indexes all examples by their entity types
    2. Calculates target counts for each entity type
    3. Iteratively removes examples from over-represented entity types
    4. Preserves examples with rare entity types
    5. Adds a controlled number of examples with no entities
    """
    # First, index examples by individual entity type
    entity_type_to_examples: Dict[int, List[int]] = {}
    example_to_entity_types: Dict[int, Set[int]] = {}
    no_entity_examples: List[int] = []
    
    # Count total occurrences of each entity type
    entity_type_counts: CounterType[int] = Counter()
    
    for i, example in enumerate(ner_dataset.dataset[split]):
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
    total_examples = len(ner_dataset.dataset[split])
    target_count = int(total_examples * target_fraction)
    
    # Calculate target counts for each entity type to achieve balance
    total_entity_types = len(entity_type_to_examples)
    
    # Reserve some portion of the target count for examples with no entities
    no_entity_target = int(target_count * no_entity_ratio)
    entity_target_count = target_count - no_entity_target
    
    # Calculate ideal count per entity type
    ideal_count_per_type = entity_target_count // total_entity_types
    
    print(f"\nBalancing {total_entity_types} entity types with target ~{ideal_count_per_type} examples each")
    print(f"Including ~{no_entity_target} examples with no entities ({no_entity_ratio*100:.1f}% of total)")
    
    # Start with all examples that have entities
    candidate_indices = set(example_to_entity_types.keys())
    
    # Sort entity types by frequency (most common first)
    entity_types_by_frequency = sorted(
        entity_type_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Track how many examples we have for each entity type
    current_entity_counts = entity_type_counts.copy()
    
    # Iteratively remove examples from over-represented entity types
    while len(candidate_indices) > entity_target_count:
        # Find the most over-represented entity type
        most_abundant_type, _ = max(
            [(et, current_entity_counts[et]) for et in entity_type_counts],
            key=lambda x: x[1] - ideal_count_per_type if x[1] > ideal_count_per_type else -float('inf')
        )
        
        if current_entity_counts[most_abundant_type] <= ideal_count_per_type:
            # No more over-represented types, break
            break
        
        # Find examples that have this entity type but don't have any rare entity types
        removable_examples = []
        
        for idx in entity_type_to_examples[most_abundant_type]:
            if idx not in candidate_indices:
                continue
                
            # Check if this example has any rare entity types
            has_rare_type = False
            for et in example_to_entity_types[idx]:
                if current_entity_counts[et] <= ideal_count_per_type:
                    has_rare_type = True
                    break
                    
            if not has_rare_type:
                removable_examples.append(idx)
        
        if not removable_examples:
            # No more examples can be removed without affecting rare types
            break
            
        # Remove one example
        if removable_examples:
            idx_to_remove = random.choice(removable_examples)
            candidate_indices.remove(idx_to_remove)
            
            # Update counts
            for et in example_to_entity_types[idx_to_remove]:
                current_entity_counts[et] -= 1
    
    # If we still have too many examples, randomly remove some
    if len(candidate_indices) > entity_target_count:
        candidate_indices = set(random.sample(list(candidate_indices), entity_target_count))
    
    # Add examples with no entities
    if no_entity_examples and no_entity_target > 0:
        no_entity_sample_count = min(no_entity_target, len(no_entity_examples))
        no_entity_samples = random.sample(no_entity_examples, no_entity_sample_count)
        candidate_indices.update(no_entity_samples)
        print(f"Added {len(no_entity_samples)} examples with no entities")
    
    return list(candidate_indices)[:target_count]


# Create the balanced dataset
sampled_indices = create_balanced_sample(ner_data, split="train", target_fraction=0.9)
balanced_dataset = ner_data.dataset["train"].select(sampled_indices)

# Print sampling statistics
print(f"\nSampled {len(sampled_indices)} examples from {len(ner_data.dataset['train'])} total examples")

# Print entity type counts in the balanced dataset
entity_counts = Counter()
for idx in sampled_indices:
    if idx < len(ner_data.dataset["train"]):
        entity_types = get_entity_types(ner_data.dataset["train"][idx], "ner_tags")
        if -1 not in entity_types:
            entity_counts.update(entity_types)

print("\nEntity type counts in balanced dataset:")
for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {ner_data.id2label.get(entity_type, entity_type)}: {count}")


def compare_distributions(original_dist: CounterType, balanced_dist: CounterType, ner_dataset: NERDataset):
    print("\nComparison of tag distributions (original vs. sampled):")
    for tag_id in set(original_dist.keys()) | set(balanced_dist.keys()):
        orig_count = original_dist.get(tag_id, 0)
        bal_count = balanced_dist.get(tag_id, 0)

        orig_percent = 100 * orig_count / sum(original_dist.values()) if sum(original_dist.values()) > 0 else 0
        bal_percent = 100 * bal_count / sum(balanced_dist.values()) if sum(balanced_dist.values()) > 0 else 0

        change = bal_percent - orig_percent

        # Handle the special -1 case (no entities)
        if tag_id == -1:
            print(f"  NO_ENTITY: {orig_percent:.2f}% → {bal_percent:.2f}% ({change:+.2f}%)")
        else:
            print(f"  {ner_dataset.id2label.get(tag_id, tag_id)}: {orig_percent:.2f}% → {bal_percent:.2f}% ({change:+.2f}%)")


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
