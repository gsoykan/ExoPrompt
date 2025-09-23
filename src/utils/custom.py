import json
import math
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split


def split_list_randomly(input_list, split_ratio=0.8):
    """
    Splits a list into two parts randomly.

    Args:
        input_list (list): The list to split.
        split_ratio (float): The ratio for the first part (default is 0.8).

    Returns:
        tuple: Two lists, the first with `split_ratio` of elements and the second with the rest.
    """
    # Shuffle the list
    shuffled_list = input_list[:]
    random.shuffle(shuffled_list)

    # Calculate the split index
    split_index = int(len(shuffled_list) * split_ratio)

    # Split the list
    first_part = shuffled_list[:split_index]
    second_part = shuffled_list[split_index:]

    return first_part, second_part


def split_dataset_sequentially(
    dataset: Dataset, train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> Tuple[Dataset, Dataset, Dataset]:
    train_ratio = train_val_test_ratio[0]
    val_ratio = train_val_test_ratio[1]
    test_ratio = train_val_test_ratio[2]

    assert math.isclose(
        train_ratio + val_ratio + test_ratio, 1, rel_tol=1e-9, abs_tol=0.0
    ), f"sum of ratios should be equal to 1! value: {train_ratio + val_ratio + test_ratio}"

    dataset_size = len(dataset)
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, dataset_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def split_dataset_sequentially_into_two(
    dataset: Dataset, split_ratio: float = 0.5
) -> Tuple[Dataset, Dataset]:
    """
    Splits a dataset sequentially into two subsets based on a given ratio.

    Args:
        dataset (Dataset): The full dataset to split.
        split_ratio (float): The proportion of the dataset assigned to the first subset (0.0 - 1.0). Default is 0.5.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the first subset (up to `split_ratio` of the dataset)
                                 and the second subset (remaining data).

    Raises:
        AssertionError: If `split_ratio` is not in the range [0, 1].
    """
    assert (
        0 <= split_ratio <= 1
    ), f"split_ratio must be in range [0,1], got {split_ratio}"

    dataset_size = len(dataset)
    first_end = math.floor(split_ratio * dataset_size)  # Use floor for safety

    if dataset_size == 0:
        return Subset(dataset, []), Subset(
            dataset, []
        )  # Return empty subsets for empty dataset

    if first_end == 0:
        return Subset(dataset, []), Subset(
            dataset, list(range(dataset_size))
        )  # All data in second subset

    if first_end == dataset_size:
        return Subset(dataset, list(range(dataset_size))), Subset(
            dataset, []
        )  # All data in first subset

    first_indices = list(range(0, first_end))
    second_indices = list(range(first_end, dataset_size))

    first_dataset = Subset(dataset, first_indices)
    second_dataset = Subset(dataset, second_indices)

    return first_dataset, second_dataset


def get_sequential_slice_from_dataset(
    dataset: Dataset,
    start_ratio: float,
    end_ratio: float,
) -> Subset:
    """
    Extracts a sequential subset of a dataset based on the specified ratio range.

    Example usage:
        - To extract a subset between 30% and 50% of the dataset:
          `get_sequential_slice_from_dataset(dataset, 0.3, 0.5)`

    Args:
        dataset (Dataset): The full dataset.
        start_ratio (float): The starting percentage (0.0 - 1.0).
        end_ratio (float): The ending percentage (0.0 - 1.0).

    Returns:
        Subset: A subset of the original dataset between start_ratio and end_ratio.

    Raises:
        AssertionError: If input ratios are invalid.

    Reference:
        Cerqueira, V., Torgo, L. & Mozetič, I. (2020). Evaluating time series forecasting models:
        an empirical study on performance estimation methods.
        Machine Learning, 109, 1997–2028. https://doi.org/10.1007/s10994-020-05910-7
    """
    assert (
        0 <= start_ratio <= 1
    ), f"start_ratio should be in range [0,1], got {start_ratio}"
    assert 0 <= end_ratio <= 1, f"end_ratio should be in range [0,1], got {end_ratio}"
    assert (
        start_ratio < end_ratio
    ), f"start_ratio ({start_ratio}) must be less than end_ratio ({end_ratio})"

    dataset_size = len(dataset)
    start_index = math.floor(start_ratio * dataset_size)
    end_index = min(math.floor(end_ratio * dataset_size), dataset_size)

    if start_index >= dataset_size or start_index == end_index:
        # Return an empty subset if invalid
        return Subset(dataset, [])

    slice_indices = list(range(start_index, end_index))
    return Subset(dataset, slice_indices)


def split_dataset_randomly(
    dataset: Dataset,
    train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    # if you'd checked the docs you would not have to write this :)
    #  https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    #  "If a list of fractions that sum up to 1 is given, the lengths will be computed automatically
    #  as floor(frac * len(dataset)) for each fraction provided."
    train_ratio = train_val_test_ratio[0]
    val_ratio = train_val_test_ratio[1]
    test_ratio = train_val_test_ratio[2]

    assert math.isclose(
        train_ratio + val_ratio + test_ratio, 1, rel_tol=1e-9, abs_tol=0.0
    ), f"sum of ratios should be equal to 1! value: {train_ratio + val_ratio + test_ratio}"

    # Calculate the number of samples for each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure total size matches

    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=(train_size, val_size, test_size),
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, val_dataset, test_dataset


def read_json_file(file_path):
    """
    Reads a JSON file and returns the data.

    :param file_path: str, path to the JSON file
    :return: dict, content of the JSON file
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print("The file is not a valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")
