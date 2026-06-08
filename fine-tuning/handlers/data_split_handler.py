from collections import Counter

from sklearn.model_selection import train_test_split


class DataSplitHandler:
    @staticmethod
    def split_records(records, seed):
        label_counts = Counter(record["category"] for record in records)
        smallest_class_size = min(label_counts.values())
        if smallest_class_size < 3:
            raise ValueError(
                "Each category needs at least 3 examples for sklearn-based train/validation/test stratified splits. "
                f"Smallest category size: {smallest_class_size}"
            )

        labels = [record["category"] for record in records]
        train_records, temp_records = train_test_split(
            records,
            test_size=0.30,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )

        temp_labels = [record["category"] for record in temp_records]
        validation_records, test_records = train_test_split(
            temp_records,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=temp_labels,
        )

        return train_records, validation_records, test_records
