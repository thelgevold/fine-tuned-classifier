from collections import Counter

from sklearn.model_selection import train_test_split


class DataSplitHandler:
    @staticmethod
    def split_records(records, seed):
        if not records:
            return [], [], []

        missing_group_id = [record["question"] for record in records if not record.get("group_id")]
        if missing_group_id:
            raise ValueError(
                "Every training record must include a non-empty group_id so paraphrase families "
                "stay together during splitting."
            )

        groups = {}
        for record in records:
            group_id = record["group_id"]
            category = record["category"]
            if group_id in groups and groups[group_id]["category"] != category:
                raise ValueError(
                    f"group_id '{group_id}' spans multiple categories: "
                    f"{groups[group_id]['category']} and {category}"
                )
            groups.setdefault(group_id, {"category": category, "records": []})
            groups[group_id]["records"].append(record)

        grouped_items = [
            {"group_id": group_id, "category": payload["category"]}
            for group_id, payload in groups.items()
        ]

        label_counts = Counter(item["category"] for item in grouped_items)
        smallest_class_size = min(label_counts.values())
        if smallest_class_size < 3:
            raise ValueError(
                "Each category needs at least 3 paraphrase groups for grouped train/validation/test "
                f"stratified splits. Smallest category size: {smallest_class_size}"
            )

        labels = [item["category"] for item in grouped_items]
        train_groups, temp_groups = train_test_split(
            grouped_items,
            test_size=0.30,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )

        temp_labels = [item["category"] for item in temp_groups]
        validation_groups, test_groups = train_test_split(
            temp_groups,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=temp_labels,
        )

        def expand(group_items):
            expanded = []
            for item in group_items:
                expanded.extend(groups[item["group_id"]]["records"])
            return expanded

        train_records = expand(train_groups)
        validation_records = expand(validation_groups)
        test_records = expand(test_groups)

        return train_records, validation_records, test_records
