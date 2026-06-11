import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from domain.categories import CATEGORY_OUTPUT_CODES
from handlers.data_split_handler import DataSplitHandler
from handlers.prompt_handler import PromptHandler


BASE_MODEL = "Qwen/Qwen3-0.6B"
DATA_PATH = Path("data/category_train.json")
MAX_SEQ_LENGTH = 512
OUTPUT_CODE_TO_CATEGORY = {code: category for category, code in CATEGORY_OUTPUT_CODES.items()}
VALID_CODES = set(OUTPUT_CODE_TO_CATEGORY)
VALID_CATEGORIES = set(CATEGORY_OUTPUT_CODES)


def format_example(example, eos_token, label_mode):
    prompt = PromptHandler.create_categorize_query_prompt(
        example["question"], CATEGORY_OUTPUT_CODES, label_mode=label_mode
    )
    target = get_target_label(example["category"], label_mode)
    return {"text": prompt + target + eos_token}


def get_target_label(category, label_mode):
    if label_mode == "category":
        return category
    return CATEGORY_OUTPUT_CODES[category]


def normalize_prediction(text):
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    first_token = cleaned.split()[0].rstrip(":").strip().upper()
    return first_token


def normalize_category_prediction(text):
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    normalized = cleaned.splitlines()[0].strip().rstrip(":").lower()
    return normalized


def parse_raw_prediction(text, label_mode):
    marker = "Category:" if label_mode == "category" else "Code:"
    raw_prediction = text.split(marker)[-1].strip()
    if label_mode == "category":
        return normalize_category_prediction(raw_prediction)
    return normalize_prediction(raw_prediction)


def predict_category(model, tokenizer, question, label_mode):
    prompt = PromptHandler.create_categorize_query_prompt(
        question, CATEGORY_OUTPUT_CODES, label_mode=label_mode
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=None,
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return parse_raw_prediction(text, label_mode)


def evaluate_split(model, tokenizer, records, label_mode):
    predictions = []
    correct = 0
    invalid = 0
    per_category = Counter()
    per_category_correct = Counter()

    for record in records:
        predicted_label = predict_category(model, tokenizer, record["question"], label_mode)
        expected = record["category"]
        expected_label = get_target_label(expected, label_mode)
        predicted_category = (
            predicted_label if label_mode == "category" else OUTPUT_CODE_TO_CATEGORY.get(predicted_label)
        )
        predicted_code = (
            CATEGORY_OUTPUT_CODES.get(predicted_label) if label_mode == "category" else predicted_label
        )
        is_correct = predicted_label == expected_label

        predictions.append(
            {
                "question": record["question"],
                "expected_category": expected,
                "expected_code": CATEGORY_OUTPUT_CODES[expected],
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "predicted_code": predicted_code,
                "predicted_category": predicted_category,
                "correct": is_correct,
            }
        )

        per_category[expected] += 1
        if is_correct:
            correct += 1
            per_category_correct[expected] += 1
        if is_invalid_prediction(predicted_label, label_mode):
            invalid += 1

    accuracy = correct / len(records) if records else 0.0
    per_category_accuracy = {
        category: (
            per_category_correct[category] / per_category[category]
            if per_category[category]
            else 0.0
        )
        for category in sorted(per_category)
    }

    metrics = {
        "examples": len(records),
        "correct": correct,
        "misclassified_predictions": len(records) - correct,
        "accuracy": accuracy,
        "invalid_predictions": invalid,
        "invalid_code_predictions": invalid,
        "per_category_accuracy": per_category_accuracy,
    }

    return metrics, predictions


def is_invalid_prediction(predicted_label, label_mode):
    if label_mode == "category":
        return predicted_label not in VALID_CATEGORIES
    return predicted_label not in VALID_CODES


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the category model with Unsloth.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/house-qwen3-0.6b"))
    parser.add_argument(
        "--label-mode",
        choices=("code", "category"),
        default="code",
        help="Train the model to emit opaque two-letter codes or full category names.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    lora_dir = output_dir / "lora"
    reports_dir = output_dir / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    full_dataset = load_dataset("json", data_files=str(args.data_path), split="train")
    all_records = [dict(record) for record in full_dataset]
    train_records, validation_records, test_records = DataSplitHandler.split_records(all_records, args.seed)

    write_json(reports_dir / "train_split.json", train_records)
    write_json(reports_dir / "validation_split.json", validation_records)
    write_json(reports_dir / "test_split.json", test_records)

    train_dataset = Dataset.from_list(train_records)
    train_dataset = train_dataset.map(
        format_example,
        fn_kwargs={"eos_token": tokenizer.eos_token or "", "label_mode": args.label_mode},
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = Dataset.from_list(validation_records)
    validation_dataset = validation_dataset.map(
        format_example,
        fn_kwargs={"eos_token": tokenizer.eos_token or "", "label_mode": args.label_mode},
        remove_columns=validation_dataset.column_names,
    )

    sft_config_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "dataset_text_field": "text",
        "max_seq_length": MAX_SEQ_LENGTH,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 5,
        "learning_rate": 2e-4,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_steps": 100,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "report_to": "none",
    }

    sft_config_parameters = inspect.signature(SFTConfig).parameters
    if "evaluation_strategy" in sft_config_parameters:
        sft_config_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sft_config_parameters:
        sft_config_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in sft_config_parameters:
        sft_config_kwargs["save_strategy"] = "epoch"
    if "load_best_model_at_end" in sft_config_parameters:
        sft_config_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sft_config_parameters:
        sft_config_kwargs["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in sft_config_parameters:
        sft_config_kwargs["greater_is_better"] = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=SFTConfig(**sft_config_kwargs),
    )

    trainer.train()

    FastLanguageModel.for_inference(model)

    validation_metrics, validation_predictions = evaluate_split(
        model, tokenizer, validation_records, args.label_mode
    )

    test_metrics, test_predictions = evaluate_split(
        model, tokenizer, test_records, args.label_mode
    )

    report = {
        "base_model": args.base_model,
        "data_path": str(args.data_path),
        "label_mode": args.label_mode,
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_evaluated": True,
    }

    test_predictions_path = reports_dir / "test_predictions.json"

    write_json(reports_dir / "metrics.json", report)
    write_json(reports_dir / "validation_predictions.json", validation_predictions)
    write_json(test_predictions_path, test_predictions)

    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    print("Training complete")
    print(f"LoRA adapter saved to: {lora_dir}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.3f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Reports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
