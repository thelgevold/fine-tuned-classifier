import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path("/workspace/outputs/house-qwen3-0.6b")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into a base model.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    lora_dir = args.output_dir / "lora"
    merged_dir = args.output_dir / "merged"

    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, str(lora_dir))
    model = model.merge_and_unload()

    model.save_pretrained(
        str(merged_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer.save_pretrained(str(merged_dir))

    print("Merged model saved to:", merged_dir)


if __name__ == "__main__":
    main()
