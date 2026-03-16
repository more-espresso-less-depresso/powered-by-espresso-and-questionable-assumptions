import json
import torch

from pathlib import Path

def save_dict_to_json(data: dict, filepath: str, indent: int = 4):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"Saved JSON to: {filepath}")
    
def load_json_to_dict(filepath: str) -> dict:
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_list_to_jsonl(data: list, filepath: str, indent: int = 4):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, indent=indent, ensure_ascii=False)
            f.write("\n")

    print(f"Saved JSON to: {filepath}")

def save_stack_to_pt(activations: dict, filepath: str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    
    torch.save(activations, filepath)
    
    print(f"Saved activation tensors to: {filepath}")

def load_pt_to_stack(filepath: str) -> torch.stack:
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    
    tensors = torch.load(filepath)
    print(f"Loaded tensors of shape: {tensors.shape}")
    return tensors