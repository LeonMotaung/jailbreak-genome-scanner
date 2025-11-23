"""Ingest prompts - Simple JSONL/CSV loader."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_prompts(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load prompts from JSONL or CSV file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        List of prompt dictionaries with prompt_id, attack_family, text
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".jsonl":
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prompt = json.loads(line)
                    # Normalize to expected format
                    if "prompt_id" not in prompt and "key" in prompt:
                        prompt["prompt_id"] = prompt["key"]
                    if "text" not in prompt and "input" in prompt:
                        prompt["text"] = prompt["input"]
                    if "attack_family" not in prompt:
                        prompt["attack_family"] = prompt.get("strategy", "unknown")
                    prompts.append(prompt)
                except json.JSONDecodeError:
                    continue
        return prompts
    
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
        
        prompts = []
        for _, row in df.iterrows():
            prompt = {
                "prompt_id": str(row.get("key", row.get("prompt_id", f"prompt_{len(prompts)}"))),
                "attack_family": str(row.get("attack_family", "unknown")),
                "text": str(row.get("input", row.get("text", "")))
            }
            if "numeric_eval" in row:
                prompt["numeric_eval"] = float(row["numeric_eval"])
            prompts.append(prompt)
        
        return prompts
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .jsonl or .csv")


def save_prompts(prompts: List[Dict[str, Any]], output_path: Path):
    """Save prompts to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

