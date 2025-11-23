"""Run prompts through Model Under Test - Wrapper for existing defender system."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.defenders.llm_defender import LLMDefender


def create_model(
    config: Optional[Any] = None,
    *,
    model_type: str = "mock",
    model_name: str = "mock-model",
    api_endpoint: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create model adapter using existing defender system.
    
    Args:
        config: Optional config object (ignored, kept for compatibility)
        model_type: Retained for compatibility (mock or remote)
        model_name: Model identifier
        api_endpoint: HTTP endpoint serving the model (required for non-mock)
        **kwargs: Additional config (unused)
        
    Returns:
        Model adapter instance
    """
    endpoint = api_endpoint or kwargs.get("api_endpoint")
    
    if model_type == "mock":
        endpoint = endpoint or "mock://prototype-mock"
    elif not endpoint:
        raise ValueError("api_endpoint is required for non-mock models")
    
    defender = LLMDefender(
        model_name=model_name or "mock-model",
        api_endpoint=endpoint
    )
    return RemoteAdapter(defender)


class RemoteAdapter:
    """Remote model adapter wrapper."""
    
    def __init__(self, defender: LLMDefender):
        self.defender = defender
        self.name = defender.model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response (synchronous wrapper)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.defender.generate_response(prompt, **kwargs)
            )
            return response
        finally:
            loop.close()


def run_prompts(
    prompts: List[Dict[str, Any]],
    model: Any,
    output_path: Path
) -> List[Dict[str, Any]]:
    """
    Run prompts through model and save responses.
    
    Args:
        prompts: List of prompt dictionaries
        model: Model adapter instance
        output_path: Path to save responses.jsonl
        
    Returns:
        List of response dictionaries
    """
    responses = []
    
    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data.get("text", prompt_data.get("prompt", ""))
        prompt_id = prompt_data.get("prompt_id", f"prompt_{i}")
        
        try:
            response_text = model.generate(prompt_text)
            
            response_data = {
                "prompt_id": prompt_id,
                "attack_family": prompt_data.get("attack_family", "unknown"),
                "prompt": prompt_text,
                "response": response_text,
                "model": model.name
            }
            
            responses.append(response_data)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(prompts)} prompts...")
        
        except Exception as e:
            print(f"Error processing prompt {prompt_id}: {e}")
            responses.append({
                "prompt_id": prompt_id,
                "attack_family": prompt_data.get("attack_family", "unknown"),
                "prompt": prompt_text,
                "response": f"ERROR: {str(e)}",
                "model": "error"
            })
    
    # Save responses
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(responses)} responses to {output_path}")
    
    return responses

