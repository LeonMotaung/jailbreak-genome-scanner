"""Run prompts through Model Under Test - Wrapper for existing defender system."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.defenders.llm_defender import LLMDefender
from src.models.jailbreak import AttackStrategy


def create_model(
    model_type: str = "mock",
    model_name: str = "mock-model",
    **kwargs
) -> Any:
    """
    Create model adapter using existing defender system.
    
    Args:
        model_type: mock, openai, anthropic, or lambda
        model_name: Model identifier
        **kwargs: Additional config (api_key, instance_id, etc.)
        
    Returns:
        Model adapter instance
    """
    if model_type == "mock":
        # Create mock defender
        defender = LLMDefender(
            model_name="mock-model",
            model_type="mock"
        )
        return MockAdapter(defender)
    
    elif model_type == "openai":
        api_key = kwargs.get("api_key") or kwargs.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        defender = LLMDefender(
            model_name=model_name or "gpt-3.5-turbo",
            model_type="openai",
            api_key=api_key
        )
        return OpenAIAdapter(defender)
    
    elif model_type == "lambda":
        instance_id = kwargs.get("instance_id")
        api_endpoint = kwargs.get("api_endpoint")
        defender = LLMDefender(
            model_name=model_name,
            model_type="local",
            use_lambda=True,
            lambda_instance_id=instance_id,
            lambda_api_endpoint=api_endpoint
        )
        return LambdaAdapter(defender)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class MockAdapter:
    """Mock model adapter wrapper."""
    
    def __init__(self, defender: LLMDefender):
        self.defender = defender
        self.name = "mock-model"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response (synchronous wrapper)."""
        # Use defender's async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.defender.generate_response(prompt, **kwargs)
            )
            return response
        finally:
            loop.close()


class OpenAIAdapter:
    """OpenAI adapter wrapper."""
    
    def __init__(self, defender: LLMDefender):
        self.defender = defender
        self.name = defender.model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.defender.generate_response(prompt, **kwargs)
            )
            return response
        finally:
            loop.close()


class LambdaAdapter:
    """Lambda Cloud adapter wrapper."""
    
    def __init__(self, defender: LLMDefender):
        self.defender = defender
        self.name = defender.model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Lambda instance."""
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

