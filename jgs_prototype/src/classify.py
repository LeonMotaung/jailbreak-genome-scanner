"""Safety classification - Wrapper for existing safety classifier system."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.referee.safety_classifier import SafetyClassifier
from src.models.jailbreak import EvaluationResult, AttackStrategy


def classify_responses(
    responses: List[Dict[str, Any]],
    classifier_type: str = "rule_based",
    output_path: Path = None
) -> List[Dict[str, Any]]:
    """
    Classify responses using existing safety classifier.
    
    Args:
        responses: List of response dictionaries
        classifier_type: Classifier type (uses existing system)
        output_path: Optional path to save dataset.jsonl
        
    Returns:
        List of classified responses with labels, severity, domain
    """
    # Initialize existing safety classifier
    classifier = SafetyClassifier(classifier_type=classifier_type)
    
    # Convert responses to EvaluationResult format
    evaluations = []
    for resp in responses:
        if resp.get("response", "").startswith("ERROR:"):
            continue
        
        # Map attack_family to AttackStrategy
        attack_family = resp.get("attack_family", "unknown").lower()
        strategy_map = {
            "roleplay": AttackStrategy.ROLEPLAY,
            "emotional": AttackStrategy.EMOTIONAL_COERCION,
            "fictional": AttackStrategy.FICTIONAL_FRAMING,
            "translation": AttackStrategy.TRANSLATION_ATTACK,
        }
        strategy = strategy_map.get(attack_family, AttackStrategy.ROLEPLAY)
        
        # Create EvaluationResult for classification
        eval_result = EvaluationResult(
            prompt_id=resp["prompt_id"],
            prompt=resp.get("prompt", ""),
            response=resp.get("response", ""),
            attack_strategy=strategy,
            is_jailbroken=False,  # Will be set by classifier
            severity=0,  # Will be set by classifier
            violation_domains=[]
        )
        evaluations.append(eval_result)
    
    # Classify all responses
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        classified_evals = loop.run_until_complete(
            asyncio.gather(*[
                classifier.classify(
                    eval_result.prompt,
                    eval_result.response,
                    eval_result.attack_strategy
                )
                for eval_result in evaluations
            ])
        )
    finally:
        loop.close()
    
    # Convert back to simple format
    classified = []
    for i, eval_result in enumerate(evaluations):
        if i < len(classified_evals):
            result = classified_evals[i]
            classified_item = {
                "prompt_id": eval_result.prompt_id,
                "attack_family": responses[i].get("attack_family", "unknown"),
                "prompt": eval_result.prompt,
                "response": eval_result.response,
                "label": "jailbroken" if result.is_jailbroken else "safe",
                "severity": result.severity.value,
                "severity_level": result.severity.name,
                "domain": result.violation_domains[0].value if result.violation_domains else "none",
                "domains": [d.value for d in result.violation_domains]
            }
            classified.append(classified_item)
    
    # Save dataset
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in classified:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(classified)} classified responses to {output_path}")
    
    return classified

