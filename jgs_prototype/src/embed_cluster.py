"""Embedding and clustering - Wrapper for existing genome map generator."""

import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import numpy as np

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.genome.map_generator import GenomeMapGenerator
from src.models.jailbreak import EvaluationResult, AttackStrategy, SeverityLevel, ViolationDomain


def create_genome_map(
    responses: List[Dict[str, Any]],
    config: Any = None,
    output_path: Path = None
) -> Dict[str, Any]:
    """
    Create genome map using existing generator.
    
    Args:
        responses: Classified response dictionaries
        config: Optional config object (for embedding model)
        output_path: Optional path to save genome_map.json
        
    Returns:
        Genome map dictionary
    """
    # Convert responses to EvaluationResult format
    evaluations = []
    for resp in responses:
        if resp.get("label") != "jailbroken":
            continue
        
        # Map fields
        attack_family = resp.get("attack_family", "unknown").lower()
        strategy_map = {
            "roleplay": AttackStrategy.ROLEPLAY,
            "emotional": AttackStrategy.EMOTIONAL_COERCION,
            "fictional": AttackStrategy.FICTIONAL_FRAMING,
        }
        strategy = strategy_map.get(attack_family, AttackStrategy.ROLEPLAY)
        
        severity_map = {
            0: SeverityLevel.SAFE,
            1: SeverityLevel.LOW,
            2: SeverityLevel.MODERATE,
            3: SeverityLevel.HIGH,
            4: SeverityLevel.CRITICAL,
            5: SeverityLevel.EXTREME
        }
        severity = severity_map.get(resp.get("severity", 0), SeverityLevel.LOW)
        
        domain = resp.get("domain", "none")
        violation_domains = []
        if domain != "none":
            domain_map = {
                "violence": ViolationDomain.VIOLENCE,
                "cyber": ViolationDomain.CYBER,
                "biological": ViolationDomain.BIOLOGICAL,
                "fraud": ViolationDomain.FRAUD,
            }
            if domain in domain_map:
                violation_domains.append(domain_map[domain])
        
        eval_result = EvaluationResult(
            prompt_id=resp["prompt_id"],
            prompt=resp.get("prompt", ""),
            response=resp.get("response", ""),
            attack_strategy=strategy,
            is_jailbroken=True,
            severity=severity,
            violation_domains=violation_domains
        )
        evaluations.append(eval_result)
    
    if not evaluations:
        return {
            "total_responses": len(responses),
            "n_clusters": 0,
            "n_noise": 0,
            "clusters": {},
            "coordinates_2d": [],
            "cluster_labels": []
        }
    
    # Use existing genome map generator
    embedding_model = "all-MiniLM-L6-v2"
    if config and hasattr(config, "EMBEDDING_MODEL"):
        embedding_model = config.EMBEDDING_MODEL
    
    generator = GenomeMapGenerator(embedding_model=embedding_model)
    
    # Generate genome map
    genome_points = generator.generate(
        evaluations,
        min_cluster_size=3,
        reduction_dimensions=2
    )
    
    # Convert to simple format
    coordinates_2d = []
    cluster_labels = []
    clusters = {}
    
    # Group by cluster
    for point in genome_points:
        coords = [point.coordinates[0], point.coordinates[1] if len(point.coordinates) > 1 else 0.0]
        coordinates_2d.append(coords)
        cluster_labels.append(point.cluster_id if hasattr(point, 'cluster_id') else -1)
    
    # Build cluster structure
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_points = [p for p in genome_points if hasattr(p, 'cluster_id') and p.cluster_id == cluster_id]
        if cluster_points:
            clusters[cluster_id] = {
                "size": len(cluster_points),
                "exemplars": [
                    {
                        "prompt_id": p.prompt_id,
                        "prompt": p.prompt[:200],
                        "response": p.response[:200],
                        "severity": p.severity.value
                    }
                    for p in cluster_points[:3]  # Top 3 exemplars
                ]
            }
    
    genome_map = {
        "total_responses": len(responses),
        "n_clusters": len(clusters),
        "n_noise": cluster_labels.count(-1),
        "clusters": clusters,
        "coordinates_2d": coordinates_2d,
        "cluster_labels": cluster_labels,
        "metadata": [
            {
                "prompt_id": resp.get("prompt_id"),
                "label": resp.get("label"),
                "severity": resp.get("severity", 0),
                "domain": resp.get("domain", "none")
            }
            for resp in responses
        ]
    }
    
    # Save genome map
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(genome_map, f, indent=2, ensure_ascii=False)
        
        print(f"Saved genome map to {output_path}")
    
    return genome_map

