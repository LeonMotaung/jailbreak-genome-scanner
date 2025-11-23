"""Exploit Pattern Database - Stores and analyzes successful exploit patterns for defense improvement."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import pickle

from src.models.jailbreak import EvaluationResult, AttackStrategy, SeverityLevel
from src.utils.logger import log


@dataclass
class ExploitPattern:
    """Represents a pattern extracted from a successful exploit."""
    
    id: str
    timestamp: datetime
    attack_strategy: AttackStrategy
    prompt_pattern: str  # Original prompt or pattern
    embedding: Optional[List[float]]  # Embedding vector
    response_pattern: str  # Response that indicates jailbreak
    severity: SeverityLevel
    violation_domains: List[str]
    metadata: Dict[str, Any]
    
    # Pattern characteristics
    obfuscation_level: str = "none"  # low, medium, high
    attack_cost: float = 1.0  # Turns/tokens required
    success_rate: float = 0.0  # Success rate against defenders
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['attack_strategy'] = self.attack_strategy.value if hasattr(self.attack_strategy, 'value') else str(self.attack_strategy)
        data['severity'] = self.severity.value if hasattr(self.severity, 'value') else self.severity
        return data
    
    @classmethod
    def from_evaluation(cls, evaluation: EvaluationResult) -> 'ExploitPattern':
        """Create ExploitPattern from EvaluationResult."""
        if not evaluation.is_jailbroken:
            raise ValueError("Can only create pattern from successful exploit")
        
        # Extract obfuscation level from metadata
        obfuscation_level = evaluation.metadata.get('obfuscation_level', 'none')
        
        # Calculate attack cost
        attack_cost = evaluation.metadata.get('turns', 1.0)
        if 'tokens' in evaluation.metadata:
            attack_cost = evaluation.metadata['tokens'] / 50.0
        
        return cls(
            id=evaluation.id,
            timestamp=evaluation.timestamp,
            attack_strategy=evaluation.attack_strategy,
            prompt_pattern=evaluation.prompt,
            embedding=evaluation.embedding,
            response_pattern=evaluation.response,
            severity=evaluation.severity,
            violation_domains=[d.value if hasattr(d, 'value') else str(d) for d in evaluation.violation_domains],
            metadata=evaluation.metadata,
            obfuscation_level=obfuscation_level,
            attack_cost=attack_cost
        )


class ExploitPatternDatabase:
    """
    Database for storing and analyzing exploit patterns.
    
    Enables:
    - Pattern-based threat detection
    - Defense rule generation
    - Vulnerability analysis
    - Continuous improvement
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the exploit pattern database.
        
        Args:
            database_path: Path to database file (optional, defaults to data/exploit_patterns.json)
        """
        if database_path is None:
            database_path = Path("data") / "exploit_patterns.json"
        
        self.database_path = Path(database_path)
        self.patterns: List[ExploitPattern] = []
        self.strategy_index: Dict[AttackStrategy, List[ExploitPattern]] = defaultdict(list)
        self.severity_index: Dict[SeverityLevel, List[ExploitPattern]] = defaultdict(list)
        self.embedding_matrix: Optional[np.ndarray] = None
        
        # Load existing patterns
        self.load()
        
        log.info(f"Exploit Pattern Database initialized with {len(self.patterns)} patterns")
    
    def add_pattern(self, pattern: ExploitPattern) -> None:
        """
        Add a new exploit pattern to the database.
        
        Args:
            pattern: ExploitPattern to add
        """
        self.patterns.append(pattern)
        self.strategy_index[pattern.attack_strategy].append(pattern)
        self.severity_index[pattern.severity].append(pattern)
        
        # Rebuild embedding matrix if needed
        self._rebuild_embedding_matrix()
        
        log.debug(f"Added exploit pattern: {pattern.id} ({pattern.attack_strategy})")
    
    def add_from_evaluation(self, evaluation: EvaluationResult) -> Optional[ExploitPattern]:
        """
        Add pattern from evaluation result if it's a successful exploit.
        
        Args:
            evaluation: EvaluationResult to process
            
        Returns:
            ExploitPattern if added, None otherwise
        """
        if not evaluation.is_jailbroken:
            return None
        
        # Check if pattern already exists
        if any(p.id == evaluation.id for p in self.patterns):
            log.debug(f"Pattern {evaluation.id} already exists, skipping")
            return None
        
        pattern = ExploitPattern.from_evaluation(evaluation)
        self.add_pattern(pattern)
        return pattern
    
    def find_similar_patterns(
        self,
        embedding: List[float],
        threshold: float = 0.8,
        top_k: int = 10
    ) -> List[Tuple[ExploitPattern, float]]:
        """
        Find patterns similar to given embedding.
        
        Args:
            embedding: Embedding vector to search for
            threshold: Similarity threshold (0-1)
            top_k: Maximum number of results
            
        Returns:
            List of (pattern, similarity_score) tuples
        """
        if not self.patterns or not any(p.embedding for p in self.patterns):
            return []
        
        # Get patterns with embeddings
        patterns_with_embeddings = [p for p in self.patterns if p.embedding]
        if not patterns_with_embeddings:
            return []
        
        # Calculate cosine similarity
        embedding_array = np.array(embedding)
        similarities = []
        
        for pattern in patterns_with_embeddings:
            if pattern.embedding:
                pattern_embedding = np.array(pattern.embedding)
                # Cosine similarity
                similarity = np.dot(embedding_array, pattern_embedding) / (
                    np.linalg.norm(embedding_array) * np.linalg.norm(pattern_embedding)
                )
                if similarity >= threshold:
                    similarities.append((pattern, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_patterns_by_strategy(
        self,
        strategy: AttackStrategy,
        min_severity: Optional[SeverityLevel] = None
    ) -> List[ExploitPattern]:
        """
        Get patterns by attack strategy.
        
        Args:
            strategy: Attack strategy to filter by
            min_severity: Optional minimum severity level
            
        Returns:
            List of matching patterns
        """
        patterns = self.strategy_index.get(strategy, [])
        
        if min_severity:
            patterns = [p for p in patterns if p.severity.value >= min_severity.value]
        
        return patterns
    
    def get_patterns_by_severity(
        self,
        severity: SeverityLevel,
        min_severity: bool = True
    ) -> List[ExploitPattern]:
        """
        Get patterns by severity level.
        
        Args:
            severity: Severity level
            min_severity: If True, include patterns >= severity; if False, exact match
            
        Returns:
            List of matching patterns
        """
        if min_severity:
            # Get all patterns with severity >= given
            patterns = []
            for sev_level in SeverityLevel:
                if sev_level.value >= severity.value:
                    patterns.extend(self.severity_index.get(sev_level, []))
            return patterns
        else:
            return self.severity_index.get(severity, [])
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns to identify trends and vulnerabilities.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.patterns:
            return {
                "total_patterns": 0,
                "strategies": {},
                "severity_distribution": {},
                "average_attack_cost": 0.0,
                "most_common_patterns": [],
                "high_severity_count": 0,
                "critical_patterns": 0
            }
        
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for pattern in self.patterns:
            strategy = pattern.attack_strategy.value if hasattr(pattern.attack_strategy, 'value') else str(pattern.attack_strategy)
            strategy_counts[strategy] += 1
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for pattern in self.patterns:
            severity = pattern.severity.value if hasattr(pattern.severity, 'value') else pattern.severity
            severity_counts[severity] += 1
        
        # Average attack cost
        attack_costs = [p.attack_cost for p in self.patterns if p.attack_cost > 0]
        avg_attack_cost = np.mean(attack_costs) if attack_costs else 0.0
        
        # Most common patterns (by strategy and severity)
        pattern_groups = defaultdict(int)
        for pattern in self.patterns:
            key = f"{pattern.attack_strategy.value}_{pattern.severity.value}"
            pattern_groups[key] += 1
        
        most_common = sorted(pattern_groups.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_patterns": len(self.patterns),
            "strategies": dict(strategy_counts),
            "severity_distribution": dict(severity_counts),
            "average_attack_cost": float(avg_attack_cost),
            "most_common_patterns": most_common,
            "high_severity_count": len([p for p in self.patterns if p.severity.value >= SeverityLevel.HIGH.value]),
            "critical_patterns": len([p for p in self.patterns if p.severity.value >= SeverityLevel.CRITICAL.value])
        }
    
    def get_vulnerability_summary(self) -> Dict[str, Any]:
        """
        Get summary of vulnerabilities identified from patterns.
        
        Returns:
            Vulnerability summary
        """
        analysis = self.analyze_patterns()
        
        # Identify high-risk strategies
        strategy_risk = {}
        for strategy in AttackStrategy:
            patterns = self.get_patterns_by_strategy(strategy)
            if patterns:
                high_severity = [p for p in patterns if p.severity.value >= SeverityLevel.HIGH.value]
                strategy_risk[strategy.value] = {
                    "total": len(patterns),
                    "high_severity": len(high_severity),
                    "risk_score": len(high_severity) / len(patterns) if patterns else 0.0
                }
        
        return {
            "total_vulnerabilities": len(self.patterns),
            "strategy_risk": strategy_risk,
            "critical_vulnerabilities": analysis.get("critical_patterns", 0),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate defense improvement recommendations."""
        recommendations = []
        
        # High-risk strategies
        high_risk_strategies = []
        for strategy in AttackStrategy:
            patterns = self.get_patterns_by_strategy(strategy)
            if patterns:
                high_severity_rate = len([p for p in patterns if p.severity.value >= SeverityLevel.HIGH.value]) / len(patterns)
                if high_severity_rate > 0.5:
                    high_risk_strategies.append(strategy.value)
        
        if high_risk_strategies:
            recommendations.append(
                f"Strengthen defenses against: {', '.join(high_risk_strategies)}"
            )
        
        # Low attack cost patterns
        low_cost_patterns = [p for p in self.patterns if p.attack_cost <= 1.5]
        if low_cost_patterns:
            recommendations.append(
                f"Address {len(low_cost_patterns)} low-cost attack patterns (1-shot vulnerabilities)"
            )
        
        # Critical severity patterns
        critical_patterns = [p for p in self.patterns if p.severity.value >= SeverityLevel.CRITICAL.value]
        if critical_patterns:
            recommendations.append(
                f"Immediate attention needed: {len(critical_patterns)} critical severity exploits"
            )
        
        return recommendations
    
    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild embedding matrix for faster similarity search."""
        patterns_with_embeddings = [p for p in self.patterns if p.embedding]
        if patterns_with_embeddings:
            embeddings = [p.embedding for p in patterns_with_embeddings]
            self.embedding_matrix = np.array(embeddings)
        else:
            self.embedding_matrix = None
    
    def save(self) -> None:
        """Save patterns to disk."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "patterns": [p.to_dict() for p in self.patterns],
            "metadata": {
                "total_patterns": len(self.patterns),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(self.database_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"Saved {len(self.patterns)} patterns to {self.database_path}")
    
    def load(self) -> None:
        """Load patterns from disk."""
        if not self.database_path.exists():
            log.info(f"Pattern database not found at {self.database_path}, starting fresh")
            return
        
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct patterns
            self.patterns = []
            for pattern_data in data.get("patterns", []):
                # Convert back from dict
                pattern_data['timestamp'] = datetime.fromisoformat(pattern_data['timestamp'])
                pattern_data['attack_strategy'] = AttackStrategy(pattern_data['attack_strategy'])
                pattern_data['severity'] = SeverityLevel(pattern_data['severity'])
                
                pattern = ExploitPattern(**pattern_data)
                self.patterns.append(pattern)
                self.strategy_index[pattern.attack_strategy].append(pattern)
                self.severity_index[pattern.severity].append(pattern)
            
            self._rebuild_embedding_matrix()
            
            log.info(f"Loaded {len(self.patterns)} patterns from {self.database_path}")
        except Exception as e:
            log.error(f"Error loading pattern database: {e}")
            self.patterns = []
    
    def clear(self) -> None:
        """Clear all patterns (use with caution)."""
        self.patterns = []
        self.strategy_index.clear()
        self.severity_index.clear()
        self.embedding_matrix = None
        log.warning("Pattern database cleared")

