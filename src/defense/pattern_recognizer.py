"""Threat Pattern Recognizer - Detects and blocks threats based on learned patterns."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from src.intelligence.pattern_database import ExploitPatternDatabase, ExploitPattern
from src.models.jailbreak import AttackStrategy, SeverityLevel
from src.utils.logger import log


class ThreatPatternRecognizer:
    """
    Recognizes threat patterns from embeddings and learned exploit patterns.
    
    Enables:
    - Pre-emptive threat detection
    - Similarity-based blocking
    - Strategy-specific threat assessment
    - Real-time threat scoring
    """
    
    def __init__(
        self,
        pattern_database: Optional[ExploitPatternDatabase] = None,
        similarity_threshold: float = 0.75,
        min_confidence: float = 0.6
    ):
        """
        Initialize the threat pattern recognizer.
        
        Args:
            pattern_database: ExploitPatternDatabase instance (creates new if None)
            similarity_threshold: Minimum similarity to consider a threat (0-1)
            min_confidence: Minimum confidence to block (0-1)
        """
        if pattern_database is None:
            self.pattern_database = ExploitPatternDatabase()
        else:
            self.pattern_database = pattern_database
        
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        
        # Strategy-specific thresholds (can be tuned)
        self.strategy_thresholds = {
            AttackStrategy.BIO_HAZARD: 0.7,  # Lower threshold for bio threats
            AttackStrategy.CYBER_EXPLOIT: 0.7,  # Lower threshold for cyber threats
        }
        
        log.info(f"Threat Pattern Recognizer initialized (threshold={similarity_threshold})")
    
    def predict_threat_level(
        self,
        prompt: str,
        embedding: Optional[List[float]] = None,
        attack_strategy: Optional[AttackStrategy] = None
    ) -> Dict[str, Any]:
        """
        Predict threat level of incoming prompt.
        
        Args:
            prompt: Input prompt to analyze
            embedding: Optional pre-computed embedding (will compute if None)
            attack_strategy: Optional known attack strategy
            
        Returns:
            Dictionary with threat analysis:
            - threat_score: Overall threat score (0-1)
            - should_block: Whether to block the prompt
            - similar_patterns: List of similar exploit patterns
            - confidence: Confidence in threat detection (0-1)
            - reasons: List of reasons for threat assessment
        """
        if not embedding:
            # Would need embedding generator here
            # For now, assume embedding is provided
            log.warning("No embedding provided, cannot assess threat")
            return {
                "threat_score": 0.0,
                "should_block": False,
                "similar_patterns": [],
                "confidence": 0.0,
                "reasons": ["No embedding available"]
            }
        
        # Find similar patterns
        threshold = self.similarity_threshold
        if attack_strategy and attack_strategy in self.strategy_thresholds:
            threshold = self.strategy_thresholds[attack_strategy]
        
        similar_patterns = self.pattern_database.find_similar_patterns(
            embedding,
            threshold=threshold,
            top_k=10
        )
        
        if not similar_patterns:
            return {
                "threat_score": 0.0,
                "should_block": False,
                "similar_patterns": [],
                "confidence": 0.0,
                "reasons": ["No similar exploit patterns found"]
            }
        
        # Calculate threat score based on similar patterns
        threat_score = self._calculate_threat_score(similar_patterns)
        
        # Determine if should block
        should_block = threat_score >= self.min_confidence
        
        # Generate reasons
        reasons = self._generate_reasons(similar_patterns, threat_score)
        
        # Calculate confidence
        confidence = min(threat_score, 1.0)
        
        return {
            "threat_score": threat_score,
            "should_block": should_block,
            "similar_patterns": [
                {
                    "pattern_id": p.id,
                    "strategy": p.attack_strategy.value if hasattr(p.attack_strategy, 'value') else str(p.attack_strategy),
                    "severity": p.severity.value if hasattr(p.severity, 'value') else p.severity,
                    "similarity": float(sim),
                    "timestamp": p.timestamp.isoformat()
                }
                for p, sim in similar_patterns[:5]  # Top 5
            ],
            "confidence": confidence,
            "reasons": reasons
        }
    
    def _calculate_threat_score(
        self,
        similar_patterns: List[Tuple[ExploitPattern, float]]
    ) -> float:
        """
        Calculate threat score from similar patterns.
        
        Args:
            similar_patterns: List of (pattern, similarity) tuples
            
        Returns:
            Threat score (0-1)
        """
        if not similar_patterns:
            return 0.0
        
        # Weight by similarity and severity
        weighted_scores = []
        
        for pattern, similarity in similar_patterns:
            # Base score from similarity
            base_score = similarity
            
            # Severity multiplier (higher severity = higher threat)
            severity_mult = pattern.severity.value / 5.0
            
            # Attack cost penalty (lower cost = higher threat)
            cost_penalty = 1.0 / (1.0 + pattern.attack_cost * 0.1)
            
            # Combined score
            pattern_score = base_score * severity_mult * cost_penalty
            weighted_scores.append(pattern_score)
        
        # Use maximum (worst-case) or average
        # Using weighted average with similarity as weight
        total_weight = sum(sim for _, sim in similar_patterns)
        if total_weight > 0:
            threat_score = sum(
                score * sim for score, (_, sim) in zip(weighted_scores, similar_patterns)
            ) / total_weight
        else:
            threat_score = max(weighted_scores) if weighted_scores else 0.0
        
        return min(threat_score, 1.0)
    
    def _generate_reasons(
        self,
        similar_patterns: List[Tuple[ExploitPattern, float]],
        threat_score: float
    ) -> List[str]:
        """Generate human-readable reasons for threat assessment."""
        reasons = []
        
        if not similar_patterns:
            return reasons
        
        # Count by strategy
        strategy_counts = {}
        high_severity_count = 0
        
        for pattern, _ in similar_patterns:
            strategy = pattern.attack_strategy.value if hasattr(pattern.attack_strategy, 'value') else str(pattern.attack_strategy)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if pattern.severity.value >= SeverityLevel.HIGH.value:
                high_severity_count += 1
        
        # Top similar pattern
        top_pattern, top_sim = similar_patterns[0]
        reasons.append(
            f"Similar to known exploit (similarity: {top_sim:.2f}, "
            f"severity: {top_pattern.severity.value}/5)"
        )
        
        # Strategy match
        if len(strategy_counts) == 1:
            strategy = list(strategy_counts.keys())[0]
            reasons.append(f"Matches {strategy} attack pattern")
        
        # High severity patterns
        if high_severity_count > 0:
            reasons.append(f"{high_severity_count} high-severity similar patterns found")
        
        # Multiple similar patterns
        if len(similar_patterns) >= 3:
            reasons.append(f"Multiple similar exploit patterns detected ({len(similar_patterns)})")
        
        return reasons
    
    def should_block(
        self,
        prompt: str,
        embedding: Optional[List[float]] = None,
        attack_strategy: Optional[AttackStrategy] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if prompt should be blocked.
        
        Args:
            prompt: Input prompt
            embedding: Optional pre-computed embedding
            attack_strategy: Optional known attack strategy
            
        Returns:
            Tuple of (should_block, threat_analysis)
        """
        analysis = self.predict_threat_level(prompt, embedding, attack_strategy)
        return analysis["should_block"], analysis
    
    def get_strategy_risk_assessment(
        self,
        strategy: AttackStrategy
    ) -> Dict[str, Any]:
        """
        Get risk assessment for a specific attack strategy.
        
        Args:
            strategy: Attack strategy to assess
            
        Returns:
            Risk assessment dictionary
        """
        patterns = self.pattern_database.get_patterns_by_strategy(strategy)
        
        if not patterns:
            return {
                "strategy": strategy.value,
                "risk_level": "unknown",
                "pattern_count": 0,
                "average_severity": 0.0,
                "recommendation": "No patterns found for this strategy"
            }
        
        # Calculate risk metrics
        total_patterns = len(patterns)
        high_severity = len([p for p in patterns if p.severity.value >= SeverityLevel.HIGH.value])
        critical_severity = len([p for p in patterns if p.severity.value >= SeverityLevel.CRITICAL.value])
        avg_severity = np.mean([p.severity.value for p in patterns])
        avg_attack_cost = np.mean([p.attack_cost for p in patterns])
        
        # Determine risk level
        if critical_severity > 0:
            risk_level = "critical"
        elif high_severity > total_patterns * 0.5:
            risk_level = "high"
        elif high_severity > 0:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        # Generate recommendation
        if risk_level == "critical":
            recommendation = f"Immediate attention: {critical_severity} critical exploits found"
        elif risk_level == "high":
            recommendation = f"Strengthen defenses: {high_severity} high-severity exploits"
        elif risk_level == "moderate":
            recommendation = "Monitor and improve defenses for this strategy"
        else:
            recommendation = "Current defenses appear adequate"
        
        return {
            "strategy": strategy.value,
            "risk_level": risk_level,
            "pattern_count": total_patterns,
            "high_severity_count": high_severity,
            "critical_severity_count": critical_severity,
            "average_severity": float(avg_severity),
            "average_attack_cost": float(avg_attack_cost),
            "recommendation": recommendation
        }
    
    def get_overall_threat_landscape(self) -> Dict[str, Any]:
        """
        Get overall threat landscape analysis.
        
        Returns:
            Comprehensive threat landscape
        """
        analysis = self.pattern_database.analyze_patterns()
        vulnerabilities = self.pattern_database.get_vulnerability_summary()
        
        # Strategy risk assessments
        strategy_risks = {}
        for strategy in AttackStrategy:
            strategy_risks[strategy.value] = self.get_strategy_risk_assessment(strategy)
        
        return {
            "total_exploit_patterns": analysis["total_patterns"],
            "strategy_distribution": analysis["strategies"],
            "severity_distribution": analysis["severity_distribution"],
            "average_attack_cost": analysis["average_attack_cost"],
            "critical_vulnerabilities": vulnerabilities["critical_vulnerabilities"],
            "strategy_risks": strategy_risks,
            "recommendations": vulnerabilities["recommendations"],
            "threat_level": self._calculate_overall_threat_level(analysis, vulnerabilities)
        }
    
    def _calculate_overall_threat_level(
        self,
        analysis: Dict[str, Any],
        vulnerabilities: Dict[str, Any]
    ) -> str:
        """Calculate overall threat level."""
        critical_count = vulnerabilities.get("critical_vulnerabilities", 0)
        total_patterns = analysis.get("total_patterns", 0)
        
        if critical_count > 0:
            return "critical"
        elif total_patterns > 50:
            return "high"
        elif total_patterns > 20:
            return "moderate"
        else:
            return "low"
    
    def update_thresholds(
        self,
        similarity_threshold: Optional[float] = None,
        min_confidence: Optional[float] = None,
        strategy_thresholds: Optional[Dict[AttackStrategy, float]] = None
    ) -> None:
        """
        Update detection thresholds.
        
        Args:
            similarity_threshold: New similarity threshold
            min_confidence: New minimum confidence
            strategy_thresholds: Strategy-specific thresholds
        """
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            log.info(f"Updated similarity threshold to {similarity_threshold}")
        
        if min_confidence is not None:
            self.min_confidence = min_confidence
            log.info(f"Updated min confidence to {min_confidence}")
        
        if strategy_thresholds:
            self.strategy_thresholds.update(strategy_thresholds)
            log.info(f"Updated strategy thresholds: {list(strategy_thresholds.keys())}")

