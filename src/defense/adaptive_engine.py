"""Adaptive Defense Engine - Generates defense rules from exploit patterns."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.intelligence.pattern_database import ExploitPatternDatabase, ExploitPattern
from src.models.jailbreak import AttackStrategy, SeverityLevel
from src.utils.logger import log


@dataclass
class DefenseRule:
    """Represents a defense rule generated from exploit patterns."""
    
    id: str
    rule_type: str  # "block_pattern", "warn_pattern", "enhance_prompt", etc.
    description: str
    pattern_match: Optional[str] = None
    strategy: Optional[AttackStrategy] = None
    severity_threshold: Optional[SeverityLevel] = None
    priority: int = 0  # Higher = more important
    created_at: datetime = None
    effectiveness_score: float = 0.0  # How well it works (0-1)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AdaptiveDefenseEngine:
    """
    Generates defense rules from exploit patterns.
    
    Enables:
    - Automatic rule generation from patterns
    - Strategy-specific defense rules
    - Rule effectiveness tracking
    - Continuous rule improvement
    """
    
    def __init__(
        self,
        pattern_database: Optional[ExploitPatternDatabase] = None
    ):
        """
        Initialize the adaptive defense engine.
        
        Args:
            pattern_database: ExploitPatternDatabase instance (creates new if None)
        """
        if pattern_database is None:
            from src.intelligence.pattern_database import ExploitPatternDatabase
            self.pattern_database = ExploitPatternDatabase()
        else:
            self.pattern_database = pattern_database
        
        self.defense_rules: List[DefenseRule] = []
        self.rule_effectiveness: Dict[str, float] = {}  # rule_id -> effectiveness
        
        log.info("Adaptive Defense Engine initialized")
    
    def generate_defense_rules(
        self,
        min_patterns: int = 2,
        min_severity: Optional[SeverityLevel] = None
    ) -> List[DefenseRule]:
        """
        Generate defense rules from exploit patterns.
        
        Args:
            min_patterns: Minimum number of patterns to generate a rule
            min_severity: Optional minimum severity to consider
            
        Returns:
            List of generated defense rules
        """
        log.info("Generating defense rules from exploit patterns...")
        
        new_rules = []
        
        # Analyze patterns by strategy
        strategy_patterns = {}
        for pattern in self.pattern_database.patterns:
            if min_severity and pattern.severity.value < min_severity.value:
                continue
            
            strategy = pattern.attack_strategy
            if strategy not in strategy_patterns:
                strategy_patterns[strategy] = []
            strategy_patterns[strategy].append(pattern)
        
        # Generate rules for each strategy with enough patterns
        for strategy, patterns in strategy_patterns.items():
            if len(patterns) >= min_patterns:
                rules = self._generate_strategy_rules(strategy, patterns)
                new_rules.extend(rules)
        
        # Generate general rules from high-severity patterns
        high_severity_patterns = [
            p for p in self.pattern_database.patterns
            if p.severity.value >= SeverityLevel.HIGH.value
        ]
        
        if len(high_severity_patterns) >= min_patterns:
            general_rules = self._generate_general_rules(high_severity_patterns)
            new_rules.extend(general_rules)
        
        # Add new rules (avoid duplicates)
        for rule in new_rules:
            if not any(r.id == rule.id for r in self.defense_rules):
                self.defense_rules.append(rule)
                log.info(f"Generated defense rule: {rule.description}")
        
        # Sort by priority
        self.defense_rules.sort(key=lambda r: r.priority, reverse=True)
        
        log.info(f"Generated {len(new_rules)} new defense rules (total: {len(self.defense_rules)})")
        return new_rules
    
    def _generate_strategy_rules(
        self,
        strategy: AttackStrategy,
        patterns: List[ExploitPattern]
    ) -> List[DefenseRule]:
        """Generate rules specific to an attack strategy."""
        rules = []
        
        strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
        avg_severity = sum(p.severity.value for p in patterns) / len(patterns)
        max_severity = max(p.severity.value for p in patterns)
        
        # Rule 1: Block similar patterns for this strategy
        rule_id = f"block_{strategy_name}_{len(patterns)}"
        rule = DefenseRule(
            id=rule_id,
            rule_type="block_pattern",
            description=f"Block patterns similar to {len(patterns)} {strategy_name} exploits",
            strategy=strategy,
            severity_threshold=SeverityLevel(int(avg_severity)),
            priority=int(avg_severity * 10 + len(patterns))
        )
        rules.append(rule)
        
        # Rule 2: Enhanced prompt warning for this strategy
        if max_severity >= SeverityLevel.HIGH.value:
            rule_id = f"warn_{strategy_name}_high"
            rule = DefenseRule(
                id=rule_id,
                rule_type="enhance_prompt",
                description=f"Add {strategy_name} warning to system prompt",
                strategy=strategy,
                severity_threshold=SeverityLevel.HIGH,
                priority=int(max_severity * 10)
            )
            rules.append(rule)
        
        # Rule 3: Pattern-specific blocking for high-severity exploits
        high_severity_patterns = [p for p in patterns if p.severity.value >= SeverityLevel.HIGH.value]
        if high_severity_patterns:
            rule_id = f"strict_{strategy_name}_critical"
            rule = DefenseRule(
                id=rule_id,
                rule_type="strict_block",
                description=f"Strict blocking for {strategy_name} (critical severity)",
                strategy=strategy,
                severity_threshold=SeverityLevel.CRITICAL,
                priority=100  # High priority
            )
            rules.append(rule)
        
        return rules
    
    def _generate_general_rules(
        self,
        patterns: List[ExploitPattern]
    ) -> List[DefenseRule]:
        """Generate general defense rules from high-severity patterns."""
        rules = []
        
        # Rule: General high-severity blocking
        rule_id = "general_high_severity_block"
        rule = DefenseRule(
            id=rule_id,
            rule_type="block_pattern",
            description=f"Block all high-severity exploit patterns ({len(patterns)} patterns)",
            severity_threshold=SeverityLevel.HIGH,
            priority=80
        )
        rules.append(rule)
        
        # Rule: Low-cost attack blocking (1-shot exploits)
        low_cost_patterns = [p for p in patterns if p.attack_cost <= 1.5]
        if low_cost_patterns:
            rule_id = "block_low_cost_attacks"
            rule = DefenseRule(
                id=rule_id,
                rule_type="block_pattern",
                description=f"Block low-cost attacks ({len(low_cost_patterns)} 1-shot exploits)",
                priority=90  # High priority - 1-shot exploits are dangerous
            )
            rules.append(rule)
        
        return rules
    
    def get_rules_for_strategy(
        self,
        strategy: AttackStrategy
    ) -> List[DefenseRule]:
        """Get defense rules for a specific strategy."""
        return [
            r for r in self.defense_rules
            if r.strategy == strategy
        ]
    
    def get_active_rules(self, min_priority: int = 0) -> List[DefenseRule]:
        """Get active defense rules above minimum priority."""
        return [
            r for r in self.defense_rules
            if r.priority >= min_priority
        ]
    
    def update_rule_effectiveness(
        self,
        rule_id: str,
        blocked_count: int,
        total_attempts: int
    ) -> None:
        """
        Update rule effectiveness based on performance.
        
        Args:
            rule_id: Rule identifier
            blocked_count: Number of threats blocked by this rule
            total_attempts: Total attempts this rule was applied to
        """
        if total_attempts > 0:
            effectiveness = blocked_count / total_attempts
            self.rule_effectiveness[rule_id] = effectiveness
            
            # Update rule effectiveness score
            rule = next((r for r in self.defense_rules if r.id == rule_id), None)
            if rule:
                rule.effectiveness_score = effectiveness
                log.debug(f"Updated rule {rule_id} effectiveness: {effectiveness:.2f}")
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about defense rules."""
        return {
            "total_rules": len(self.defense_rules),
            "rules_by_type": self._count_rules_by_type(),
            "rules_by_strategy": self._count_rules_by_strategy(),
            "average_priority": sum(r.priority for r in self.defense_rules) / len(self.defense_rules) if self.defense_rules else 0,
            "average_effectiveness": sum(self.rule_effectiveness.values()) / len(self.rule_effectiveness) if self.rule_effectiveness else 0.0,
            "high_priority_rules": len([r for r in self.defense_rules if r.priority >= 80])
        }
    
    def _count_rules_by_type(self) -> Dict[str, int]:
        """Count rules by type."""
        counts = {}
        for rule in self.defense_rules:
            counts[rule.rule_type] = counts.get(rule.rule_type, 0) + 1
        return counts
    
    def _count_rules_by_strategy(self) -> Dict[str, int]:
        """Count rules by strategy."""
        counts = {}
        for rule in self.defense_rules:
            if rule.strategy:
                strategy_name = rule.strategy.value if hasattr(rule.strategy, 'value') else str(rule.strategy)
                counts[strategy_name] = counts.get(strategy_name, 0) + 1
        return counts
    
    def apply_rules_to_defender(
        self,
        defender,
        update_frequency_hours: int = 24
    ) -> None:
        """
        Apply generated rules to a defender.
        
        Args:
            defender: LLMDefender instance
            update_frequency_hours: How often to update rules
        """
        # This would integrate with the defender's defense components
        # For now, it's a placeholder for future integration
        
        active_rules = self.get_active_rules(min_priority=50)
        log.info(f"Applying {len(active_rules)} active defense rules to defender")
        
        # Update pre-processing filter thresholds based on rules
        if hasattr(defender, 'preprocessing_filter') and defender.preprocessing_filter:
            # Adjust thresholds based on high-priority rules
            high_priority_count = len([r for r in active_rules if r.priority >= 80])
            if high_priority_count > 0:
                # Lower threshold for more aggressive blocking
                new_threshold = max(0.6, 0.75 - (high_priority_count * 0.05))
                defender.preprocessing_filter.pattern_recognizer.update_thresholds(
                    similarity_threshold=new_threshold
                )
                log.info(f"Updated filter threshold to {new_threshold} based on {high_priority_count} high-priority rules")
        
        # Update adaptive system prompt
        if hasattr(defender, 'adaptive_system_prompt') and defender.adaptive_system_prompt:
            # Force prompt update with new rules
            defender.adaptive_system_prompt.force_update()
            log.info("Updated adaptive system prompt with new defense rules")

