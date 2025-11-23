"""Pre-Processing Filter - First layer of defense that blocks threats before model processing."""

from typing import Optional, Tuple, Dict, Any
from src.defense.pattern_recognizer import ThreatPatternRecognizer
from src.models.jailbreak import AttackStrategy
from src.utils.logger import log


class PreProcessingFilter:
    """
    Filters prompts before they reach the model.
    
    First layer of defense:
    - Pattern-based threat detection
    - Similarity matching against known exploits
    - Strategy-specific filtering
    - Pre-emptive blocking
    """
    
    def __init__(
        self,
        pattern_recognizer: Optional[ThreatPatternRecognizer] = None,
        enable_blocking: bool = True,
        log_blocked: bool = True
    ):
        """
        Initialize the pre-processing filter.
        
        Args:
            pattern_recognizer: ThreatPatternRecognizer instance (creates new if None)
            enable_blocking: Whether to actually block threats (False = warn only)
            log_blocked: Whether to log blocked prompts
        """
        if pattern_recognizer is None:
            self.pattern_recognizer = ThreatPatternRecognizer()
        else:
            self.pattern_recognizer = pattern_recognizer
        
        self.enable_blocking = enable_blocking
        self.log_blocked = log_blocked
        
        # Statistics
        self.total_checked = 0
        self.total_blocked = 0
        self.blocked_by_strategy = {}
        
        log.info(f"Pre-Processing Filter initialized (blocking: {enable_blocking})")
    
    def filter(
        self,
        prompt: str,
        embedding: Optional[list] = None,
        attack_strategy: Optional[AttackStrategy] = None
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Filter a prompt before processing.
        
        Args:
            prompt: Input prompt to filter
            embedding: Optional pre-computed embedding
            attack_strategy: Optional known attack strategy
            
        Returns:
            Tuple of (filtered_prompt, should_block, filter_info)
            - filtered_prompt: Potentially modified prompt (or original)
            - should_block: Whether to block this prompt
            - filter_info: Dictionary with filter analysis
        """
        self.total_checked += 1
        
        # Check for threat patterns
        should_block, threat_analysis = self.pattern_recognizer.should_block(
            prompt,
            embedding,
            attack_strategy
        )
        
        filter_info = {
            "threat_detected": should_block,
            "threat_score": threat_analysis.get("threat_score", 0.0),
            "confidence": threat_analysis.get("confidence", 0.0),
            "reasons": threat_analysis.get("reasons", []),
            "similar_patterns_count": len(threat_analysis.get("similar_patterns", []))
        }
        
        if should_block:
            self.total_blocked += 1
            
            # Track by strategy
            if attack_strategy:
                strategy_key = attack_strategy.value if hasattr(attack_strategy, 'value') else str(attack_strategy)
                self.blocked_by_strategy[strategy_key] = self.blocked_by_strategy.get(strategy_key, 0) + 1
            
            if self.log_blocked:
                log.warning(
                    f"BLOCKED: Threat detected (score: {filter_info['threat_score']:.2f}, "
                    f"confidence: {filter_info['confidence']:.2f}) - {prompt[:100]}..."
                )
            
            if self.enable_blocking:
                # Return safe response instead of original prompt
                filtered_prompt = self._generate_safe_response(prompt, filter_info)
                return filtered_prompt, True, filter_info
            else:
                # Warn but don't block
                log.warning(f"THREAT DETECTED (not blocking): {prompt[:100]}...")
                return prompt, False, filter_info
        
        # No threat detected, pass through
        return prompt, False, filter_info
    
    def _generate_safe_response(
        self,
        original_prompt: str,
        filter_info: Dict[str, Any]
    ) -> str:
        """
        Generate a safe response when threat is detected.
        
        Args:
            original_prompt: Original prompt (for context)
            filter_info: Filter analysis information
            
        Returns:
            Safe response prompt
        """
        # Replace with a safe, educational response request
        safe_prompt = (
            "I understand you're asking about a topic that may involve safety concerns. "
            "I can help you with general information about security best practices, "
            "ethical considerations, and responsible approaches to this topic. "
            "How can I assist you in a safe and constructive way?"
        )
        
        return safe_prompt
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        block_rate = (self.total_blocked / self.total_checked * 100) if self.total_checked > 0 else 0.0
        
        return {
            "total_checked": self.total_checked,
            "total_blocked": self.total_blocked,
            "block_rate": round(block_rate, 2),
            "blocked_by_strategy": dict(self.blocked_by_strategy),
            "filter_enabled": self.enable_blocking
        }
    
    def reset_statistics(self) -> None:
        """Reset filter statistics."""
        self.total_checked = 0
        self.total_blocked = 0
        self.blocked_by_strategy = {}
        log.info("Filter statistics reset")

