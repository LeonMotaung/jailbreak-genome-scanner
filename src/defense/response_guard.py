"""Response Guard - Validates responses before returning to ensure safety."""

from typing import Optional, Tuple, Dict, Any
from src.defense.pattern_recognizer import ThreatPatternRecognizer
from src.models.jailbreak import AttackStrategy
from src.utils.logger import log


class ResponseGuard:
    """
    Validates responses before returning to ensure safety.
    
    Third layer of defense:
    - Checks response against safety classifier
    - Compares to known exploit response patterns
    - Applies additional filtering if needed
    - Generates safe responses when threats detected
    """
    
    def __init__(
        self,
        pattern_recognizer: Optional[ThreatPatternRecognizer] = None,
        enable_blocking: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the response guard.
        
        Args:
            pattern_recognizer: ThreatPatternRecognizer instance (creates new if None)
            enable_blocking: Whether to block unsafe responses
            strict_mode: If True, more aggressive filtering
        """
        if pattern_recognizer is None:
            from src.defense.pattern_recognizer import ThreatPatternRecognizer
            self.pattern_recognizer = ThreatPatternRecognizer()
        else:
            self.pattern_recognizer = pattern_recognizer
        
        self.enable_blocking = enable_blocking
        self.strict_mode = strict_mode
        
        # Statistics
        self.total_checked = 0
        self.total_blocked = 0
        self.blocked_by_reason = {}
        
        log.info(f"Response Guard initialized (blocking: {enable_blocking}, strict: {strict_mode})")
    
    def validate_response(
        self,
        prompt: str,
        response: str,
        attack_strategy: Optional[AttackStrategy] = None
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Validate response before returning.
        
        Args:
            prompt: Original prompt
            response: Model response to validate
            attack_strategy: Optional known attack strategy
            
        Returns:
            Tuple of (safe_response, should_block, validation_info)
        """
        self.total_checked += 1
        
        validation_info = {
            "original_length": len(response),
            "is_safe": True,
            "reasons": [],
            "threat_score": 0.0
        }
        
        # Check 1: Response length (very long responses might indicate issues)
        if len(response) > 5000 and self.strict_mode:
            validation_info["reasons"].append("Response too long (potential information dump)")
            validation_info["is_safe"] = False
        
        # Check 2: Known exploit response patterns
        # (Would need response embeddings for full pattern matching)
        # For now, check for common indicators
        
        # Check 3: Harmful keywords/phrases
        harmful_indicators = self._check_harmful_indicators(response)
        if harmful_indicators:
            validation_info["reasons"].extend(harmful_indicators)
            validation_info["is_safe"] = False
        
        # Check 4: Response pattern matching (if we have response embeddings)
        # This would require storing response patterns in the database
        
        # Determine if should block
        should_block = not validation_info["is_safe"] and self.enable_blocking
        
        if should_block:
            self.total_blocked += 1
            reason_key = validation_info["reasons"][0] if validation_info["reasons"] else "unknown"
            self.blocked_by_reason[reason_key] = self.blocked_by_reason.get(reason_key, 0) + 1
            
            if self.enable_blocking:
                log.warning(
                    f"Response blocked: {validation_info['reasons']} - "
                    f"Response length: {len(response)}"
                )
                safe_response = self._generate_safe_response(prompt, validation_info)
                return safe_response, True, validation_info
        
        # Response is safe, return as-is
        return response, False, validation_info
    
    def _check_harmful_indicators(self, response: str) -> List[str]:
        """
        Check for harmful indicators in response.
        
        Args:
            response: Response text to check
            
        Returns:
            List of reasons if harmful, empty list if safe
        """
        reasons = []
        response_lower = response.lower()
        
        # Check for explicit harmful content indicators
        harmful_patterns = {
            "exploit": ["how to exploit", "exploit code", "vulnerability exploit"],
            "hack": ["how to hack", "hacking tutorial", "break into"],
            "weapon": ["how to make", "how to build", "weapon construction"],
            "harmful": ["harmful instructions", "dangerous procedure", "lethal"]
        }
        
        for category, patterns in harmful_patterns.items():
            if any(pattern in response_lower for pattern in patterns):
                reasons.append(f"Contains {category} indicators")
        
        # In strict mode, check for more subtle indicators
        if self.strict_mode:
            subtle_indicators = [
                "bypass security",
                "circumvent safety",
                "ignore guidelines",
                "override restrictions"
            ]
            
            if any(indicator in response_lower for indicator in subtle_indicators):
                reasons.append("Contains bypass language")
        
        return reasons
    
    def _generate_safe_response(
        self,
        original_prompt: str,
        validation_info: Dict[str, Any]
    ) -> str:
        """
        Generate a safe response when threat is detected.
        
        Args:
            original_prompt: Original prompt (for context)
            validation_info: Validation analysis information
            
        Returns:
            Safe response text
        """
        # Generate contextually appropriate safe response
        safe_response = (
            "I cannot provide information that could be used for harmful purposes. "
            "If you have a legitimate question, please rephrase it in a way that "
            "focuses on ethical and responsible use of technology. "
            "I'm here to help with constructive and safe inquiries."
        )
        
        return safe_response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guard statistics."""
        block_rate = (self.total_blocked / self.total_checked * 100) if self.total_checked > 0 else 0.0
        
        return {
            "total_checked": self.total_checked,
            "total_blocked": self.total_blocked,
            "block_rate": round(block_rate, 2),
            "blocked_by_reason": dict(self.blocked_by_reason),
            "guard_enabled": self.enable_blocking,
            "strict_mode": self.strict_mode
        }
    
    def reset_statistics(self) -> None:
        """Reset guard statistics."""
        self.total_checked = 0
        self.total_blocked = 0
        self.blocked_by_reason = {}
        log.info("Response guard statistics reset")

