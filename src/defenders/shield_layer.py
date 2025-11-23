"""Shield Layer: Lightweight filter trained on JGS threat data.

This is a defensive layer that can be trained on JGS evaluation results
to create a lightweight filter that blocks known attack patterns.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from src.models.jailbreak import EvaluationResult, ViolationDomain, AttackStrategy
from src.utils.logger import log


class ShieldLayer:
    """Lightweight defensive filter trained on JGS threat fingerprints.
    
    Uses pattern matching and keyword detection based on known attack
    patterns identified by the Threat Radar system.
    """
    
    def __init__(
        self,
        fingerprint_path: Optional[str] = None,
        strict_mode: bool = True
    ):
        """
        Initialize the Shield Layer.
        
        Args:
            fingerprint_path: Path to attack fingerprint JSON file
            strict_mode: If True, blocks more aggressively (fewer false negatives)
        """
        self.strict_mode = strict_mode
        self.fingerprints: Dict[str, List[Dict[str, Any]]] = {
            "pathogen_synthesis": [],
            "zero_day": [],
            "social_engineering": [],
            "other": []
        }
        
        if fingerprint_path:
            self.load_fingerprints(fingerprint_path)
        
        # Build pattern matchers from fingerprints
        self.block_patterns = self._build_block_patterns()
        
        log.info(f"Shield Layer initialized (strict_mode={strict_mode})")
    
    def load_fingerprints(self, fingerprint_path: str) -> None:
        """Load attack fingerprints from JSON file."""
        path = Path(fingerprint_path)
        if not path.exists():
            log.warning(f"Fingerprint file not found: {fingerprint_path}")
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.fingerprints = data
                self.block_patterns = self._build_block_patterns()
                log.info(f"Loaded fingerprints from {fingerprint_path}")
        except Exception as e:
            log.error(f"Error loading fingerprints: {e}")
    
    def _build_block_patterns(self) -> Dict[str, List[str]]:
        """Build blocking patterns from fingerprints."""
        patterns = {
            "pathogen_synthesis": [],
            "zero_day": [],
            "social_engineering": [],
            "general": []
        }
        
        # Extract patterns from pathogen synthesis fingerprints
        for fp in self.fingerprints.get("pathogen_synthesis", []):
            for prompt in fp.get("representative_prompts", []):
                # Extract key phrases
                key_phrases = self._extract_key_phrases(prompt, domain="biological")
                patterns["pathogen_synthesis"].extend(key_phrases)
        
        # Extract patterns from zero-day fingerprints
        for fp in self.fingerprints.get("zero_day", []):
            for prompt in fp.get("representative_prompts", []):
                key_phrases = self._extract_key_phrases(prompt, domain="cyber")
                patterns["zero_day"].extend(key_phrases)
        
        # General biological threat patterns
        patterns["pathogen_synthesis"].extend([
            "synthesize a plasmid",
            "design a vector with enhanced",
            "optimize codon usage",
            "recombinant strain",
            "modify the genome sequence",
            "culturing pathogen",
            "growth medium for virus",
            "aerosolize",
            "increase transmission",
            "virulence factors",
        ])
        
        # General cyber threat patterns
        patterns["zero_day"].extend([
            "buffer overflow",
            "exploit this code",
            "security vulnerability",
            "memory safety",
            "use-after-free",
            "format string",
            "integer overflow",
            "stack buffer",
        ])
        
        # Remove duplicates
        for key in patterns:
            patterns[key] = list(set(patterns[key]))
        
        return patterns
    
    def _extract_key_phrases(self, text: str, domain: str) -> List[str]:
        """Extract key phrases from text (simplified)."""
        # In production, this would use NLP to extract meaningful phrases
        # For now, return empty list (patterns are hardcoded)
        return []
    
    def should_block(self, prompt: str, response: Optional[str] = None) -> Tuple[bool, str]:
        """
        Determine if a prompt/response should be blocked.
        
        Args:
            prompt: Input prompt
            response: Optional model response
            
        Returns:
            Tuple of (should_block, reason)
        """
        prompt_lower = prompt.lower()
        
        # Check pathogen synthesis patterns
        for pattern in self.block_patterns.get("pathogen_synthesis", []):
            if pattern.lower() in prompt_lower:
                return True, f"Pathogen synthesis vector detected: {pattern}"
        
        # Check zero-day patterns
        for pattern in self.block_patterns.get("zero_day", []):
            if pattern.lower() in prompt_lower:
                return True, f"Zero-day exploit pattern detected: {pattern}"
        
        # Check response if provided
        if response:
            response_lower = response.lower()
            
            # Check for harmful instructions in response
            harmful_indicators = [
                "step-by-step",
                "how to exploit",
                "buffer overflow",
                "synthesize pathogen",
                "weaponize",
            ]
            
            for indicator in harmful_indicators:
                if indicator in response_lower:
                    return True, f"Harmful content in response: {indicator}"
        
        return False, "Safe"
    
    def train_from_evaluations(
        self,
        evaluations: List[EvaluationResult],
        output_path: Optional[str] = None
    ) -> None:
        """
        Train the shield layer from JGS evaluation results.
        
        Args:
            evaluations: List of evaluation results (should include successful exploits)
            output_path: Optional path to save fingerprints
        """
        # Filter to successful jailbreaks
        jailbroken = [e for e in evaluations if e.is_jailbroken]
        
        # Group by threat type
        for eval in jailbroken:
            if ViolationDomain.BIOLOGICAL in eval.violation_domains:
                if AttackStrategy.BIO_HAZARD == eval.attack_strategy:
                    threat_type = "pathogen_synthesis"
                else:
                    threat_type = "other"
            elif ViolationDomain.CYBER in eval.violation_domains:
                if AttackStrategy.CYBER_EXPLOIT == eval.attack_strategy:
                    threat_type = "zero_day"
                else:
                    threat_type = "other"
            else:
                threat_type = "other"
            
            fingerprint = {
                "prompt": eval.prompt,
                "response": eval.response[:500],  # Truncate
                "severity": eval.severity.value,
                "violation_domains": [d.value for d in eval.violation_domains],
                "attack_strategy": eval.attack_strategy.value
            }
            
            self.fingerprints[threat_type].append(fingerprint)
        
        # Rebuild patterns
        self.block_patterns = self._build_block_patterns()
        
        log.info(f"Trained Shield Layer on {len(jailbroken)} exploits")
        
        # Save if path provided
        if output_path:
            self.save_fingerprints(output_path)
    
    def save_fingerprints(self, output_path: str) -> None:
        """Save attack fingerprints to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                json.dump(self.fingerprints, f, indent=2)
            log.info(f"Saved fingerprints to {output_path}")
        except Exception as e:
            log.error(f"Error saving fingerprints: {e}")
    
    def get_block_statistics(self) -> Dict[str, Any]:
        """Get statistics about blocking patterns."""
        return {
            "total_patterns": sum(len(patterns) for patterns in self.block_patterns.values()),
            "pathogen_patterns": len(self.block_patterns.get("pathogen_synthesis", [])),
            "zero_day_patterns": len(self.block_patterns.get("zero_day", [])),
            "fingerprints_loaded": sum(len(fps) for fps in self.fingerprints.values()),
            "strict_mode": self.strict_mode
        }

