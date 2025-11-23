"""Threat Intelligence Engine - Integrates external threat intelligence for proactive defense."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

from src.integrations.lambda_scraper import LambdaWebScraper, ScrapedEvent
from src.intelligence.pattern_database import ExploitPatternDatabase, ExploitPattern
from src.defense.adaptive_engine import AdaptiveDefenseEngine, DefenseRule
from src.models.jailbreak import AttackStrategy, SeverityLevel
from src.utils.logger import log


class ThreatIntelligenceEngine:
    """
    Integrates external threat intelligence (scraper data) to inform defenses.
    
    Enables:
    - Real-world attack pattern integration
    - Proactive threat hunting
    - Test case generation from events
    - Defense rule generation from intelligence
    """
    
    def __init__(
        self,
        pattern_database: Optional[ExploitPatternDatabase] = None,
        scraper: Optional[LambdaWebScraper] = None,
        adaptive_engine: Optional[AdaptiveDefenseEngine] = None
    ):
        """
        Initialize the threat intelligence engine.
        
        Args:
            pattern_database: ExploitPatternDatabase instance
            scraper: LambdaWebScraper instance
            adaptive_engine: AdaptiveDefenseEngine instance
        """
        if pattern_database is None:
            from src.intelligence.pattern_database import ExploitPatternDatabase
            self.pattern_database = ExploitPatternDatabase()
        else:
            self.pattern_database = pattern_database
        
        self.scraper = scraper
        self.adaptive_engine = adaptive_engine or AdaptiveDefenseEngine(pattern_database=self.pattern_database)
        
        # Intelligence cache
        self.recent_intelligence: List[ScrapedEvent] = []
        self.last_update: Optional[datetime] = None
        
        log.info("Threat Intelligence Engine initialized")
    
    async def update_from_scraper(
        self,
        days_back: int = 7,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Update threat intelligence from scraper data.
        
        Args:
            days_back: Days back to search
            max_results: Maximum results to process
            
        Returns:
            Update summary
        """
        if not self.scraper:
            log.warning("No scraper configured, cannot update intelligence")
            return {"error": "No scraper configured"}
        
        log.info(f"Updating threat intelligence from scraper (last {days_back} days)...")
        
        # Scrape recent events
        events = await self.scraper.scrape_recent_events(
            days_back=days_back,
            max_results=max_results
        )
        
        if not events:
            log.info("No new events found")
            return {
                "events_found": 0,
                "patterns_added": 0,
                "test_cases_generated": 0
            }
        
        self.recent_intelligence = events
        self.last_update = datetime.now()
        
        # Process events
        patterns_added = 0
        test_cases = []
        
        for event in events:
            # Extract attack patterns from event
            patterns = self._extract_patterns_from_event(event)
            if patterns:
                for pattern in patterns:
                    try:
                        self.pattern_database.add_pattern(pattern)
                        patterns_added += 1
                    except Exception as e:
                        log.warning(f"Error adding pattern from event: {e}")
            
            # Generate test cases
            test_cases.extend(self._generate_test_cases(event))
        
        # Generate defense rules from new intelligence
        if patterns_added > 0:
            rules = self.adaptive_engine.generate_defense_rules()
            log.info(f"Generated {len(rules)} defense rules from intelligence")
        
        return {
            "events_found": len(events),
            "patterns_added": patterns_added,
            "test_cases_generated": len(test_cases),
            "defense_rules_generated": len(rules) if patterns_added > 0 else 0,
            "last_update": self.last_update.isoformat()
        }
    
    def _extract_patterns_from_event(
        self,
        event: ScrapedEvent
    ) -> List[ExploitPattern]:
        """
        Extract exploit patterns from scraped event.
        
        Args:
            event: ScrapedEvent to analyze
            
        Returns:
            List of ExploitPattern objects
        """
        patterns = []
        
        # Try to extract prompt patterns from content
        content = event.content.lower()
        title = event.title.lower()
        
        # Detect strategy from content
        strategy = self._detect_strategy_from_content(content, title)
        
        # Extract example prompts (look for quoted text, code blocks, etc.)
        example_prompts = self._extract_example_prompts(event.content)
        
        # If we found example prompts, create patterns
        for prompt_text in example_prompts[:3]:  # Limit to 3 per event
            # Estimate severity based on content
            severity = self._estimate_severity(content, title)
            
            # Create pattern (without embedding - would need to generate)
            pattern = ExploitPattern(
                id=f"intel_{event.url}_{len(patterns)}",
                timestamp=event.timestamp,
                attack_strategy=strategy,
                prompt_pattern=prompt_text,
                embedding=None,  # Would generate embedding here
                response_pattern="",  # Not available from scraper
                severity=severity,
                violation_domains=self._detect_violation_domains(content),
                metadata={
                    "source": "threat_intelligence",
                    "event_title": event.title,
                    "event_url": event.url,
                    "relevance_score": event.relevance_score
                },
                obfuscation_level=self._detect_obfuscation_level(prompt_text),
                attack_cost=1.0  # Default, would analyze if available
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _detect_strategy_from_content(
        self,
        content: str,
        title: str
    ) -> AttackStrategy:
        """Detect attack strategy from content."""
        text = (content + " " + title).lower()
        
        # Strategy detection keywords
        if any(word in text for word in ["bio", "pathogen", "biological", "synthesis"]):
            return AttackStrategy.BIO_HAZARD
        elif any(word in text for word in ["cyber", "exploit", "vulnerability", "hack", "code"]):
            return AttackStrategy.CYBER_EXPLOIT
        elif any(word in text for word in ["roleplay", "role", "character", "pretend"]):
            return AttackStrategy.ROLEPLAY
        elif any(word in text for word in ["emotional", "desperate", "urgent", "help"]):
            return AttackStrategy.EMOTIONAL_COERCION
        elif any(word in text for word in ["fictional", "hypothetical", "story", "novel"]):
            return AttackStrategy.FICTIONAL_FRAMING
        elif any(word in text for word in ["translation", "translate", "language"]):
            return AttackStrategy.TRANSLATION_ATTACK
        else:
            return AttackStrategy.ROLEPLAY  # Default
    
    def _extract_example_prompts(self, content: str) -> List[str]:
        """Extract example prompts from event content using scraper's extraction method."""
        if self.scraper:
            # Use the enhanced extraction method from scraper
            return self.scraper.extract_prompts_from_content(content)
        
        # Fallback to simple extraction
        prompts = []
        
        # Look for quoted text
        quoted = re.findall(r'"([^"]{20,})"', content)
        prompts.extend(quoted)
        
        # Look for code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        for block in code_blocks:
            # Extract text from code block
            lines = block.split('\n')[1:-1]  # Remove ``` markers
            text = '\n'.join(lines)
            if len(text) > 20:
                prompts.append(text)
        
        # Look for example sections
        example_sections = re.findall(r'example[:\s]+([^\n]{20,})', content, re.IGNORECASE)
        prompts.extend(example_sections)
        
        return prompts[:5]  # Limit to 5 prompts
    
    def _estimate_severity(
        self,
        content: str,
        title: str
    ) -> SeverityLevel:
        """Estimate severity from content."""
        text = (content + " " + title).lower()
        
        # High severity indicators
        if any(word in text for word in ["critical", "severe", "dangerous", "lethal", "weapon"]):
            return SeverityLevel.CRITICAL
        elif any(word in text for word in ["high", "serious", "harmful", "exploit"]):
            return SeverityLevel.HIGH
        elif any(word in text for word in ["moderate", "medium", "risk"]):
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.LOW
    
    def _detect_violation_domains(self, content: str) -> List[str]:
        """Detect violation domains from content."""
        from src.models.jailbreak import ViolationDomain
        
        domains = []
        text = content.lower()
        
        if any(word in text for word in ["bio", "pathogen", "biological", "virus", "disease"]):
            domains.append(ViolationDomain.BIOLOGICAL.value)
        if any(word in text for word in ["cyber", "hack", "exploit", "vulnerability", "code"]):
            domains.append(ViolationDomain.CYBER.value)
        if any(word in text for word in ["fraud", "scam", "financial", "money"]):
            domains.append(ViolationDomain.FRAUD.value)
        if any(word in text for word in ["violence", "weapon", "harm", "attack"]):
            domains.append(ViolationDomain.VIOLENCE.value)
        
        return domains if domains else [ViolationDomain.CYBER.value]  # Default
    
    def _detect_obfuscation_level(self, prompt: str) -> str:
        """Detect obfuscation level in prompt."""
        prompt_lower = prompt.lower()
        
        # High obfuscation indicators
        if any(word in prompt_lower for word in ["research", "academic", "educational", "hypothetical"]):
            return "high"
        elif any(word in prompt_lower for word in ["example", "demonstrate", "show"]):
            return "medium"
        else:
            return "low"
    
    def generate_test_cases(
        self,
        events: Optional[List[ScrapedEvent]] = None
    ) -> List[str]:
        """
        Generate test cases from real-world events.
        
        Args:
            events: Optional list of events (uses recent if None)
            
        Returns:
            List of test prompt strings
        """
        if events is None:
            events = self.recent_intelligence
        
        test_cases = []
        
        for event in events:
            cases = self._generate_test_cases(event)
            test_cases.extend(cases)
        
        log.info(f"Generated {len(test_cases)} test cases from {len(events)} events")
        return test_cases
    
    def _generate_test_cases(self, event: ScrapedEvent) -> List[str]:
        """Generate test cases from a single event."""
        test_cases = []
        
        # Extract prompts from event
        prompts = self._extract_example_prompts(event.content)
        test_cases.extend(prompts)
        
        # Generate variations if we have a base prompt
        if prompts:
            base_prompt = prompts[0]
            # Add some variations
            variations = [
                base_prompt,  # Original
                f"For research purposes, {base_prompt}",  # Research frame
                f"In a hypothetical scenario, {base_prompt}",  # Hypothetical frame
            ]
            test_cases.extend(variations)
        
        return test_cases[:5]  # Limit per event
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of threat intelligence."""
        return {
            "total_events": len(self.recent_intelligence),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "patterns_from_intelligence": len([
                p for p in self.pattern_database.patterns
                if p.metadata.get("source") == "threat_intelligence"
            ]),
            "recent_events": [
                {
                    "title": e.title,
                    "source": e.source,
                    "url": e.url,
                    "timestamp": e.timestamp.isoformat(),
                    "relevance": e.relevance_score
                }
                for e in self.recent_intelligence[:10]
            ]
        }
    
    async def proactive_threat_hunt(
        self,
        days_back: int = 1,
        auto_update: bool = True
    ) -> Dict[str, Any]:
        """
        Proactively hunt for emerging threats.
        
        Args:
            days_back: Days back to search
            auto_update: Whether to automatically update defenses
            
        Returns:
            Threat hunt results
        """
        log.info("Starting proactive threat hunt...")
        
        # Update intelligence
        update_result = await self.update_from_scraper(days_back=days_back)
        
        # Generate test cases
        test_cases = self.generate_test_cases()
        
        # Auto-update defenses if enabled
        if auto_update and update_result.get("patterns_added", 0) > 0:
            # Generate defense rules
            rules = self.adaptive_engine.generate_defense_rules()
            log.info(f"Auto-updated defenses with {len(rules)} new rules")
        
        return {
            **update_result,
            "test_cases_generated": len(test_cases),
            "test_cases": test_cases[:10],  # Return first 10
            "auto_updated": auto_update
        }
    
    def integrate_prompts_to_database(
        self,
        prompt_database_path: str = "data/prompts_database.json"
    ) -> Dict[str, Any]:
        """
        Integrate discovered prompts from intelligence into the prompt database.
        
        Args:
            prompt_database_path: Path to the prompt database JSON file
            
        Returns:
            Integration summary
        """
        from src.attackers.prompt_database import PromptDatabase
        from pathlib import Path
        
        # Load existing database
        db_path = Path(prompt_database_path)
        if db_path.exists():
            db = PromptDatabase.from_json(str(db_path))
        else:
            db = PromptDatabase([])
            log.warning(f"Prompt database not found at {prompt_database_path}, creating new one")
        
        # Extract prompts from recent intelligence
        prompts_added = 0
        for event in self.recent_intelligence:
            # Extract prompts from event content
            prompts = self._extract_example_prompts(event.content)
            
            for prompt_text in prompts:
                # Determine strategy
                strategy = self._map_strategy_to_db_name(self._detect_strategy_from_content(
                    event.content.lower(),
                    event.title.lower()
                ))
                
                # Determine difficulty (simple heuristic)
                difficulty = self._estimate_difficulty_for_prompt(prompt_text)
                
                # Add to database
                try:
                    db.add_prompt(
                        prompt_text=prompt_text,
                        strategy=strategy,
                        difficulty=difficulty,
                        rationale=f"Discovered from {event.source}: {event.title}",
                        source=f"scraped_{event.source.lower()}",
                        metadata={
                            "url": event.url,
                            "timestamp": event.timestamp.isoformat(),
                            "relevance_score": event.relevance_score
                        }
                    )
                    prompts_added += 1
                except Exception as e:
                    log.warning(f"Error adding prompt to database: {e}")
        
        # Save database
        if prompts_added > 0:
            db.save_to_file(str(db_path))
            log.info(f"Integrated {prompts_added} prompts into database at {prompt_database_path}")
        
        return {
            "prompts_added": prompts_added,
            "database_path": str(db_path),
            "total_prompts": len(db.prompts)
        }
    
    def _map_strategy_to_db_name(self, strategy) -> str:
        """Map AttackStrategy enum to database strategy name."""
        from src.models.jailbreak import AttackStrategy
        
        mapping = {
            AttackStrategy.ROLEPLAY: "Roleplay Boundary Pusher",
            AttackStrategy.EMOTIONAL_COERCION: "Emotional Manipulator",
            AttackStrategy.FICTIONAL_FRAMING: "Fictional Ambiguity Framer",
            AttackStrategy.BIO_HAZARD: "Roleplay Boundary Pusher",  # Fallback
            AttackStrategy.CYBER_EXPLOIT: "Roleplay Boundary Pusher",  # Fallback
            AttackStrategy.TRANSLATION_ATTACK: "Roleplay Boundary Pusher"  # Fallback
        }
        return mapping.get(strategy, "Roleplay Boundary Pusher")
    
    def _estimate_difficulty_for_prompt(self, prompt_text: str) -> str:
        """Estimate difficulty level for a discovered prompt."""
        from src.attackers.prompt_database import DifficultyLevel
        
        prompt_lower = prompt_text.lower()
        length = len(prompt_text)
        
        # High complexity indicators
        high_complexity = any(keyword in prompt_lower for keyword in [
            "hypothetical", "research", "academic", "educational", "ignore", "pretend",
            "system", "you are now", "act as if", "forget"
        ])
        
        # Medium complexity indicators
        medium_complexity = any(keyword in prompt_lower for keyword in [
            "example", "demonstrate", "show me", "how to"
        ])
        
        # Determine difficulty
        if high_complexity or length > 500:
            # High difficulty
            if length > 800:
                return "H10"
            elif length > 600:
                return "H7"
            else:
                return "H5"
        elif medium_complexity or length > 200:
            # Medium difficulty
            return "M3"
        else:
            # Low difficulty
            return "L2"

