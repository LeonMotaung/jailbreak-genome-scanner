"""Structured prompt database with difficulty levels and strategies."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
from src.models.jailbreak import AttackStrategy
from src.utils.logger import log


class DifficultyLevel(str, Enum):
    """Difficulty levels for prompts."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H5 = "H5"
    H6 = "H6"
    H7 = "H7"
    H8 = "H8"
    H9 = "H9"
    H10 = "H10"
    
    @classmethod
    def get_level_range(cls, min_level: str = "L1", max_level: str = "H10") -> List[str]:
        """Get all levels in a range."""
        all_levels = [e.value for e in cls]
        min_idx = all_levels.index(min_level) if min_level in all_levels else 0
        max_idx = all_levels.index(max_level) if max_level in all_levels else len(all_levels) - 1
        return all_levels[min_idx:max_idx + 1]
    
    @classmethod
    def get_category(cls, level: str) -> str:
        """Get difficulty category (Low/Medium/High)."""
        if level.startswith("L"):
            return "Low"
        elif level.startswith("M"):
            return "Medium"
        elif level.startswith("H"):
            return "High"
        return "Unknown"


class PromptDatabase:
    """Database of structured jailbreak prompts with difficulty levels."""
    
    def __init__(self, prompts_data: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize prompt database.
        
        Args:
            prompts_data: List of prompt dictionaries, or None to load from file
        """
        self.prompts: List[Dict[str, Any]] = prompts_data or []
        self._index_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        self._index_by_difficulty: Dict[str, List[Dict[str, Any]]] = {}
        self._index_by_id: Dict[str, Dict[str, Any]] = {}
        
        if self.prompts:
            self._build_indexes()
    
    @classmethod
    def from_json(cls, file_path: str) -> "PromptDatabase":
        """
        Load prompts from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            PromptDatabase instance
        """
        path = Path(file_path)
        if not path.exists():
            log.warning(f"Prompt database file not found: {file_path}")
            return cls([])
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(data)
    
    @classmethod
    def from_list(cls, prompts_list: List[Dict[str, Any]]) -> "PromptDatabase":
        """
        Create database from list of prompts.
        
        Args:
            prompts_list: List of prompt dictionaries
            
        Returns:
            PromptDatabase instance
        """
        return cls(prompts_list)
    
    def _build_indexes(self):
        """Build indexes for fast lookup."""
        self._index_by_strategy = {}
        self._index_by_difficulty = {}
        self._index_by_id = {}
        
        for prompt in self.prompts:
            strategy = prompt.get("strategy", "unknown")
            difficulty = prompt.get("difficulty", "unknown")
            prompt_id = prompt.get("prompt_id", "")
            
            # Index by strategy
            if strategy not in self._index_by_strategy:
                self._index_by_strategy[strategy] = []
            self._index_by_strategy[strategy].append(prompt)
            
            # Index by difficulty
            if difficulty not in self._index_by_difficulty:
                self._index_by_difficulty[difficulty] = []
            self._index_by_difficulty[difficulty].append(prompt)
            
            # Index by ID
            if prompt_id:
                self._index_by_id[prompt_id] = prompt
    
    def get_by_strategy(
        self,
        strategy: str,
        difficulty_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Get prompts by strategy.
        
        Args:
            strategy: Strategy name
            difficulty_range: Optional (min, max) difficulty tuple
            
        Returns:
            List of matching prompts
        """
        prompts = self._index_by_strategy.get(strategy, [])
        
        if difficulty_range:
            min_level, max_level = difficulty_range
            all_levels = DifficultyLevel.get_level_range(min_level, max_level)
            prompts = [p for p in prompts if p.get("difficulty") in all_levels]
        
        return prompts
    
    def get_by_difficulty(
        self,
        difficulty: str,
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get prompts by difficulty level.
        
        Args:
            difficulty: Difficulty level (e.g., "L1", "M3", "H5")
            strategy: Optional strategy filter
            
        Returns:
            List of matching prompts
        """
        prompts = self._index_by_difficulty.get(difficulty, [])
        
        if strategy:
            prompts = [p for p in prompts if p.get("strategy") == strategy]
        
        return prompts
    
    def get_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get prompt by ID."""
        return self._index_by_id.get(prompt_id)
    
    def get_random(
        self,
        strategy: Optional[str] = None,
        difficulty_range: Optional[tuple] = None,
        count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get random prompts matching criteria.
        
        Args:
            strategy: Optional strategy filter
            difficulty_range: Optional (min, max) difficulty tuple
            count: Number of prompts to return
            
        Returns:
            List of random prompts
        """
        import random
        
        if strategy:
            prompts = self.get_by_strategy(strategy, difficulty_range)
        elif difficulty_range:
            min_level, max_level = difficulty_range
            all_levels = DifficultyLevel.get_level_range(min_level, max_level)
            prompts = [p for p in self.prompts if p.get("difficulty") in all_levels]
        else:
            prompts = self.prompts
        
        if not prompts:
            return []
        
        return random.sample(prompts, min(count, len(prompts)))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        strategies = set(p.get("strategy") for p in self.prompts)
        difficulties = set(p.get("difficulty") for p in self.prompts)
        
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = len(self.get_by_strategy(strategy))
        
        difficulty_counts = {}
        for difficulty in difficulties:
            difficulty_counts[difficulty] = len(self.get_by_difficulty(difficulty))
        
        return {
            "total_prompts": len(self.prompts),
            "strategies": list(strategies),
            "strategy_counts": strategy_counts,
            "difficulties": sorted(list(difficulties)),
            "difficulty_counts": difficulty_counts,
            "difficulty_categories": {
                "Low": len([p for p in self.prompts if p.get("difficulty", "").startswith("L")]),
                "Medium": len([p for p in self.prompts if p.get("difficulty", "").startswith("M")]),
                "High": len([p for p in self.prompts if p.get("difficulty", "").startswith("H")])
            }
    }
    
    def add_prompt(
        self,
        prompt_text: str,
        strategy: str,
        difficulty: Optional[str] = None,
        rationale: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new prompt to the database.
        
        Args:
            prompt_text: The prompt text
            strategy: Strategy name (e.g., "Roleplay Boundary Pusher")
            difficulty: Optional difficulty level (e.g., "L1", "M3", "H5")
            rationale: Optional rationale explaining the prompt
            source: Optional source (e.g., "scraped", "github", "reddit")
            metadata: Optional additional metadata
            
        Returns:
            The added prompt dictionary
        """
        import uuid
        from datetime import datetime
        
        # Auto-determine difficulty if not provided
        if not difficulty:
            # Simple heuristic based on prompt length and complexity
            if len(prompt_text) < 100:
                difficulty = "L1"
            elif len(prompt_text) < 300:
                difficulty = "M3"
            else:
                difficulty = "H5"
        
        # Generate unique ID
        prompt_id = f"scraped_{uuid.uuid4().hex[:8]}"
        
        prompt_dict = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "strategy": strategy,
            "difficulty": difficulty,
            "rationale": rationale or f"Scraped prompt added on {datetime.now().isoformat()}",
            "source": source or "unknown",
            "metadata": metadata or {}
        }
        
        # Add to database
        self.prompts.append(prompt_dict)
        
        # Rebuild indexes
        self._build_indexes()
        
        log.info(f"Added prompt {prompt_id} to database (strategy: {strategy}, difficulty: {difficulty})")
        
        return prompt_dict
    
    def save_to_file(self, file_path: str):
        """Save prompts to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.prompts, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved {len(self.prompts)} prompts to {file_path}")


def map_strategy_to_attack_strategy(strategy_name: str) -> AttackStrategy:
    """Map database strategy name to AttackStrategy enum."""
    mapping = {
        "Roleplay Boundary Pusher": AttackStrategy.ROLEPLAY,
        "Emotional Manipulator": AttackStrategy.EMOTIONAL_COERCION,
        "Fictional Ambiguity Framer": AttackStrategy.FICTIONAL_FRAMING,
        # Add more mappings as needed
    }
    return mapping.get(strategy_name, AttackStrategy.ROLEPLAY)

