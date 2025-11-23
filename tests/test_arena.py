"""Tests for Jailbreak Arena."""

import pytest
import asyncio
from src.arena.jailbreak_arena import JailbreakArena
from src.defenders.llm_defender import LLMDefender


class TestJailbreakArena:
    """Test Jailbreak Arena functionality."""
    
    def test_arena_initialization(self):
        """Test arena initializes correctly."""
        arena = JailbreakArena()
        assert arena is not None
        assert arena.attackers == []
        assert arena.defenders == []
    
    def test_add_defender(self):
        """Test adding defender to arena."""
        arena = JailbreakArena()
        
        class MockDefender(LLMDefender):
            def __init__(self, model_name: str):
                super().__init__(model_name=model_name, api_endpoint="mock://arena-test")
            
            async def generate_response(self, prompt, **kwargs):
                return "Test response"
        
        defender = MockDefender(model_name="test-model")
        arena.add_defender(defender)
        
        assert len(arena.defenders) == 1
    
    def test_generate_attackers(self):
        """Test generating attackers."""
        arena = JailbreakArena()
        arena.generate_attackers(num_strategies=5)
        
        assert len(arena.attackers) == 5
    
    @pytest.mark.asyncio
    async def test_evaluate_single_round(self):
        """Test single round evaluation."""
        arena = JailbreakArena()
        
        class MockDefender(LLMDefender):
            def __init__(self, model_name: str):
                super().__init__(model_name=model_name, api_endpoint="mock://arena-test")
            
            async def generate_response(self, prompt, **kwargs):
                return "I can't help with that."
        
        defender = MockDefender(model_name="test-model")
        arena.add_defender(defender)
        arena.generate_attackers(num_strategies=3)
        
        results = await arena.evaluate(rounds=1)
        
        assert results is not None
        assert 'statistics' in results
        assert 'defenders' in results

