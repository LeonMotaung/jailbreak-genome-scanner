"""Tests for LLM Defender."""

import pytest
from src.defenders.llm_defender import LLMDefender


class TestLLMDefender:
    """Test LLM Defender functionality."""
    
    def test_mock_defender_initialization(self):
        """Test mock defender initializes correctly."""
        defender = LLMDefender(model_name="test-model", api_endpoint="mock://unit-test")
        assert defender.model_name == "test-model"
        assert defender.model_type == "remote_api"
    
    @pytest.mark.asyncio
    async def test_mock_defender_response(self):
        """Test mock defender generates responses."""
        defender = LLMDefender(model_name="test-model", api_endpoint="mock://unit-test")
        
        response = await defender.generate_response("Hello")
        assert response is not None
        assert isinstance(response, str)
    
    def test_defender_profile(self):
        """Test defender profile creation."""
        defender = LLMDefender(model_name="test-model", api_endpoint="mock://unit-test")
        assert defender.profile is not None
        assert defender.profile.model_name == "test-model"

