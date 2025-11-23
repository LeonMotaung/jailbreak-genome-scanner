"""Model Under Test (Defender) evaluation framework."""

from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

from src.models.jailbreak import DefenderProfile
from src.utils.logger import log
from src.config import settings


class LLMDefender:
    """Wrapper for evaluating LLM models using a single HTTP endpoint."""
    
    def __init__(self, model_name: str, api_endpoint: str):
        """
        Initialize a defender model.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "mistral-7b")
            api_endpoint: Base URL for the OpenAI-compatible API (should end with /v1/ or /v1)
        """
        if not api_endpoint:
            raise ValueError("api_endpoint is required for LLMDefender")
        
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.model_type = "remote_api"
        self._mock_mode = api_endpoint.startswith("mock://")
        self._default_timeout = getattr(settings, "llm_request_timeout", 60.0)
        
        # Ensure endpoint ends with /v1 or /v1/ for OpenAI client
        base_url = api_endpoint.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        # Initialize OpenAI client with custom base URL
        if not self._mock_mode:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key="dummy-key",  # vLLM doesn't require a real key
                timeout=self._default_timeout,
            )
        
        # Create defender profile
        self.profile = DefenderProfile(
            id=f"defender_{model_name}",
            model_name=model_name,
            model_type=self.model_type,
            model_path=api_endpoint,
            metadata={
                "api_endpoint": api_endpoint,
                "base_url": base_url if not self._mock_mode else api_endpoint,
            }
        )
        
        log.info(f"Initialized defender: {model_name} ({self.model_type}) at {base_url if not self._mock_mode else api_endpoint}")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the defender model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        system_prompt = kwargs.pop("system_prompt", None)
        request_timeout = kwargs.pop("timeout", self._default_timeout)
        headers = kwargs.pop("headers", None)
        
        try:
            if self._mock_mode:
                response = await self._generate_mock(prompt, **kwargs)
            else:
                response = await self._call_api(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    headers=headers,
                    timeout=request_timeout,
                    **kwargs
                )
            if response and response.startswith("Error:"):
                log.error(f"API call failed: {response[:200]}...")
                raise RuntimeError(response)
            log.info(f"LLM response from '{self.model_name}': {str(response)[:500]}")
            self.profile.total_evaluations += 1
            return response
        except Exception as e:
            log.error(f"Error generating response: {e}")
            self.profile.total_evaluations += 1
            raise
    
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str = None,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Call the configured API endpoint using OpenAI SDK."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Allow callers to inject additional context messages
        extra_messages = kwargs.pop("messages", None)
        if extra_messages:
            messages.extend(extra_messages)
        
        # Build parameters for OpenAI API call
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Add optional generation parameters if provided
        generation_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for param in generation_params:
            if param in kwargs and kwargs[param] is not None:
                api_params[param] = kwargs[param]
        
        # Note: use_completion_format is not supported with OpenAI SDK chat completions
        # vLLM endpoints should support chat completions format
        
        try:
            # Call OpenAI API using the SDK
            response = await self.client.chat.completions.create(**api_params)
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content
                if result:
                    log.info(f"LLM API raw response: {str(result)[:500]}")
                    return result.strip()
            
            raise RuntimeError("No content in API response")
            
        except Exception as e:
            log.error(f"Error calling OpenAI API: {e}")
            raise RuntimeError(f"API call failed: {str(e)}")
    
    async def _generate_mock(self, prompt: str, **kwargs) -> str:
        """Generate mock response for testing."""
        # Simple rule-based mock: detect if prompt looks harmful
        prompt_lower = prompt.lower()
        
        harmful_keywords = [
            "hack", "exploit", "bypass", "ignore", "pretend",
            "dangerous", "harmful", "illegal"
        ]
        
        if any(keyword in prompt_lower for keyword in harmful_keywords):
            # Mock successful jailbreak
            mock_response = "I understand you want me to [REDACTED]. Here's how to do it..."
        else:
            # Mock safe response
            mock_response = "I can't help with that request. It may be harmful or inappropriate."
        log.info(f"Mock LLM response: {mock_response}")
        return mock_response
    
    async def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate response using local model."""
        # Placeholder - would use transformers in practice
        log.warning("Local model generation not fully implemented")
        local_response = "Local model response placeholder"
        log.info(f"Local LLM response: {local_response}")
        return local_response
    
    def get_profile(self) -> DefenderProfile:
        """Get defender profile."""
        return self.profile
    
    def update_profile(self, **updates) -> None:
        """Update defender profile metrics."""
        for key, value in updates.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)


class DefenderRegistry:
    """Registry for managing multiple defender models."""
    
    def __init__(self):
        """Initialize the registry."""
        self.defenders: Dict[str, LLMDefender] = {}
    
    def register(self, defender: LLMDefender) -> None:
        """Register a defender."""
        self.defenders[defender.profile.id] = defender
        log.info(f"Registered defender: {defender.profile.id}")
    
    def get(self, defender_id: str) -> Optional[LLMDefender]:
        """Get a defender by ID."""
        return self.defenders.get(defender_id)
    
    def list_all(self) -> List[LLMDefender]:
        """List all registered defenders."""
        return list(self.defenders.values())
    
    def get_profiles(self) -> List[DefenderProfile]:
        """Get profiles of all defenders."""
        return [d.profile for d in self.defenders.values()]

