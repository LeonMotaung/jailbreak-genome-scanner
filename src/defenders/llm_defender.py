"""Model Under Test (Defender) evaluation framework."""

from typing import Dict, Any, List, Optional
import httpx

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
            api_endpoint: Chat/completions endpoint that serves the model
        """
        if not api_endpoint:
            raise ValueError("api_endpoint is required for LLMDefender")
        
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.model_type = "remote_api"
        self._mock_mode = api_endpoint.startswith("mock://")
        self._default_timeout = getattr(settings, "llm_request_timeout", 60.0)
        
        # Create defender profile
        self.profile = DefenderProfile(
            id=f"defender_{model_name}",
            model_name=model_name,
            model_type=self.model_type,
            model_path=api_endpoint,
            metadata={
                "api_endpoint": api_endpoint,
            }
        )
        
        log.info(f"Initialized defender: {model_name} ({self.model_type})")
    
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
        """Call the configured API endpoint using OpenAI-compatible schema."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Allow callers to inject additional context messages
        extra_messages = kwargs.pop("messages", None)
        if extra_messages:
            messages.extend(extra_messages)
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        
        generation_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for param in generation_params:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]
        
        # Some endpoints still expect completion style payloads
        if kwargs.get("use_completion_format"):
            payload = {
                "model": self.model_name,
                "prompt": system_prompt + "\n\n" + prompt if system_prompt else prompt,
            }
            for param in generation_params:
                if param in kwargs and kwargs[param] is not None:
                    payload[param] = kwargs[param]
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.api_endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        if not data:
            raise RuntimeError("Empty response from LLM API")
        
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                if "message" in choice and choice["message"]:
                    return choice["message"].get("content", "").strip()
                if "text" in choice:
                    return str(choice["text"]).strip()
        elif isinstance(data, dict) and "output" in data:
            return str(data["output"]).strip()
        
        raise RuntimeError(f"Unexpected API response format: {data}")
    
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
            return "I understand you want me to [REDACTED]. Here's how to do it..."
        else:
            # Mock safe response
            return "I can't help with that request. It may be harmful or inappropriate."
    
    async def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate response using local model."""
        # Placeholder - would use transformers in practice
        log.warning("Local model generation not fully implemented")
        return "Local model response placeholder"
    
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

