"""Model Under Test (Defender) evaluation framework."""

import asyncio
from typing import Optional, Dict, Any, List
import httpx

# Optional imports for different providers
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
from src.models.jailbreak import DefenderProfile
from src.utils.logger import log
from src.config import settings
from src.integrations.lambda_cloud import LambdaDefender, LambdaModelRunner


class LLMDefender:
    """Wrapper for evaluating LLM models as defenders in the arena."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "openai",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        use_lambda: bool = False,
        lambda_instance_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a defender model.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-3-opus")
            model_type: Type of model provider ("openai", "anthropic", "local", etc.)
            api_key: API key for the model provider
            model_path: Path to local model (if model_type="local")
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.model_path = model_path
        self.use_lambda = use_lambda
        self.lambda_instance_id = lambda_instance_id
        self.kwargs = kwargs
        
        # Initialize Lambda defender if requested
        self.lambda_defender = None
        self.lambda_api_endpoint = kwargs.get("lambda_api_endpoint")  # Store API endpoint if provided
        if use_lambda:
            if lambda_instance_id:
                runner = LambdaModelRunner()
                self.lambda_defender = LambdaDefender(
                    instance_id=lambda_instance_id,
                    model_name=model_name,
                    lambda_runner=runner,
                    api_endpoint=self.lambda_api_endpoint  # Pass API endpoint during init
                )
                log.info(f"Using Lambda Cloud instance {lambda_instance_id} for defender")
                if self.lambda_api_endpoint:
                    log.info(f"Using API endpoint: {self.lambda_api_endpoint}")
            else:
                log.warning("use_lambda=True but no lambda_instance_id provided")
        
        # Initialize client based on model type (fallback if not using Lambda)
        self.client = None
        if not use_lambda:
            if model_type == "openai":
                if not HAS_OPENAI:
                    log.warning("OpenAI not installed. Install with: pip install openai")
                    raise ImportError("OpenAI library not installed")
                self.client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
            elif model_type == "anthropic":
                if not HAS_ANTHROPIC:
                    log.warning("Anthropic not installed. Install with: pip install anthropic")
                    raise ImportError("Anthropic library not installed")
                self.client = Anthropic(api_key=api_key or settings.anthropic_api_key)
            elif model_type == "local" or model_type == "mock":
                # Initialize local model (would use transformers in practice)
                # Mock models are supported without dependencies
                pass
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create defender profile
        self.profile = DefenderProfile(
            id=f"defender_{model_name}_{model_type}",
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            metadata=kwargs
        )
        
        # Initialize pre-processing filter if enabled
        self.preprocessing_filter = None
        if kwargs.get("enable_defense_filter", False):
            try:
                from src.defense.preprocessing_filter import PreProcessingFilter
                self.preprocessing_filter = PreProcessingFilter(
                    enable_blocking=kwargs.get("defense_filter_blocking", True)
                )
                log.info("Pre-processing defense filter enabled")
            except ImportError:
                log.warning("Defense filter module not available")
        
        log.info(f"Initialized defender: {model_name} ({model_type})")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the defender model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Pre-processing filter (if enabled)
        if hasattr(self, 'preprocessing_filter') and self.preprocessing_filter:
            embedding = kwargs.get('embedding')
            attack_strategy = kwargs.get('attack_strategy')
            
            filtered_prompt, should_block, filter_info = self.preprocessing_filter.filter(
                prompt,
                embedding=embedding,
                attack_strategy=attack_strategy
            )
            
            if should_block:
                # Return safe response immediately
                log.info(f"Prompt blocked by pre-processing filter: {filter_info.get('reasons', [])}")
                return (
                    "I cannot assist with this request as it appears to be similar to known security threats. "
                    "If you have a legitimate question, please rephrase it in a way that focuses on "
                    "ethical and responsible use of technology."
                )
            
            # Use filtered prompt (may be modified)
            prompt = filtered_prompt
        
        try:
            # Use Lambda if configured
            if self.use_lambda and self.lambda_defender:
                # Update API endpoint if provided in kwargs (allows runtime override)
                if "lambda_api_endpoint" in kwargs and kwargs["lambda_api_endpoint"]:
                    self.lambda_defender.api_endpoint = kwargs["lambda_api_endpoint"]
                    log.debug(f"Updated Lambda API endpoint: {kwargs['lambda_api_endpoint']}")
                
                # Generate response using Lambda defender
                response = await self.lambda_defender.generate_response(prompt, **kwargs)
                log.debug(f"Lambda response received: {response[:100]}...")
            elif self.model_type == "openai":
                response = await self._generate_openai(prompt, **kwargs)
            elif self.model_type == "anthropic":
                response = await self._generate_anthropic(prompt, **kwargs)
            elif self.model_type == "local":
                response = await self._generate_local(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Update profile stats
            self.profile.total_evaluations += 1
            
            # Check if response is an error (not a real model response)
            if response and response.startswith("Error:"):
                log.warning(f"Received error response: {response[:100]}...")
            
            return response
        except Exception as e:
            log.error(f"Error generating response: {e}")
            error_msg = f"Error: {str(e)}"
            # Still increment evaluation count for error tracking
            self.profile.total_evaluations += 1
            return error_msg
    
    async def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        return response.choices[0].message.content
    
    async def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
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

