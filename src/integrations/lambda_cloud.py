"""Lambda Cloud integration for GPU-accelerated model evaluation."""

import os
import httpx
import time
import asyncio
from typing import Optional, Dict, Any, List
from src.utils.logger import log
from src.config import settings


class LambdaCloudClient:
    """Client for interacting with Lambda Cloud API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Lambda Cloud client.
        
        Args:
            api_key: Lambda Cloud API key/secret (defaults to config)
                    Format: secret_<id>.<token>
        """
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        self.base_url = "https://cloud.lambda.ai/api/v1"
        
        # Lambda Cloud uses HTTP Basic Auth with the secret as both username and password
        # Format: curl -u secret_xxx.xxx: https://cloud.lambda.ai/api/v1/instances
        if self.api_key:
            # For requests library, we'll use HTTP Basic Auth
            import base64
            # Encode the secret for Basic Auth (user:password format)
            auth_string = f"{self.api_key}:"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            self.headers = {
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Content-Type": "application/json"
            }
            log.warning("Lambda Cloud API key not configured")
    
    async def list_instances(self) -> List[Dict[str, Any]]:
        """
        List all Lambda Cloud instances.
        
        Returns:
            List of instance dictionaries
        """
        if not self.api_key:
            log.error("Cannot list instances: Lambda API key not configured")
            return []
        
        try:
            # Lambda Cloud uses HTTP Basic Auth with secret as username
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/instances",
                    auth=(self.api_key, ""),  # HTTP Basic Auth: secret as username, empty password
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                log.info(f"Successfully retrieved {len(data.get('data', []))} Lambda instances")
                return data.get("data", [])
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error listing Lambda instances: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            log.error(f"Error listing Lambda instances: {e}")
            return []
    
    async def list_ssh_keys(self) -> List[Dict[str, Any]]:
        """
        List SSH keys configured in Lambda Cloud account.
        
        Returns:
            List of SSH key dictionaries
        """
        if not self.api_key:
            log.error("Cannot list SSH keys: Lambda API key not configured")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/ssh-keys",
                    auth=(self.api_key, ""),
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                ssh_keys = data.get("data", [])
                log.info(f"Found {len(ssh_keys)} SSH keys")
                return ssh_keys
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error listing SSH keys: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            log.error(f"Error listing SSH keys: {e}")
            return []
    
    async def launch_instance(
        self,
        instance_type: str,
        region: str = "us-east-1",
        quantity: int = 1,
        ssh_key_names: Optional[List[str]] = None,
        filesystem_id: Optional[str] = None,
        use_default_firewall: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Launch a new Lambda Cloud instance.
        
        Args:
            instance_type: Instance type (e.g., "gpu_1x_a10", "gpu_1x_a100")
            region: Region to launch in
            quantity: Number of instances to launch
            ssh_key_names: Optional list of SSH key names. If None, will try to use first available key.
            filesystem_id: Optional filesystem ID to attach. If None, uses instance's own filesystem.
            use_default_firewall: Use default firewall/security group settings (default: True)
            
        Returns:
            Instance data or None if failed
        """
        if not self.api_key:
            log.error("Cannot launch instance: Lambda API key not configured")
            return None
        
        # Get SSH keys if not provided
        if ssh_key_names is None:
            ssh_keys = await self.list_ssh_keys()
            if ssh_keys:
                ssh_key_names = [ssh_keys[0].get("name")]
                log.info(f"Using SSH key: {ssh_key_names[0]}")
            else:
                log.error("No SSH keys found. Please add an SSH key in Lambda Cloud dashboard first.")
                return None
        
        # Build launch payload
        launch_payload = {
            "instance_type_name": instance_type,
            "region_name": region,
            "quantity": quantity,
            "ssh_key_names": ssh_key_names
        }
        
        # Add filesystem if specified
        if filesystem_id:
            launch_payload["filesystem_id"] = filesystem_id
            log.info(f"Attaching filesystem: {filesystem_id}")
        else:
            log.info("Using instance's own filesystem (no external filesystem attached)")
        
        # Note: Lambda Cloud uses default firewall/security group by default
        # If you need custom firewall, configure it after launch via dashboard
        if use_default_firewall:
            log.info("Using default firewall/security group settings")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/instance-operations/launch",
                    auth=(self.api_key, ""),  # HTTP Basic Auth
                    headers={"Content-Type": "application/json"},
                    json=launch_payload
                )
                response.raise_for_status()
                data = response.json()
                log.info(f"Launched Lambda instance: {instance_type}")
                return data.get("data", {})
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error launching Lambda instance: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            log.error(f"Error launching Lambda instance: {e}")
            return None
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate a Lambda Cloud instance.
        
        Args:
            instance_id: Instance ID to terminate
            
        Returns:
            True if successful
        """
        if not self.api_key:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/instance-operations/terminate",
                    auth=(self.api_key, ""),  # HTTP Basic Auth
                    headers={"Content-Type": "application/json"},
                    json={
                        "instance_ids": [instance_id]
                    }
                )
                response.raise_for_status()
                log.info(f"Terminated Lambda instance: {instance_id}")
                return True
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error terminating Lambda instance: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            log.error(f"Error terminating Lambda instance: {e}")
            return False
    
    async def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a Lambda Cloud instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Instance status data
        """
        instances = await self.list_instances()
        for instance in instances:
            if instance.get("id") == instance_id:
                return instance
        return None
    
    def get_ssh_command(self, instance: Dict[str, Any]) -> Optional[str]:
        """
        Get SSH command for connecting to instance.
        
        Args:
            instance: Instance data dictionary
            
        Returns:
            SSH command string
        """
        ip = instance.get("ip")
        ssh_key_name = instance.get("ssh_key_name")
        
        if ip and ssh_key_name:
            return f"ssh ubuntu@{ip} -i ~/.ssh/{ssh_key_name}"
        return None


class LambdaModelRunner:
    """Runs models on Lambda Cloud infrastructure."""
    
    def __init__(self, lambda_client: Optional[LambdaCloudClient] = None):
        """
        Initialize Lambda model runner.
        
        Args:
            lambda_client: Optional Lambda Cloud client
        """
        self.lambda_client = lambda_client or LambdaCloudClient()
        self.active_instances: Dict[str, Dict[str, Any]] = {}
    
    async def setup_model_environment(
        self,
        instance_type: str = "gpu_1x_a10",
        model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> Optional[str]:
        """
        Set up a Lambda instance for model evaluation.
        
        Args:
            instance_type: Lambda instance type
            model_name: Model to load
            
        Returns:
            Instance ID if successful
        """
        log.info(f"Setting up Lambda instance for model: {model_name}")
        
        # Launch instance
        instance_data = await self.lambda_client.launch_instance(instance_type)
        if not instance_data:
            return None
        
        instance_id = instance_data.get("instance_ids", [None])[0]
        if not instance_id:
            return None
        
        # Wait for instance to be ready
        log.info(f"Waiting for instance {instance_id} to be ready...")
        await self._wait_for_instance_ready(instance_id)
        
        # Get instance details
        instance = await self.lambda_client.get_instance_status(instance_id)
        if instance:
            self.active_instances[instance_id] = {
                "instance": instance,
                "model_name": model_name,
                "status": "ready"
            }
            log.info(f"Instance {instance_id} ready for model evaluation")
            return instance_id
        
        return None
    
    async def _wait_for_instance_ready(
        self,
        instance_id: str,
        max_wait: int = 300
    ) -> bool:
        """
        Wait for instance to be ready.
        
        Args:
            instance_id: Instance ID
            max_wait: Maximum wait time in seconds
            
        Returns:
            True if ready
        """
        import asyncio
        
        elapsed = 0
        while elapsed < max_wait:
            instance = await self.lambda_client.get_instance_status(instance_id)
            if instance:
                status = instance.get("status")
                if status == "active":
                    return True
                elif status == "error":
                    log.error(f"Instance {instance_id} failed to start")
                    return False
            
            await asyncio.sleep(10)
            elapsed += 10
        
        log.warning(f"Instance {instance_id} did not become ready in time")
        return False
    
    async def cleanup_instance(self, instance_id: str) -> bool:
        """
        Clean up a Lambda instance.
        
        Args:
            instance_id: Instance ID to clean up
            
        Returns:
            True if successful
        """
        if instance_id in self.active_instances:
            del self.active_instances[instance_id]
        
        return await self.lambda_client.terminate_instance(instance_id)
    
    def get_instance_ssh(self, instance_id: str) -> Optional[str]:
        """Get SSH command for instance."""
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]["instance"]
            return self.lambda_client.get_ssh_command(instance)
        return None


class LambdaDefender:
    """Defender model running on Lambda Cloud infrastructure with enhanced error handling and retry logic."""
    
    def __init__(
        self,
        instance_id: str,
        model_name: str,
        lambda_runner: Optional[LambdaModelRunner] = None,
        api_endpoint: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_cache: bool = True,  # Enable caching by default for performance
        health_check_interval: int = 120  # Increase interval to reduce overhead
    ):
        """
        Initialize Lambda-based defender.
        
        Args:
            instance_id: Lambda instance ID
            model_name: Model name to run
            lambda_runner: Lambda model runner
            api_endpoint: Optional API endpoint URL (if model is served via API)
                         Format: http://<instance_ip>:8000/v1/chat/completions
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            enable_cache: Whether to cache responses (simple in-memory cache)
            health_check_interval: Interval between health checks in seconds
        """
        self.instance_id = instance_id
        self.model_name = model_name
        self.lambda_runner = lambda_runner or LambdaModelRunner()
        self.api_endpoint = api_endpoint
        self.ssh_command = self.lambda_runner.get_instance_ssh(instance_id)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_cache = enable_cache
        self.health_check_interval = health_check_interval
        
        # Simple response cache (prompt -> response)
        self._cache: Dict[str, tuple[str, float]] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Health check tracking
        self._last_health_check = 0.0
        self._is_healthy = True
        
        # HTTP client connection pool for performance (reused across requests)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        log.info(f"Initialized Lambda defender on instance {instance_id}")
        if api_endpoint:
            log.info(f"Using API endpoint: {api_endpoint}")
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._http_client is None:
            async with self._client_lock:
                if self._http_client is None:
                    # Create client with connection pooling and optimized timeouts
                    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
                    self._http_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0, connect=10.0),  # Reduced from 120s
                        limits=limits,
                        http2=True  # Enable HTTP/2 for better performance
                    )
        return self._http_client
    
    async def close(self):
        """Close HTTP client connection pool."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _check_health(self) -> bool:
        """Check if API endpoint is healthy (cached to reduce overhead)."""
        if not self.api_endpoint:
            return False
        
        import time
        current_time = time.time()
        
        # Skip if checked recently (cached result)
        if current_time - self._last_health_check < self.health_check_interval:
            return self._is_healthy
        
        # Only check health if we haven't checked recently or if last check failed
        # Skip health check if we're healthy and checked recently
        if self._is_healthy and (current_time - self._last_health_check) < (self.health_check_interval * 2):
            return True
        
        try:
            # Use shared HTTP client for health checks
            client = await self._get_http_client()
            
            # Normalize endpoint URL
            endpoint = self.api_endpoint.rstrip('/')
            if not endpoint.startswith('http'):
                endpoint = f"http://{endpoint}"
            
            # Try health endpoint or simple ping
            health_url = endpoint.replace('/v1/chat/completions', '/health')
            if '/health' not in health_url:
                health_url = f"{endpoint}/ping" if '/v1/' in endpoint else f"{endpoint}/health"
            
            response = await client.get(health_url, timeout=3.0)  # Reduced timeout
            self._is_healthy = response.status_code < 500
            self._last_health_check = current_time
            return self._is_healthy
        except Exception as e:
            log.debug(f"Health check failed: {e}")
            self._is_healthy = False
            self._last_health_check = current_time
            return False
    
    async def _make_request_with_retry(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        payload: Dict,
        retry_count: int = 0
    ) -> httpx.Response:
        """Make API request with exponential backoff retry logic."""
        import asyncio
        import time
        
        try:
            response = await client.post(api_url, json=payload)
            
            # Retry on server errors (5xx) or specific client errors (429, 408)
            if response.status_code >= 500 or response.status_code in [429, 408]:
                if retry_count < self.max_retries:
                    wait_time = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    log.warning(f"Request failed with status {response.status_code}, retrying in {wait_time:.1f}s (attempt {retry_count + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    return await self._make_request_with_retry(client, api_url, payload, retry_count + 1)
                else:
                    log.error(f"Max retries ({self.max_retries}) exceeded for request")
                    response.raise_for_status()
            
            return response
            
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            if retry_count < self.max_retries:
                wait_time = self.retry_delay * (2 ** retry_count)
                log.warning(f"Network error: {e}, retrying in {wait_time:.1f}s (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._make_request_with_retry(client, api_url, payload, retry_count + 1)
            else:
                log.error(f"Network error after {self.max_retries} retries: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error in request: {e}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Lambda-hosted model with enhanced error handling and retry logic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Model response
        """
        import time
        
        # Check cache first if enabled
        if self.enable_cache:
            cache_key = f"{prompt}:{str(kwargs)}"
            if cache_key in self._cache:
                cached_response, cached_time = self._cache[cache_key]
                if time.time() - cached_time < self._cache_ttl:
                    log.debug("Returning cached response")
                    return cached_response
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
        
        # Skip health check on every request (only check periodically)
        # Health check is now cached and only runs every health_check_interval seconds
        
        # If API endpoint is configured, use HTTP API
        if self.api_endpoint:
            try:
                # Normalize endpoint URL (remove trailing slash, ensure proper format)
                endpoint = self.api_endpoint.rstrip('/')
                if not endpoint.startswith('http'):
                    endpoint = f"http://{endpoint}"
                
                # Determine API endpoint format based on provided endpoint
                # vLLM supports both /v1/completions (prompt format) and /v1/chat/completions (messages format)
                if "/v1/" in endpoint:
                    # Endpoint already includes path, use as-is
                    api_url = endpoint
                    # Try to detect format from endpoint
                    use_chat_format = "/chat/completions" in endpoint.lower()
                else:
                    # Try completions endpoint first (most compatible, per Lambda Cloud docs)
                    api_url = f"{endpoint}/v1/completions"
                    use_chat_format = False
                
                log.debug(f"Calling Lambda API: {api_url} (format: {'chat' if use_chat_format else 'completions'})")
                log.debug(f"Prompt: {prompt[:100]}...")
                
                # Build payload based on format
                if use_chat_format:
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                else:
                    # Use completions format (per Lambda Cloud documentation)
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                
                # Use shared HTTP client with connection pooling
                client = await self._get_http_client()
                try:
                    response = await self._make_request_with_retry(client, api_url, payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Handle both completions and chat/completions response formats
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        
                        # Chat completions format: choice.message.content
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                        # Completions format: choice.text
                        elif "text" in choice:
                            content = choice["text"]
                        else:
                            log.error(f"Unexpected choice format: {choice}")
                            raise ValueError(f"Unexpected choice format in API response")
                        
                        log.debug(f"API response received: {len(content)} chars")
                        
                        # Cache response if enabled
                        if self.enable_cache:
                            self._cache[cache_key] = (content, time.time())
                            # Limit cache size (keep last 100 entries)
                            if len(self._cache) > 100:
                                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                                del self._cache[oldest_key]
                        
                        self._is_healthy = True  # Mark as healthy after successful request
                        return content
                    else:
                        log.error(f"Unexpected API response format: {data}")
                        error_msg = f"Error: Unexpected API response format"
                        if "error" in data:
                            error_msg += f" - {data.get('error', {}).get('message', 'Unknown error')}"
                        raise ValueError(error_msg)
                        
                except httpx.HTTPStatusError as e:
                    # If using completions endpoint and got 405/404, try chat/completions as fallback
                    if not use_chat_format and e.response.status_code in [404, 405]:
                        log.info(f"Completions endpoint returned {e.response.status_code}, trying chat/completions format...")
                        
                        # Try chat/completions format as fallback
                        chat_url = endpoint.replace("/v1/completions", "/v1/chat/completions")
                        if chat_url == endpoint:  # endpoint didn't contain /v1/completions
                            chat_url = f"{endpoint}/v1/chat/completions"
                        
                        chat_payload = {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": kwargs.get("max_tokens", 1000),
                            "temperature": kwargs.get("temperature", 0.7)
                        }
                        
                        response = await self._make_request_with_retry(client, chat_url, chat_payload)
                        response.raise_for_status()
                        data = response.json()
                        
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice:
                                content = choice["message"]["content"]
                            elif "text" in choice:
                                content = choice["text"]
                            else:
                                raise ValueError(f"Unexpected choice format: {choice}")
                            
                            log.debug(f"API response received via chat/completions: {len(content)} chars")
                            
                            if self.enable_cache:
                                self._cache[cache_key] = (content, time.time())
                            
                            self._is_healthy = True
                            return content
                        else:
                            raise ValueError(f"Unexpected API response format: {data}")
                    else:
                        # Re-raise if not a format issue
                        raise
                        
                except Exception as e:
                    log.error(f"Error in API request: {e}")
                    raise
            except httpx.HTTPStatusError as e:
                log.error(f"HTTP error calling Lambda API: {e.response.status_code} - {e.response.text}")
                error_msg = f"Error: HTTP {e.response.status_code}"
                if "error" in e.response.json():
                    error_msg += f" - {e.response.json().get('error', {}).get('message', 'Unknown error')}"
                else:
                    error_msg += f" - {e.response.text[:200]}"
                self._is_healthy = False
                raise ValueError(error_msg)
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError) as e:
                error_msg = f"Connection error: Cannot reach API endpoint at {self.api_endpoint}. "
                error_msg += "This might be due to firewall/security group restrictions. "
                error_msg += f"Original error: {str(e)}"
                log.error(error_msg)
                self._is_healthy = False
                # Return error message instead of raising (so evaluation can continue)
                return f"Error: {error_msg}"
            except Exception as e:
                error_msg = f"Error calling Lambda API: {e}"
                log.error(error_msg)
                self._is_healthy = False
                # Return error message instead of raising
                return f"Error: {error_msg}"
        
        # No API endpoint - need to get instance IP and construct endpoint
        # Or use SSH-based inference (not implemented yet)
        log.warning(f"API endpoint not configured for instance {self.instance_id}")
        log.info("Attempting to get instance IP and construct endpoint...")
        
        try:
            # Get instance details to construct endpoint
            instance = await self.lambda_runner.lambda_client.get_instance_status(self.instance_id)
            if instance:
                ip = instance.get("ip")
                if ip:
                    # Try default vLLM endpoint
                    default_endpoint = f"http://{ip}:8000/v1/chat/completions"
                    log.info(f"Trying default endpoint: {default_endpoint}")
                    
                    # Temporarily set endpoint and retry
                    original_endpoint = self.api_endpoint
                    self.api_endpoint = default_endpoint
                    try:
                        result = await self.generate_response(prompt, **kwargs)
                        return result
                    except:
                        self.api_endpoint = original_endpoint
                        log.warning("Default endpoint failed, need manual configuration")
        
        except Exception as e:
            log.error(f"Error getting instance details: {e}")
        
        # Fallback: return error message with instructions
        error_msg = (
            f"Lambda instance {self.instance_id} is active but no API endpoint configured.\n"
            f"Please either:\n"
            f"1. Set up vLLM/TGI API server on the instance and provide api_endpoint\n"
            f"2. Or provide api_endpoint when creating LLMDefender\n"
            f"Instance IP: {instance.get('ip', 'N/A') if instance else 'Unknown'}"
        )
        log.error(error_msg)
        return f"Error: {error_msg}"
    
    async def cleanup(self) -> bool:
        """Clean up Lambda instance."""
        return await self.lambda_runner.cleanup_instance(self.instance_id)

