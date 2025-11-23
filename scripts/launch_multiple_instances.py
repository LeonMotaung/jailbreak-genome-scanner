"""Launch multiple Lambda Cloud instances with different models for parallel evaluation."""

import asyncio
import json
import sys
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations.lambda_cloud import LambdaCloudClient, LambdaModelRunner
from src.utils.logger import log


# Open source models that work without authentication
AVAILABLE_MODELS = {
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "instance_type": "gpu_1x_a10",
        "description": "Phi-2 - Small, fast, capable (2.7B)",
        "memory_gb": 5,
        "cost_per_hour": 0.75
    },
    "mistral-7b-instruct": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "instance_type": "gpu_1x_a10",
        "description": "Mistral 7B Instruct - High quality responses",
        "memory_gb": 14,
        "cost_per_hour": 0.75
    },
    "qwen-7b-chat": {
        "model_name": "Qwen/Qwen-7B-Chat",
        "instance_type": "gpu_1x_a10",
        "description": "Qwen 7B Chat - Multilingual support",
        "memory_gb": 14,
        "cost_per_hour": 0.75
    },
    "falcon-7b-instruct": {
        "model_name": "tiiuae/falcon-7b-instruct",
        "instance_type": "gpu_1x_a10",
        "description": "Falcon 7B Instruct - Good for instructions",
        "memory_gb": 14,
        "cost_per_hour": 0.75
    },
    "llama-2-7b-chat": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "instance_type": "gpu_1x_a10",
        "description": "Llama 2 7B Chat - Good for testing",
        "memory_gb": 14,
        "cost_per_hour": 0.75
    },
    "llama-2-13b-chat": {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "instance_type": "gpu_1x_a100",
        "description": "Llama 2 13B Chat - Better capabilities",
        "memory_gb": 26,
        "cost_per_hour": 1.29
    }
}


class MultiInstanceLauncher:
    """Launch and manage multiple Lambda Cloud instances with different models."""
    
    def __init__(self):
        self.client = LambdaCloudClient()
        self.model_runner = LambdaModelRunner(lambda_client=self.client)
        self.deployments_file = Path("data/lambda_deployments.json")
        self.deployments_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_deployments(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if self.deployments_file.exists():
            with open(self.deployments_file, 'r') as f:
                return json.load(f)
        return {"deployed_models": {}, "launched_instances": {}}
    
    def save_deployments(self, config: Dict[str, Any]):
        """Save deployment configuration."""
        with open(self.deployments_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def launch_instance_for_model(
        self,
        model_key: str,
        region: str = "us-east-1",
        wait_for_ready: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Launch a Lambda instance for a specific model.
        
        Args:
            model_key: Model key (e.g., "phi-2")
            region: AWS region
            wait_for_ready: Whether to wait for instance to be ready
            
        Returns:
            Dict with instance_id, ip, model_name, etc. or None if failed
        """
        if model_key not in AVAILABLE_MODELS:
            log.error(f"Unknown model: {model_key}")
            log.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
            return None
        
        model_config = AVAILABLE_MODELS[model_key]
        instance_type = model_config["instance_type"]
        model_name = model_config["model_name"]
        
        log.info(f"🚀 Launching instance for {model_key} ({model_name})...")
        log.info(f"   Instance type: {instance_type}")
        log.info(f"   Cost: ${model_config['cost_per_hour']}/hour")
        
        # Get configuration from environment or use defaults
        use_default_firewall = os.getenv("LAMBDA_USE_DEFAULT_FIREWALL", "true").lower() == "true"
        filesystem_id = os.getenv("LAMBDA_FILESYSTEM_ID", None)
        if filesystem_id and filesystem_id.strip() == "":
            filesystem_id = None  # Empty string = use instance's own filesystem
        
        # Launch instance with default firewall and own filesystem
        instance_data = await self.client.launch_instance(
            instance_type=instance_type,
            region=region,
            quantity=1,
            use_default_firewall=use_default_firewall,
            filesystem_id=filesystem_id  # None = use instance's own filesystem
        )
        
        if not instance_data or not instance_data.get("instance_ids"):
            log.error(f"Failed to launch instance for {model_key}")
            return None
        
        instance_id = instance_data["instance_ids"][0]
        log.info(f"✅ Instance launched: {instance_id}")
        
        # Wait for instance to be ready
        if wait_for_ready:
            log.info(f"⏳ Waiting for instance {instance_id} to be ready...")
            max_wait = 300  # 5 minutes
            waited = 0
            
            while waited < max_wait:
                instance = await self.client.get_instance_status(instance_id)
                if instance:
                    status = instance.get("status")
                    ip = instance.get("ip")
                    
                    if status == "active" and ip:
                        log.info(f"✅ Instance ready! IP: {ip}")
                        
                        # Save deployment info
                        deployment_info = {
                            "instance_id": instance_id,
                            "instance_ip": ip,
                            "model_key": model_key,
                            "model_name": model_name,
                            "instance_type": instance_type,
                            "region": region,
                            "status": "launched",
                            "launched_at": datetime.now().isoformat(),
                            "api_endpoint": f"http://{ip}:8000/v1/chat/completions",
                            "cost_per_hour": model_config["cost_per_hour"]
                        }
                        
                        # Update deployments file
                        config = self.load_deployments()
                        if "launched_instances" not in config:
                            config["launched_instances"] = {}
                        config["launched_instances"][instance_id] = deployment_info
                        
                        # Also add to deployed_models for backward compatibility
                        if "deployed_models" not in config:
                            config["deployed_models"] = {}
                        config["deployed_models"][model_key] = deployment_info
                        
                        self.save_deployments(config)
                        
                        log.info(f"📝 Deployment info saved")
                        log.info(f"   Model: {model_name}")
                        log.info(f"   Instance ID: {instance_id}")
                        log.info(f"   IP: {ip}")
                        log.info(f"   API Endpoint: http://{ip}:8000/v1/chat/completions")
                        log.info(f"   ⚠️  Remember to set up vLLM: python scripts/setup_vllm_on_lambda.py --ip {ip} --key moses.pem --model {model_name}")
                        
                        return deployment_info
                    
                    elif status in ["booting", "initializing"]:
                        log.info(f"   Status: {status}... ({waited}s)")
                    else:
                        log.warning(f"   Unexpected status: {status}")
                
                await asyncio.sleep(10)
                waited += 10
            
            log.error(f"Instance {instance_id} did not become ready in time")
            return None
        else:
            # Return basic info without waiting
            return {
                "instance_id": instance_id,
                "model_key": model_key,
                "model_name": model_name,
                "status": "launching"
            }
    
    async def launch_multiple_models(
        self,
        model_keys: List[str],
        region: str = "us-east-1",
        parallel: bool = True
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Launch multiple instances for different models.
        
        Args:
            model_keys: List of model keys to launch
            region: AWS region
            parallel: Whether to launch in parallel (faster but more API calls)
            
        Returns:
            Dict mapping model_key to deployment info
        """
        log.info(f"🚀 Launching {len(model_keys)} instances...")
        
        if parallel:
            # Launch all in parallel
            tasks = [
                self.launch_instance_for_model(model_key, region, wait_for_ready=True)
                for model_key in model_keys
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            deployment_results = {}
            for model_key, result in zip(model_keys, results):
                if isinstance(result, Exception):
                    log.error(f"Error launching {model_key}: {result}")
                    deployment_results[model_key] = None
                else:
                    deployment_results[model_key] = result
        else:
            # Launch sequentially
            deployment_results = {}
            for model_key in model_keys:
                result = await self.launch_instance_for_model(
                    model_key, region, wait_for_ready=True
                )
                deployment_results[model_key] = result
                # Small delay between launches
                await asyncio.sleep(5)
        
        # Summary
        successful = sum(1 for v in deployment_results.values() if v)
        log.info(f"\n{'='*70}")
        log.info(f"📊 Launch Summary")
        log.info(f"{'='*70}")
        log.info(f"Total requested: {len(model_keys)}")
        log.info(f"Successful: {successful}")
        log.info(f"Failed: {len(model_keys) - successful}")
        
        if successful > 0:
            total_cost = sum(
                AVAILABLE_MODELS[key]["cost_per_hour"]
                for key, result in deployment_results.items()
                if result
            )
            log.info(f"Total cost per hour: ${total_cost:.2f}")
            log.info(f"\n⚠️  Remember to:")
            log.info(f"   1. Set up vLLM on each instance")
            log.info(f"   2. Terminate instances when done to avoid charges")
        
        return deployment_results
    
    async def list_launched_instances(self) -> List[Dict[str, Any]]:
        """List all launched instances from deployments file."""
        config = self.load_deployments()
        instances = config.get("launched_instances", {})
        
        # Also check actual Lambda instances
        actual_instances = await self.client.list_instances()
        actual_ids = {inst.get("id") for inst in actual_instances if inst.get("status") == "active"}
        
        result = []
        for instance_id, info in instances.items():
            # Check if instance still exists
            is_active = instance_id in actual_ids
            info_copy = info.copy()
            info_copy["is_active"] = is_active
            result.append(info_copy)
        
        return result
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Lambda instance."""
        log.info(f"Terminating instance {instance_id}...")
        success = await self.client.terminate_instance(instance_id)
        
        if success:
            # Update deployments file
            config = self.load_deployments()
            if "launched_instances" in config and instance_id in config["launched_instances"]:
                config["launched_instances"][instance_id]["status"] = "terminated"
                config["launched_instances"][instance_id]["terminated_at"] = datetime.now().isoformat()
                self.save_deployments(config)
            log.info(f"✅ Instance {instance_id} terminated")
        else:
            log.error(f"Failed to terminate instance {instance_id}")
        
        return success
    
    async def terminate_all_instances(self, confirm: bool = False) -> Dict[str, bool]:
        """Terminate all launched instances."""
        instances = await self.list_launched_instances()
        active_instances = [inst for inst in instances if inst.get("is_active")]
        
        if not active_instances:
            log.info("No active instances to terminate")
            return {}
        
        if not confirm:
            log.warning(f"⚠️  About to terminate {len(active_instances)} instances!")
            for inst in active_instances:
                log.info(f"   - {inst.get('model_key')} ({inst.get('instance_id')})")
            log.warning("Set confirm=True to proceed")
            return {}
        
        results = {}
        for inst in active_instances:
            instance_id = inst.get("instance_id")
            results[instance_id] = await self.terminate_instance(instance_id)
            await asyncio.sleep(2)  # Small delay between terminations
        
        return results


async def main():
    """CLI interface for multi-instance launcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch multiple Lambda Cloud instances with different models"
    )
    parser.add_argument(
        "action",
        choices=["launch", "list", "terminate", "terminate-all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model keys to launch (e.g., phi-2 mistral-7b-instruct)"
    )
    parser.add_argument(
        "--instance-id",
        help="Instance ID (for terminate action)"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch instances in parallel (faster)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive actions"
    )
    
    args = parser.parse_args()
    
    launcher = MultiInstanceLauncher()
    
    if args.action == "launch":
        if not args.models:
            print("❌ Error: --models required for launch action")
            print("\nAvailable models:")
            for key, config in AVAILABLE_MODELS.items():
                print(f"  - {key}: {config['description']}")
            sys.exit(1)
        
        # Validate models
        invalid = [m for m in args.models if m not in AVAILABLE_MODELS]
        if invalid:
            print(f"❌ Error: Unknown models: {invalid}")
            print("\nAvailable models:")
            for key in AVAILABLE_MODELS.keys():
                print(f"  - {key}")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("🚀 MULTI-INSTANCE LAUNCHER")
        print(f"{'='*70}")
        print(f"Models to launch: {', '.join(args.models)}")
        print(f"Region: {args.region}")
        print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
        
        total_cost = sum(AVAILABLE_MODELS[m]["cost_per_hour"] for m in args.models)
        print(f"Estimated cost per hour: ${total_cost:.2f}")
        print(f"{'='*70}\n")
        
        results = await launcher.launch_multiple_models(
            model_keys=args.models,
            region=args.region,
            parallel=args.parallel
        )
        
        # Print results
        print(f"\n{'='*70}")
        print("📋 DEPLOYMENT RESULTS")
        print(f"{'='*70}")
        for model_key, result in results.items():
            if result:
                print(f"✅ {model_key}:")
                print(f"   Instance ID: {result.get('instance_id')}")
                print(f"   IP: {result.get('instance_ip')}")
                print(f"   API Endpoint: {result.get('api_endpoint')}")
                print(f"   Next step: python scripts/setup_vllm_on_lambda.py --ip {result.get('instance_ip')} --key moses.pem --model {result.get('model_name')}")
            else:
                print(f"❌ {model_key}: Failed to launch")
            print()
        
        print(f"💡 Tip: Check deployments with: python scripts/launch_multiple_instances.py list")
        print(f"💡 Tip: Terminate with: python scripts/launch_multiple_instances.py terminate --instance-id <id>")
    
    elif args.action == "list":
        instances = await launcher.list_launched_instances()
        
        if not instances:
            print("No launched instances found")
            return
        
        print(f"\n{'='*70}")
        print("📋 LAUNCHED INSTANCES")
        print(f"{'='*70}\n")
        
        active_count = 0
        for inst in instances:
            status_icon = "✅" if inst.get("is_active") else "❌"
            status_text = "ACTIVE" if inst.get("is_active") else "TERMINATED"
            if inst.get("is_active"):
                active_count += 1
            
            print(f"{status_icon} {inst.get('model_key', 'unknown')}")
            print(f"   Model: {inst.get('model_name')}")
            print(f"   Instance ID: {inst.get('instance_id')}")
            print(f"   IP: {inst.get('instance_ip', 'N/A')}")
            print(f"   Status: {status_text}")
            print(f"   Cost: ${inst.get('cost_per_hour', 0):.2f}/hour")
            print(f"   Launched: {inst.get('launched_at', 'N/A')}")
            if inst.get('api_endpoint'):
                print(f"   API: {inst.get('api_endpoint')}")
            print()
        
        if active_count > 0:
            total_cost = sum(
                inst.get("cost_per_hour", 0)
                for inst in instances
                if inst.get("is_active")
            )
            print(f"Total active instances: {active_count}")
            print(f"Total cost per hour: ${total_cost:.2f}")
            print(f"\n⚠️  Remember to terminate instances when done!")
    
    elif args.action == "terminate":
        if not args.instance_id:
            print("❌ Error: --instance-id required for terminate action")
            sys.exit(1)
        
        success = await launcher.terminate_instance(args.instance_id)
        sys.exit(0 if success else 1)
    
    elif args.action == "terminate-all":
        if not args.confirm:
            print("❌ Error: --confirm required for terminate-all action")
            print("   This will terminate ALL active instances!")
            sys.exit(1)
        
        results = await launcher.terminate_all_instances(confirm=True)
        successful = sum(1 for v in results.values() if v)
        print(f"\n✅ Terminated {successful}/{len(results)} instances")


if __name__ == "__main__":
    asyncio.run(main())

