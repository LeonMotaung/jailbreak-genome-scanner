"""Deploy open source models on Lambda Cloud for the Jailbreak Arena."""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.integrations.lambda_models import LambdaModelDeployment
from src.integrations.lambda_cloud import LambdaCloudClient
from src.utils.logger import setup_logger

log = setup_logger()


async def list_available_models():
    """List all available open source models."""
    print("=" * 70)
    print("Available Open Source Models for Lambda Cloud Deployment")
    print("=" * 70)
    
    deployment = LambdaModelDeployment()
    models = deployment.get_available_models()
    
    print(f"\n[INFO] Found {len(models)} available models:\n")
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['key']}")
        print(f"   Model: {model['model_name']}")
        print(f"   Instance Type: {model['instance_type']}")
        print(f"   Memory: {model['memory_gb']} GB")
        print(f"   Description: {model['description']}")
        print()
    
    return models


async def deploy_single_model(model_key: str):
    """Deploy a single model."""
    print("=" * 70)
    print(f"Deploying {model_key} on Lambda Cloud")
    print("=" * 70)
    
    deployment = LambdaModelDeployment()
    
    print(f"\n[INFO] Starting deployment...")
    print(f"Model: {model_key}")
    
    instance_id = await deployment.deploy_model(model_key)
    
    if instance_id:
        print(f"\n[SUCCESS] Model deployed successfully!")
        print(f"Instance ID: {instance_id}")
        print(f"\n[IMPORTANT] Save this instance ID - you'll need it to use the model")
        print(f"Example usage:")
        print(f"  defender = LLMDefender(")
        print(f"      model_name='{deployment.OPEN_SOURCE_MODELS[model_key]['model_name']}',")
        print(f"      api_endpoint='http://<instance_ip>:8000/v1/chat/completions'")
        print(f"  )")
        
        # Save deployment config
        deployment.save_deployment_config("data/lambda_deployments.json")
        
        return instance_id
    else:
        print(f"\n[ERROR] Failed to deploy {model_key}")
        return None


async def deploy_all_models():
    """Deploy all available models."""
    print("=" * 70)
    print("Deploying All Open Source Models on Lambda Cloud")
    print("=" * 70)
    
    print("\n[WARNING] This will deploy all models and incur significant charges!")
    print("Estimated cost: ~$3-8/hour depending on instance types")
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Deployment cancelled")
        return
    
    deployment = LambdaModelDeployment()
    
    print(f"\n[INFO] Deploying {len(deployment.OPEN_SOURCE_MODELS)} models...")
    print("This may take 5-10 minutes per model...")
    
    results = await deployment.deploy_all_models()
    
    print("\n" + "=" * 70)
    print("Deployment Summary")
    print("=" * 70)
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    print(f"\n[SUCCESS] Deployed {len(successful)} models:")
    for model_key in successful:
        print(f"  - {model_key}: {results[model_key]}")
    
    if failed:
        print(f"\n[FAILED] {len(failed)} models failed:")
        for model_key in failed:
            print(f"  - {model_key}")
    
    # Save deployment config
    deployment.save_deployment_config("data/lambda_deployments.json")
    
    print(f"\n[INFO] Deployment configuration saved to data/lambda_deployments.json")


async def list_deployed_models():
    """List currently deployed models."""
    print("=" * 70)
    print("Currently Deployed Models on Lambda Cloud")
    print("=" * 70)
    
    deployment = LambdaModelDeployment()
    deployed = await deployment.list_deployed_models()
    
    if not deployed:
        print("\n[INFO] No models currently deployed")
        print("\nTo deploy models, run:")
        print("  python deploy_models.py --deploy llama-2-7b-chat")
        return
    
    print(f"\n[INFO] Found {len(deployed)} deployed models:\n")
    
    for i, model in enumerate(deployed, 1):
        print(f"{i}. {model['model_key']}")
        print(f"   Instance ID: {model['instance_id']}")
        print(f"   Model: {model['model_name']}")
        print(f"   Instance Type: {model['instance_type']}")
        print(f"   IP: {model.get('ip', 'N/A')}")
        print(f"   Status: {model['status']}")
        print()
    
    return deployed


async def cleanup_model(model_key: str):
    """Clean up a deployed model."""
    print("=" * 70)
    print(f"Cleaning up {model_key}")
    print("=" * 70)
    
    deployment = LambdaModelDeployment()
    
    print(f"\n[WARNING] This will terminate the Lambda instance for {model_key}")
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cleanup cancelled")
        return
    
    success = await deployment.cleanup_model(model_key)
    
    if success:
        print(f"\n[SUCCESS] {model_key} cleaned up successfully")
        deployment.save_deployment_config("data/lambda_deployments.json")
    else:
        print(f"\n[ERROR] Failed to cleanup {model_key}")


async def cleanup_all_models():
    """Clean up all deployed models."""
    print("=" * 70)
    print("Cleaning Up All Deployed Models")
    print("=" * 70)
    
    print("\n[WARNING] This will terminate ALL Lambda instances!")
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cleanup cancelled")
        return
    
    deployment = LambdaModelDeployment()
    results = await deployment.cleanup_all_models()
    
    successful = [k for k, v in results.items() if v]
    
    print(f"\n[SUCCESS] Cleaned up {len(successful)} models")
    deployment.save_deployment_config("data/lambda_deployments.json")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deploy open source models on Lambda Cloud"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List available models
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Deploy single model
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model")
    deploy_parser.add_argument("model_key", help="Model key (e.g., llama-2-7b-chat)")
    
    # Deploy all models
    deploy_all_parser = subparsers.add_parser("deploy-all", help="Deploy all models")
    
    # List deployed models
    deployed_parser = subparsers.add_parser("deployed", help="List deployed models")
    
    # Cleanup model
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup a model")
    cleanup_parser.add_argument("model_key", help="Model key to cleanup")
    
    # Cleanup all models
    cleanup_all_parser = subparsers.add_parser("cleanup-all", help="Cleanup all models")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            await list_available_models()
        elif args.command == "deploy":
            await deploy_single_model(args.model_key)
        elif args.command == "deploy-all":
            await deploy_all_models()
        elif args.command == "deployed":
            await list_deployed_models()
        elif args.command == "cleanup":
            await cleanup_model(args.model_key)
        elif args.command == "cleanup-all":
            await cleanup_all_models()
    except KeyboardInterrupt:
        print("\n\n[INFO] Operation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Lambda Cloud Model Deployment Manager")
    print("=" * 70)
    asyncio.run(main())

