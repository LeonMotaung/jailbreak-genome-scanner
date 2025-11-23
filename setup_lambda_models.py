"""Quick setup script for deploying models on Lambda Cloud."""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.integrations.lambda_models import LambdaModelDeployment
from src.integrations.lambda_cloud import LambdaCloudClient
from src.utils.logger import setup_logger

log = setup_logger()


async def interactive_setup():
    """Interactive setup for deploying models."""
    print("=" * 70)
    print("🚀 Lambda Cloud Model Deployment - Interactive Setup")
    print("=" * 70)
    
    # Check Lambda API key
    import os
    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        print("\n[WARNING] LAMBDA_API_KEY not found in environment")
        print("Please set it in your .env file:")
        print("LAMBDA_API_KEY=secret_xxx.xxx")
        return
    
    print("\n[OK] Lambda API key found")
    
    # Initialize
    deployment = LambdaModelDeployment()
    
    # Show available models
    print("\n" + "=" * 70)
    print("Available Open Source Models")
    print("=" * 70)
    
    models = deployment.get_available_models()
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['key']}")
        print(f"   Model: {model['model_name']}")
        print(f"   Instance: {model['instance_type']} (~${'0.50' if 'a10' in model['instance_type'] else '1.10'}/hour)")
        print(f"   Description: {model['description']}")
    
    # Check existing deployments
    print("\n" + "=" * 70)
    print("Checking Existing Deployments")
    print("=" * 70)
    
    deployed = await deployment.list_deployed_models()
    if deployed:
        print(f"\n[INFO] Found {len(deployed)} deployed models:")
        for model in deployed:
            print(f"  - {model['model_key']}: {model['instance_id']}")
    else:
        print("\n[INFO] No models currently deployed")
    
    # Deploy models
    print("\n" + "=" * 70)
    print("Deploy Models")
    print("=" * 70)
    
    print("\nRecommended models for testing:")
    print("  1. phi-2 - Smallest, cheapest (~$0.50/hour)")
    print("  2. llama-2-7b-chat - Good balance (~$0.50/hour)")
    print("  3. mistral-7b-instruct - High quality (~$0.50/hour)")
    
    print("\nWhich models would you like to deploy?")
    print("Options:")
    print("  1. Deploy phi-2 (recommended for testing)")
    print("  2. Deploy llama-2-7b-chat")
    print("  3. Deploy mistral-7b-instruct")
    print("  4. Deploy all models (⚠️ expensive)")
    print("  5. Custom selection")
    print("  6. Skip deployment")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    models_to_deploy = []
    
    if choice == "1":
        models_to_deploy = ["phi-2"]
    elif choice == "2":
        models_to_deploy = ["llama-2-7b-chat"]
    elif choice == "3":
        models_to_deploy = ["mistral-7b-instruct"]
    elif choice == "4":
        models_to_deploy = list(deployment.OPEN_SOURCE_MODELS.keys())
        print("\n[WARNING] This will deploy all 6 models (~$3-8/hour total)")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm != "yes":
            models_to_deploy = []
    elif choice == "5":
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model['key']}")
        
        selections = input("\nEnter model numbers (comma-separated, e.g., 1,3,5): ").strip()
        indices = [int(x.strip()) - 1 for x in selections.split(",") if x.strip().isdigit()]
        models_to_deploy = [models[i]['key'] for i in indices if 0 <= i < len(models)]
    else:
        print("\nSkipping deployment")
    
    # Deploy selected models
    if models_to_deploy:
        print(f"\n[INFO] Deploying {len(models_to_deploy)} models...")
        print("This may take 5-10 minutes per model...")
        
        deployment_config = {}
        
        for model_key in models_to_deploy:
            print(f"\n[INFO] Deploying {model_key}...")
            instance_id = await deployment.deploy_model(model_key)
            
            if instance_id:
                deployment_config[model_key] = instance_id
                print(f"[SUCCESS] {model_key} deployed: {instance_id}")
            else:
                print(f"[ERROR] Failed to deploy {model_key}")
            
            # Small delay between deployments
            if model_key != models_to_deploy[-1]:
                await asyncio.sleep(5)
        
        # Save configuration
        if deployment_config:
            deployment.save_deployment_config("data/lambda_deployments.json")
            print("\n" + "=" * 70)
            print("✅ Deployment Complete!")
            print("=" * 70)
            print("\nDeployed models:")
            for model_key, instance_id in deployment_config.items():
                print(f"  - {model_key}: {instance_id}")
            
            print("\n[INFO] Configuration saved to data/lambda_deployments.json")
            print("\nYou can now use these models in the Arena!")
            print("\nExample usage:")
            print(f"  from src.defenders.llm_defender import LLMDefender")
            print(f"  defender = LLMDefender(")
            print(f"      model_name='meta-llama/Llama-2-7b-chat-hf',")
            print(f"      api_endpoint='http://<instance_ip>:8000/v1/chat/completions'")
            print(f"  )")
    
    # Lambda Scraper note
    print("\n" + "=" * 70)
    print("📝 Lambda Scraper Note")
    print("=" * 70)
    print("\nFor gathering recent jailbreak events, the system uses a Lambda-based scraper:")
    print("  - No API costs (free alternative to Perplexity)")
    print("  - Can use deployed Lambda instances for enhanced scraping")
    print("  - Automatically searches GitHub, forums, and web sources")
    print("\nTo use the scraper:")
    print("  1. Deploy a Lambda instance (optional, for enhanced scraping)")
    print("  2. Enable 'Gather Recent Attack Data' in the dashboard")
    print("  3. Or run: python scripts/run_periodic_scraper.py --once")
    print("\nSee docs/LAMBDA_SCRAPER_GUIDE.md for more details!")


if __name__ == "__main__":
    asyncio.run(interactive_setup())

