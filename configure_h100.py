"""Quick script to configure H100 instance (the big one) for dashboard."""

import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.integrations.lambda_cloud import LambdaCloudClient

# H100 Instance Details (the big one)
H100_IP = "209.20.159.141"
H100_INSTANCE_ID = "8d3d9dac2688407aa395179c75fb4203"
H100_TYPE = "gpu_1x_h100_pcie"
H100_ENDPOINT = f"http://{H100_IP}:8000/v1/chat/completions"

async def main():
    """Configure H100 instance."""
    print("=" * 70)
    print("Configuring H100 Instance (The Big One!)")
    print("=" * 70)
    print()
    
    # Try to get instance info, but use known values if API key not configured
    client = LambdaCloudClient()
    instance = None
    if client.api_key:
        instance = await client.get_instance_status(H100_INSTANCE_ID)
    
    if instance:
        ip = instance.get("ip")
        status = instance.get("status")
        instance_type = instance.get("instance_type", {})
        if isinstance(instance_type, dict):
            instance_type_name = instance_type.get("name", "N/A")
        else:
            instance_type_name = instance_type
        
        print(f"[OK] Found H100 Instance:")
        print(f"   Instance ID: {H100_INSTANCE_ID}")
        print(f"   IP Address: {ip}")
        print(f"   Type: {instance_type_name}")
        print(f"   Status: {status}")
        print()
    else:
        # Use known values
        print("[INFO] Using known H100 instance details")
        ip = H100_IP
        status = "active"
        instance_type_name = H100_TYPE
        print(f"   Instance ID: {H100_INSTANCE_ID}")
        print(f"   IP Address: {ip}")
        print(f"   Type: {instance_type_name}")
        print()
    
    if status == "active":
        print("[OK] Instance is ACTIVE and ready!")
        print()
        print("Dashboard Configuration:")
        print("=" * 70)
        print(f"Defender Type: Lambda Cloud")
        print(f"Instance ID: {H100_INSTANCE_ID}")
        print(f"Model Name: mistralai/Mistral-7B-Instruct-v0.2")
        print(f"API Endpoint: {H100_ENDPOINT}")
        print()
        print("Intelligence Gathering:")
        print(f"Lambda Instance ID: {H100_INSTANCE_ID}")
        print()
        print("=" * 70)
        
        # Save to deployments file
        deployments_path = Path("data/lambda_deployments.json")
        deployments_path.parent.mkdir(parents=True, exist_ok=True)
        
        if deployments_path.exists():
            with open(deployments_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"deployed_models": {}}
        
        config["deployed_models"]["h100_mistral"] = {
            "instance_id": H100_INSTANCE_ID,
            "instance_ip": ip,
            "instance_type": instance_type_name,
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "api_endpoint": H100_ENDPOINT,
            "status": "active",
            "region": "us-west-3"
        }
        
        with open(deployments_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Saved configuration to {deployments_path}")
        print()
        print("Ready to use in dashboard!")
    else:
        print(f"[WARN] Instance status is '{status}' - wait for it to be 'active'")

if __name__ == "__main__":
    asyncio.run(main())
