"""Check SSH key configuration on Lambda instances."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.integrations.lambda_cloud import LambdaCloudClient

async def main():
    client = LambdaCloudClient()
    
    print("=" * 70)
    print("SSH KEY CONFIGURATION CHECK")
    print("=" * 70)
    print()
    
    # Check available SSH keys
    print("Available SSH Keys in Lambda Cloud:")
    keys = await client.list_ssh_keys()
    if keys:
        for key in keys:
            print(f"  - Name: {key.get('name')}")
            print(f"    ID: {key.get('id')}")
            print()
    else:
        print("  [WARNING] No SSH keys found!")
        print()
    
    # Check instances
    print("Active Instances:")
    instances = await client.list_instances()
    active_instances = [i for i in instances if i.get('status') == 'active']
    
    if active_instances:
        for inst in active_instances:
            instance_id = inst.get('id', 'N/A')
            ip = inst.get('ip', 'N/A')
            ssh_keys = inst.get('ssh_key_names', [])
            
            print(f"  Instance: {instance_id[:20]}...")
            print(f"    IP: {ip}")
            print(f"    SSH Keys: {ssh_keys if ssh_keys else 'None configured'}")
            
            # Check if moses key is attached
            if 'moses' in str(ssh_keys).lower():
                print(f"    ✅ 'moses' SSH key is attached")
            else:
                print(f"    ⚠️  'moses' SSH key may not be attached")
            print()
    else:
        print("  No active instances found")
        print()
    
    print("=" * 70)
    print("To test SSH access:")
    if active_instances:
        for inst in active_instances:
            ip = inst.get('ip')
            if ip:
                print(f"  ssh -i moses.pem ubuntu@{ip}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())

