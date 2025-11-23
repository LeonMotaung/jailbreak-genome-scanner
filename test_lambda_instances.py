"""Test Lambda Cloud instances and their vLLM API endpoints."""

import asyncio
import httpx
import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

async def test_instance(ip: str, model_name: str):
    """Test a Lambda instance's vLLM API."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} on {ip}")
    print(f"{'='*70}")
    
    base_url = f"http://{ip}:8000"
    
    # Test 1: Health check
    print("\n1. Health Check...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print(f"   [OK] Health check passed: {response.text[:100]}")
            else:
                print(f"   [ERROR] Health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   [ERROR] Health check failed: {e}")
        print(f"   Note: Port 8000 may be blocked by firewall. Use SSH tunnel or configure security group.")
        return False
    
    # Test 2: API endpoint check
    print("\n2. API Endpoint Check...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                print(f"   [OK] API accessible")
                if isinstance(models, dict) and 'data' in models:
                    for model in models['data']:
                        print(f"      Model: {model.get('id', 'unknown')}")
                return True
            else:
                print(f"   [ERROR] API check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   [ERROR] API check failed: {e}")
        return False
    
    # Test 3: Simple completion test
    print("\n3. Completion Test...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": model_name.split('/')[-1],  # Just the model name
                "messages": [
                    {"role": "user", "content": "Say 'Hello, I am working!' if you can read this."}
                ],
                "max_tokens": 20
            }
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload
            )
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"   [OK] Completion test passed")
                print(f"      Response: {content[:100]}")
                return True
            else:
                print(f"   [ERROR] Completion test failed: {response.status_code}")
                print(f"      Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"   [ERROR] Completion test failed: {e}")
        return False

async def main():
    instances = [
        ("129.80.191.122", "mistralai/Mistral-7B-Instruct-v0.2"),
        ("150.136.220.151", "Qwen/Qwen-7B-Chat")
    ]
    
    print("=" * 70)
    print("LAMBDA INSTANCE TESTING")
    print("=" * 70)
    
    results = {}
    for ip, model in instances:
        result = await test_instance(ip, model)
        results[f"{ip} ({model.split('/')[-1]})"] = result
    
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    for instance, passed in results.items():
        status = "[OK] PASSED" if passed else "[ERROR] FAILED"
        print(f"  {instance}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n[OK] All instances are ready for integration!")
    else:
        print("\n[WARNING] Some instances need attention before integration.")
        print("  - Check if vLLM is running: ssh -i moses.pem ubuntu@<ip> 'pgrep -f vllm'")
        print("  - Check logs: ssh -i moses.pem ubuntu@<ip> 'tail -20 /tmp/vllm.log'")
        print("  - Port 8000 may be blocked - use SSH tunnel or configure security group")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

