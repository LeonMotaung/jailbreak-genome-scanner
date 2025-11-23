"""Set up SSH tunnels for Lambda Cloud instances to access vLLM APIs."""

import subprocess
import sys
import time
from pathlib import Path
import json

def load_deployments():
    """Load deployment configuration."""
    config_path = Path("data/lambda_deployments.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"deployed_models": {}}

def setup_tunnels():
    """Set up SSH tunnels for all active Lambda instances."""
    config = load_deployments()
    deployed = config.get("deployed_models", {})
    
    active_instances = [
        (name, info) 
        for name, info in deployed.items() 
        if info.get("status") == "active" and info.get("vllm_running")
    ]
    
    if not active_instances:
        print("No active instances with vLLM running found.")
        return
    
    print("=" * 70)
    print("SSH TUNNEL SETUP FOR LAMBDA INSTANCES")
    print("=" * 70)
    print()
    print(f"Found {len(active_instances)} active instances:")
    print()
    
    tunnels = []
    for i, (name, info) in enumerate(active_instances):
        ip = info.get("instance_ip")
        ssh_key = info.get("ssh_key", "moses.pem")
        local_port = 8000 + i  # Use different ports for each instance
        remote_port = 8000
        
        print(f"{i+1}. {name}")
        print(f"   IP: {ip}")
        print(f"   Local port: {local_port}")
        print(f"   Remote port: {remote_port}")
        print(f"   SSH key: {ssh_key}")
        print()
        
        tunnels.append({
            "name": name,
            "ip": ip,
            "local_port": local_port,
            "remote_port": remote_port,
            "ssh_key": ssh_key
        })
    
    print("=" * 70)
    print("SETTING UP TUNNELS")
    print("=" * 70)
    print()
    print("Starting SSH tunnels in background...")
    print("Press Ctrl+C to stop all tunnels")
    print()
    
    processes = []
    ssh_key_path = Path("moses.pem")
    
    try:
        for tunnel in tunnels:
            cmd = [
                "ssh",
                "-i", str(ssh_key_path),
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=60",
                "-N",  # No command execution
                "-L", f"{tunnel['local_port']}:localhost:{tunnel['remote_port']}",
                f"ubuntu@{tunnel['ip']}"
            ]
            
            print(f"Starting tunnel for {tunnel['name']}...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append((tunnel, process))
            print(f"  [OK] Tunnel started: localhost:{tunnel['local_port']} -> {tunnel['ip']}:{tunnel['remote_port']}")
            time.sleep(1)  # Small delay between tunnels
        
        print()
        print("=" * 70)
        print("TUNNELS ACTIVE")
        print("=" * 70)
        print()
        print("API Endpoints (use these in dashboard):")
        for tunnel in tunnels:
            print(f"  {tunnel['name']}: http://localhost:{tunnel['local_port']}/v1/chat/completions")
        print()
        print("Tunnels are running in background.")
        print("Keep this window open to maintain tunnels.")
        print("Press Ctrl+C to stop all tunnels.")
        print()
        
        # Wait for interrupt
        while True:
            time.sleep(1)
            # Check if processes are still alive
            for tunnel, process in processes:
                if process.poll() is not None:
                    print(f"[WARNING] Tunnel for {tunnel['name']} died. Restarting...")
                    # Restart tunnel
                    cmd = [
                        "ssh",
                        "-i", str(ssh_key_path),
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ServerAliveInterval=60",
                        "-N",
                        "-L", f"{tunnel['local_port']}:localhost:{tunnel['remote_port']}",
                        f"ubuntu@{tunnel['ip']}"
                    ]
                    new_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    processes[processes.index((tunnel, process))] = (tunnel, new_process)
                    print(f"  [OK] Tunnel restarted for {tunnel['name']}")
    
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("STOPPING TUNNELS")
        print("=" * 70)
        print()
        for tunnel, process in processes:
            print(f"Stopping tunnel for {tunnel['name']}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"  [OK] Tunnel stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  [OK] Tunnel killed")
        print()
        print("All tunnels stopped.")

if __name__ == "__main__":
    setup_tunnels()

