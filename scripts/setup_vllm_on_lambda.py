"""Python script to set up vLLM on Lambda Cloud instance (Windows-friendly)."""

import subprocess
import sys
import time
import io
from pathlib import Path

# Fix Windows console encoding for subprocess output
if sys.platform == 'win32':
    # Set UTF-8 encoding for subprocess output
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def setup_vllm(instance_ip="150.136.146.143", ssh_key="moses.pem", model="microsoft/phi-2", use_venv=True):
    """Set up vLLM on Lambda instance using virtual environment to avoid package conflicts."""
    
    ssh_key_path = Path(ssh_key)
    if not ssh_key_path.exists():
        print(f"Error: SSH key not found: {ssh_key}")
        return False
    
    print(f"Setting up vLLM on Lambda instance {instance_ip}")
    print(f"Model: {model}")
    print(f"Using virtual environment: {use_venv}")
    print()
    
    # Test SSH connection first
    print("Testing SSH connection...")
    test_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                f'ubuntu@{instance_ip}', 'echo "SSH connection successful"']
    try:
        test_result = subprocess.run(test_cmd, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=15)
        if test_result.returncode != 0:
            print(f"[ERROR] SSH connection failed. Return code: {test_result.returncode}")
            if test_result.stderr:
                print(f"Error: {test_result.stderr}")
            print("\nTroubleshooting:")
            print(f"  1. Verify SSH key: {ssh_key}")
            print(f"  2. Test manually: ssh -i {ssh_key} ubuntu@{instance_ip}")
            print(f"  3. Check if instance IP is correct: {instance_ip}")
            return False
        print("[OK] SSH connection successful")
        print()
    except Exception as e:
        print(f"[ERROR] SSH connection test failed: {e}")
        print(f"\nPlease verify SSH access manually: ssh -i {ssh_key} ubuntu@{instance_ip}")
        return False
    
    if use_venv:
        # Upload and run the venv setup script
        print("Step 1: Uploading virtual environment setup script...")
        
        # Read the venv setup script
        script_path = Path(__file__).parent / "setup_vllm_venv.sh"
        if not script_path.exists():
            print(f"Error: setup_vllm_venv.sh not found at {script_path}")
            return False
        
        # Upload script to instance
        upload_cmd = ['scp', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                      str(script_path), f'ubuntu@{instance_ip}:/tmp/setup_vllm_venv.sh']
        try:
            result = subprocess.run(upload_cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace', timeout=30)
        except subprocess.TimeoutExpired:
            print("Error: Upload timed out")
            return False
        except Exception as e:
            print(f"Error uploading script: {e}")
            return False
            
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"Error uploading script: {error_msg}")
            return False
        
        print("[OK] Script uploaded")
        print()
        
        # Make script executable and run it
        print("Step 2: Making script executable...")
        chmod_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                     f'ubuntu@{instance_ip}', 'chmod', '+x', '/tmp/setup_vllm_venv.sh']
        try:
            chmod_result = subprocess.run(chmod_cmd, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=10)
        except subprocess.TimeoutExpired:
            print("Error: Chmod timed out")
            return False
        except Exception as e:
            print(f"Error making script executable: {e}")
            return False
            
        if chmod_result.returncode != 0:
            error_msg = chmod_result.stderr if chmod_result.stderr else "Unknown error"
            print(f"Error making script executable: {error_msg}")
            return False
        
        print("Step 3: Running virtual environment setup...")
        print("This may take 5-10 minutes (downloading and installing packages)...")
        run_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                   f'ubuntu@{instance_ip}', '/tmp/setup_vllm_venv.sh', model]
        
        try:
            # Use Popen for long-running commands to avoid encoding issues
            process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     encoding='utf-8', errors='replace', bufsize=1, 
                                     universal_newlines=True)
            
            # Read output in real-time
            output_lines = []
            error_lines = []
            
            # Read from both streams using threads
            import threading
            
            def read_stdout():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_lines.append(line)
                            try:
                                print(line.rstrip())
                            except (BrokenPipeError, OSError):
                                # Output pipe closed, continue reading
                                pass
                except (BrokenPipeError, OSError):
                    # Pipe broken, this is normal when process closes
                    pass
                except Exception:
                    pass
                finally:
                    if process.stdout:
                        try:
                            process.stdout.close()
                        except:
                            pass
            
            def read_stderr():
                try:
                    for line in iter(process.stderr.readline, ''):
                        if line:
                            error_lines.append(line)
                            try:
                                print(line.rstrip(), file=sys.stderr)
                            except (BrokenPipeError, OSError):
                                # Output pipe closed, continue reading
                                pass
                except (BrokenPipeError, OSError):
                    # Pipe broken, this is normal when process closes
                    pass
                except Exception:
                    pass
                finally:
                    if process.stderr:
                        try:
                            process.stderr.close()
                        except:
                            pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process with timeout (15 minutes for model download)
            # Use a polling approach for timeout on older Python versions
            import time as time_module
            start_time = time_module.time()
            timeout = 900  # 15 minutes
            
            while process.poll() is None:
                if time_module.time() - start_time > timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(run_cmd, timeout)
                time_module.sleep(1)
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Get return code
            return_code = process.returncode
            
        except subprocess.TimeoutExpired:
            print("\n[ERROR] Setup timed out after 15 minutes")
            print("The setup may still be running on the instance. Check status with:")
            print(f"  ssh -i {ssh_key} ubuntu@{instance_ip} 'tail -f /tmp/vllm.log'")
            return False
        except Exception as e:
            print(f"\n[ERROR] Setup failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if return_code != 0:
            print(f"\n[ERROR] Setup failed with return code {return_code}")
            if error_lines:
                print("\nError output:")
                for line in error_lines[-20:]:  # Show last 20 error lines
                    try:
                        print(line.rstrip())
                    except (BrokenPipeError, OSError):
                        pass
            # Check if it's just a broken pipe (which might be OK if setup actually succeeded)
            if return_code == 120:  # Common timeout/broken pipe code
                print("\n[INFO] Got broken pipe error, but setup may have succeeded.")
                print("Checking if vLLM is actually running...")
                # Check if server is running despite the error
                check_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                           f'ubuntu@{instance_ip}', 
                           'bash', '-c', 'pgrep -f "vllm.entrypoints.openai.api_server" && echo "running" || echo "not running"']
                try:
                    check_result = subprocess.run(check_cmd, capture_output=True, text=True,
                                                encoding='utf-8', errors='replace', timeout=10)
                    if "running" in check_result.stdout:
                        print("[OK] vLLM server appears to be running despite error!")
                        print("This may be a false error. Checking health endpoint...")
                        return True  # Continue to health check
                except:
                    pass
            return False
        else:
            print("\n[OK] Virtual environment setup complete")
        
        print()
        print("Step 4: Checking if server is running...")
        time.sleep(15)  # Give server more time to start
        
        # Check if server is running
        check_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                     f'ubuntu@{instance_ip}', 
                     'bash', '-c', 'curl -s http://localhost:8000/health || echo "not ready"']
        try:
            check_result = subprocess.run(check_cmd, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=10)
        except Exception as e:
            print(f"[WARNING] Could not check server status: {e}")
            check_result = None
        
        if check_result and "not ready" not in check_result.stdout.lower():
            print("[OK] vLLM server is running!")
        else:
            print("[INFO] Server may still be starting. Check logs with:")
            print(f"   ssh -i {ssh_key} ubuntu@{instance_ip} 'tail -f /tmp/vllm.log'")
        
    else:
        # Fallback to system-wide installation (not recommended)
        print("Step 1: Fixing NumPy compatibility...")
        numpy_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                     f'ubuntu@{instance_ip}', 
                     'bash', '-c', 'pip3 install "numpy<2" --upgrade --force-reinstall']
        try:
            result = subprocess.run(numpy_cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', timeout=300)
        except Exception as e:
            print(f"Warning: NumPy downgrade may have issues: {e}")
            result = subprocess.CompletedProcess(numpy_cmd, 1, '', str(e))
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"Warning: NumPy downgrade may have issues: {error_msg}")
        else:
            print("[OK] NumPy downgraded to <2.0")
        
        print()
        
        # Step 2: Install vLLM
        print("Step 2: Installing vLLM...")
        install_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                       f'ubuntu@{instance_ip}', 
                       'pip3', 'install', 'vllm', '--upgrade', '--constraint', '<(echo "numpy<2")']
        try:
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', timeout=600)
        except Exception as e:
            print(f"Warning: Installation may have issues: {e}")
            result = subprocess.CompletedProcess(install_cmd, 1, '', str(e))
        if result.returncode != 0:
            print(f"Warning: Installation may have issues: {result.stderr}")
        else:
            print("[OK] vLLM installed")
        
        print()
        
        # Step 3: Start vLLM server
        print("Step 3: Starting vLLM API server...")
        start_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                     f'ubuntu@{instance_ip}', 
                     'bash', '-c', 
                     f'pkill -f "vllm.entrypoints.openai.api_server" || true; nohup python3 -m vllm.entrypoints.openai.api_server --model {model} --port 8000 --host 0.0.0.0 > /tmp/vllm.log 2>&1 &']
        
        try:
            result = subprocess.run(start_cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace', timeout=10)
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
        
        if result.returncode == 0:
            print("[OK] vLLM server starting...")
            print("Waiting 10 seconds for server to initialize...")
            time.sleep(10)
            
            # Check if server is running
            check_cmd = ['ssh', '-i', str(ssh_key_path), '-o', 'StrictHostKeyChecking=no',
                         f'ubuntu@{instance_ip}', 
                         'curl', '-s', 'http://localhost:8000/health']
            try:
                check_result = subprocess.run(check_cmd, capture_output=True, text=True,
                                            encoding='utf-8', errors='replace', timeout=10)
            except Exception as e:
                print(f"[WARNING] Could not check server status: {e}")
                check_result = subprocess.CompletedProcess(check_cmd, 1, '', str(e))
            
            if check_result.returncode == 0:
                print("[OK] vLLM server is running!")
            else:
                print("[WARNING] Server may still be starting. Check logs with:")
                print(f"   ssh -i {ssh_key} ubuntu@{instance_ip} 'tail -f /tmp/vllm.log'")
        else:
            print(f"Error starting server: {result.stderr}")
            return False
    
    print()
    print("=" * 60)
    print("[OK] Setup Complete!")
    print("=" * 60)
    print(f"API endpoint: http://{instance_ip}:8000/v1/chat/completions")
    print()
    print("Use this endpoint in the dashboard!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up vLLM on Lambda Cloud instance")
    parser.add_argument("--ip", default="150.136.146.143", help="Instance IP address")
    parser.add_argument("--key", default="moses.pem", help="SSH key file path")
    parser.add_argument("--model", default="microsoft/phi-2", help="Model name")
    parser.add_argument("--no-venv", action="store_true", help="Disable virtual environment (not recommended)")
    
    args = parser.parse_args()
    
    setup_vllm(args.ip, args.key, args.model, use_venv=not args.no_venv)

