"""
Deploy Jailbreak Genome Scanner Dashboard to Modal

This script deploys the Streamlit dashboard to Modal's serverless platform.
Similar to the vLLM deployment, it packages the dashboard and all dependencies.

Usage:
    modal serve deploy/dashboard.py   # Run locally with hot-reload
    modal deploy deploy/dashboard.py  # Deploy to production
"""

import subprocess
from pathlib import Path

import modal

# Define the project root and dashboard paths
project_root = Path(__file__).parent.parent
dashboard_dir = project_root / "dashboard"
src_dir = project_root / "src"
data_dir = project_root / "data"

# Create the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # Core Streamlit
        "streamlit~=1.31.0",
        # Data visualization
        "plotly~=5.18.0",
        "pandas~=2.0.0",
        "numpy~=1.24.0",
        "matplotlib~=3.8.0",
        "seaborn~=0.13.0",
        # LLM APIs
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "cohere>=4.47",
        # Async and HTTP
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
        # Data models
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        # Environment
        "python-dotenv>=1.0.0",
        # Vector database
        "chromadb>=0.4.22",
        "sentence-transformers>=2.2.0",
        # ML and clustering
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.5",
        "scipy>=1.11.0",
        # Graph analysis
        "networkx>=3.2.1",
        "python-louvain>=0.16",
        # Utilities
        "python-dateutil>=2.8.2",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        # Local models (optional, for Lambda integration)
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "openai",
        # Security
        "python-jose>=3.3.0",
        "cryptography>=42.0.0",
    )
    # Add the entire dashboard directory (includes arena_dashboard.py, CSS, templates)
    .add_local_dir(dashboard_dir, "/root/dashboard")
    # Add the entire src directory (contains all the modules)
    .add_local_dir(src_dir, "/root/src")
    # Add data directory for prompts database and other data
    .add_local_dir(data_dir, "/root/data")
)

# =============================================================================
# MODAL VLLM ENDPOINTS - Paste your Modal vLLM deployment URLs here
# =============================================================================
# To get these URLs, run: modal app list
# Then for each deployed vLLM function, get the URL from the Modal dashboard
# or by running: modal function get-url <app-name> <function-name>
#
# Example URLs will look like:
# https://your-workspace--example-vllm-inference-serve-llama2-7b-chat.modal.run
#
MODAL_ENDPOINTS = {
    # Paste your Modal vLLM endpoint URLs here:
    "MODAL_LLAMA2_7B_URL": "",  # e.g., "https://workspace--app-serve-llama2-7b-chat.modal.run"
    "MODAL_LLAMA2_13B_URL": "",  # e.g., "https://workspace--app-serve-llama2-13b-chat.modal.run"
    "MODAL_MISTRAL_7B_URL": "",  # e.g., "https://workspace--app-serve-mistral-7b-instruct.modal.run"
    "MODAL_PHI2_URL": "",  # e.g., "https://workspace--app-serve-phi2.modal.run"
    "MODAL_QWEN_8B_URL": "",  # e.g., "https://workspace--app-serve-qwen-8b.modal.run"
    # Add more endpoints as needed:
    # "MODAL_CUSTOM_MODEL_URL": "",
}

# Filter out empty endpoints
MODAL_ENDPOINTS = {k: v for k, v in MODAL_ENDPOINTS.items() if v}

# =============================================================================

# Create the Modal app
app = modal.App(name="jailbreak-genome-scanner-dashboard", image=image)

STREAMLIT_PORT = 8000
MINUTES = 60  # seconds


@app.function(
    # Allow multiple concurrent users
    # Keep the dashboard running for 30 minutes after last request
    container_idle_timeout=30 * MINUTES,
    # Timeout for starting the dashboard
    timeout=10 * MINUTES,
    
    # Create a persistent volume for logs and cached data
    volumes={
        "/root/logs": modal.Volume.from_name("dashboard-logs", create_if_missing=True),
    },
    # Pass Modal endpoints as environment variables
    env=MODAL_ENDPOINTS,
)
@modal.web_server(port=STREAMLIT_PORT, startup_timeout=5 * MINUTES)
def serve():
    """
    Spawn the Streamlit server in a subprocess.

    The @modal.web_server decorator exposes the port to the internet,
    and the subprocess.Popen starts Streamlit in the background.
    """
    # Change to the dashboard directory
    import os

    os.chdir("/root/dashboard")

    # Build the Streamlit command - just run arena_dashboard.py
    cmd = [
        "streamlit",
        "run",
        "arena_dashboard.py",
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.address",
        "0.0.0.0",
        # Disable CORS and XSRF protection for Modal deployment
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        # Disable file watcher for production
        "--server.fileWatcherType=none",
        # Set headless mode
        "--server.headless=true",
    ]

    # Start Streamlit in the background
    subprocess.Popen(" ".join(cmd), shell=True)


# Optional: Add a local entrypoint for testing
@app.local_entrypoint()
def main():
    """
    Local entrypoint to get the dashboard URL.

    Usage:
        modal run deploy/dashboard.py
    """
    print("🚀 Jailbreak Genome Scanner Dashboard")
    print("=" * 50)
    print(f"Dashboard URL: {serve.web_url}")
    print("=" * 50)
    print("\nThe dashboard is now running on Modal!")
    print("Visit the URL above to access the dashboard.")

    if MODAL_ENDPOINTS:
        print("\n📡 Modal vLLM Endpoints configured:")
        for key, url in MODAL_ENDPOINTS.items():
            print(f"  • {key}: {url}")
    else:
        print("\n⚠️  No Modal vLLM endpoints configured.")
        print("Edit deploy/dashboard.py and add your endpoint URLs to MODAL_ENDPOINTS")

    print("\nTo deploy permanently, run:")
    print("  modal deploy deploy/dashboard.py")


# Helper function to get URLs from deployed vLLM functions
@app.function()
def get_vllm_endpoints():
    """
    Helper function to retrieve URLs from deployed vLLM functions.

    Usage:
        modal run deploy/dashboard.py::get_vllm_endpoints
    """
    print("🔍 Looking for deployed vLLM functions...")
    print("\nTo manually get URLs for your vLLM endpoints:")
    print("1. List your apps: modal app list")
    print("2. Find your vLLM app (e.g., 'example-vllm-inference')")
    print("3. Get function URLs:")
    print("   modal function get-url example-vllm-inference serve_llama2_7b_chat")
    print("   modal function get-url example-vllm-inference serve_llama2_13b_chat")
    print("   modal function get-url example-vllm-inference serve_mistral_7b_instruct")
    print("   modal function get-url example-vllm-inference serve_phi2")
    print("   modal function get-url example-vllm-inference serve_qwen_8b")
    print("\n4. Paste the URLs into the MODAL_ENDPOINTS dict in deploy/dashboard.py")
