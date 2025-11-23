"""
Deploy Jailbreak Genome Scanner Dashboard to Modal

This script deploys the Streamlit dashboard to Modal's serverless platform.
Similar to the vLLM deployment, it packages the dashboard and all dependencies.

Usage:
    modal serve deploy/dashboard.py   # Run locally with hot-reload
    modal deploy deploy/dashboard.py  # Deploy to production
"""

import shlex
import subprocess
from pathlib import Path

import modal

# Define the project root and dashboard paths
project_root = Path(__file__).parent.parent
dashboard_dir = project_root / "dashboard"
src_dir = project_root / "src"
data_dir = project_root / "data"

# Main dashboard script
dashboard_script_local = dashboard_dir / "arena_dashboard.py"
dashboard_script_remote = "/root/dashboard/arena_dashboard.py"

# Additional dashboard files
css_file_local = dashboard_dir / "professional_theme.css"
css_file_remote = "/root/dashboard/professional_theme.css"

templates_dir_local = dashboard_dir / "templates"
templates_dir_remote = "/root/dashboard/templates"

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
        # Security
        "python-jose>=3.3.0",
        "cryptography>=42.0.0",
    )
    # Add the dashboard files
    .add_local_file(dashboard_script_local, dashboard_script_remote)
    .add_local_file(css_file_local, css_file_remote)
    .add_local_dir(templates_dir_local, templates_dir_remote)
    # Add the entire src directory (contains all the modules)
    .add_local_dir(src_dir, "/root/src")
    # Add data directory for prompts database and other data
    .add_local_dir(data_dir, "/root/data")
)

# Create the Modal app
app = modal.App(name="jailbreak-genome-scanner-dashboard", image=image)

# Check if required files exist
if not dashboard_script_local.exists():
    raise RuntimeError(
        f"Dashboard script not found at {dashboard_script_local}! "
        "Ensure you're running this from the project root."
    )

STREAMLIT_PORT = 8000
MINUTES = 60  # seconds


@app.function(
    # Allow multiple concurrent users
    concurrency_limit=10,
    # Keep the dashboard running for 30 minutes after last request
    container_idle_timeout=30 * MINUTES,
    # Timeout for starting the dashboard
    timeout=10 * MINUTES,
    # Create a persistent volume for logs and cached data
    volumes={
        "/root/logs": modal.Volume.from_name("dashboard-logs", create_if_missing=True),
        "/root/.cache": modal.Volume.from_name(
            "dashboard-cache", create_if_missing=True
        ),
    },
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

    os.chdir("/root")

    # Build the Streamlit command
    target = shlex.quote(dashboard_script_remote)
    cmd = [
        "streamlit",
        "run",
        target,
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
    print("\nTo deploy permanently, run:")
    print("  modal deploy deploy/dashboard.py")
