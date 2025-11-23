"""Streamlit dashboard - Wrapper that can use existing dashboard or simplified version."""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Optional

# Try to use existing dashboard, fallback to simple version
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dashboard.arena_dashboard import main as existing_dashboard_main
    USE_EXISTING = True
except ImportError:
    USE_EXISTING = False


def load_results(output_dir: Path) -> Dict[str, Any]:
    """Load pipeline results from output directory."""
    output_dir = Path(output_dir)
    results = {}
    
    jvi_path = output_dir / "jvi_report.json"
    if jvi_path.exists():
        with open(jvi_path, 'r') as f:
            results["jvi_report"] = json.load(f)
    
    genome_path = output_dir / "genome_map.json"
    if genome_path.exists():
        with open(genome_path, 'r') as f:
            results["genome_map"] = json.load(f)
    
    dataset_path = output_dir / "dataset.jsonl"
    if dataset_path.exists():
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        results["dataset"] = dataset
    
    return results


def render_simple_dashboard(results: Dict[str, Any], show_harmful: bool = False):
    """Simple dashboard implementation."""
    st.set_page_config(page_title="JGS Dashboard", layout="wide")
    st.title("🛡️ Jailbreak Genome Scanner Dashboard")
    
    if not results:
        st.error("No results found. Run pipeline first.")
        return
    
    # JVI Section
    if "jvi_report" in results:
        jvi_report = results["jvi_report"]
        st.header("📊 Jailbreak Vulnerability Index (JVI)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("JVI Score", f"{jvi_report.get('jvi_percentage', 0):.2f}%")
        with col2:
            st.metric("Exploit Rate", f"{jvi_report.get('components', {}).get('exploit_rate', 0):.2%}")
        with col3:
            st.metric("Risk Level", jvi_report.get('risk_assessment', {}).get('level', 'UNKNOWN'))
    
    # Genome Map
    if "genome_map" in results:
        genome_map = results["genome_map"]
        st.header("🗺️ Genome Map")
        
        if "coordinates_2d" in genome_map and len(genome_map["coordinates_2d"]) > 0:
            coords = genome_map["coordinates_2d"]
            labels = genome_map.get("cluster_labels", [])
            
            df_map = pd.DataFrame({
                "x": [c[0] for c in coords],
                "y": [c[1] if len(c) > 1 else 0 for c in coords],
                "cluster": labels
            })
            
            fig = px.scatter(df_map, x="x", y="y", color="cluster", title="2D Genome Map")
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Stats
    if "dataset" in results:
        dataset = results["dataset"]
        st.header("📈 Statistics")
        
        df = pd.DataFrame(dataset)
        jailbroken = df[df["label"] == "jailbroken"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(dataset))
        with col2:
            st.metric("Jailbroken", len(jailbroken))
        with col3:
            st.metric("Safe", len(dataset) - len(jailbroken))
        
        # Leaderboard
        if not df.empty:
            st.subheader("Attack Family Leaderboard")
            family_stats = df.groupby("attack_family").agg({
                "prompt_id": "count",
                "label": lambda x: (x == "jailbroken").sum()
            }).reset_index()
            family_stats.columns = ["Attack Family", "Total", "Successful"]
            family_stats["Success Rate"] = family_stats["Successful"] / family_stats["Total"]
            family_stats = family_stats.sort_values("Success Rate", ascending=False)
            
            st.dataframe(family_stats, use_container_width=True)


def main():
    """Main dashboard entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JGS Dashboard")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"),
                       help="Output directory with pipeline results")
    parser.add_argument("--use-existing", action="store_true",
                       help="Use existing full dashboard")
    
    args = parser.parse_args()
    
    if USE_EXISTING and args.use_existing:
        # Use existing full dashboard
        existing_dashboard_main()
    else:
        # Use simple dashboard
        results = load_results(args.output_dir)
        render_simple_dashboard(results)


if __name__ == "__main__":
    # When running as streamlit app
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Default: simple dashboard
        results = load_results(Path("data/output"))
        render_simple_dashboard(results)

