"""Pipeline orchestration - Uses all existing functionality."""

from pathlib import Path
from typing import Optional
import argparse

from src.config import Config
from src.ingest import load_prompts
from src.run_model import create_model, run_prompts
from src.classify import classify_responses
from src.embed_cluster import create_genome_map
from src.compute_jvi import compute_jvi, save_jvi_report


def run_pipeline(
    input_file: Path,
    config: Config,
    output_dir: Optional[Path] = None,
    model_kwargs: Optional[dict] = None
):
    """
    Run complete JGS pipeline using existing fully functional system.
    
    Args:
        input_file: Path to prompts.jsonl or prompts.csv
        config: Configuration object
        output_dir: Output directory (defaults to config.OUTPUT_DIR)
        model_kwargs: Optional model configuration overrides
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Jailbreak Genome Scanner Pipeline")
    print("Using existing fully functional system")
    print("=" * 60)
    
    # Step 1: Ingest prompts
    print("\n[1/5] Ingesting prompts...")
    prompts = load_prompts(input_file)
    print(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Step 2: Run model
    print("\n[2/5] Running prompts through model...")
    model = create_model(config, **(model_kwargs or {}))
    responses = run_prompts(
        prompts,
        model,
        output_dir / "responses.jsonl"
    )
    
    # Step 3: Classify responses
    print("\n[3/5] Classifying responses...")
    classified = classify_responses(
        responses,
        classifier_type="rule_based",
        output_path=output_dir / "dataset.jsonl"
    )
    
    # Step 4: Embed and cluster
    print("\n[4/5] Creating genome map (embedding, clustering)...")
    genome_map = create_genome_map(
        classified,
        config,
        output_path=output_dir / "genome_map.json"
    )
    
    # Step 5: Compute JVI
    print("\n[5/5] Computing JVI...")
    jvi_report = compute_jvi(
        classified,
        genome_map,
        weights=config.DEFAULT_WEIGHTS
    )
    save_jvi_report(jvi_report, output_dir / "jvi_report.json")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  - Total prompts: {len(prompts)}")
    print(f"  - Responses: {len(responses)}")
    print(f"  - Classified: {len(classified)}")
    print(f"  - Clusters: {genome_map.get('n_clusters', 0)}")
    print(f"  - JVI Score: {jvi_report['jvi']:.4f} ({jvi_report['jvi_percentage']:.2f}%)")
    print(f"  - Risk Level: {jvi_report['risk_assessment']['level']}")
    print(f"\nOutput files saved to: {output_dir}")
    
    return {
        "prompts": prompts,
        "responses": responses,
        "classified": classified,
        "genome_map": genome_map,
        "jvi_report": jvi_report
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="JGS Pipeline")
    parser.add_argument("input_file", type=Path, help="Input prompts.jsonl or prompts.csv")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--model-type", default="mock", help="Model type: mock, openai, lambda")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--instance-id", help="Lambda instance ID")
    parser.add_argument("--api-endpoint", help="API endpoint URL")
    parser.add_argument("--show-harmful", action="store_true", help="Show harmful content")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override with CLI args
    if args.show_harmful:
        config.SHOW_HARMFUL = True
    if args.model_type:
        config.MODEL_TYPE = args.model_type
    if args.model_name:
        config.MODEL_NAME = args.model_name
    
    config.ensure_dirs()
    
    # Model kwargs
    model_kwargs = {
        "model_type": config.MODEL_TYPE,
        "model_name": config.MODEL_NAME
    }
    
    if args.instance_id:
        model_kwargs["instance_id"] = args.instance_id
    if args.api_endpoint:
        model_kwargs["api_endpoint"] = args.api_endpoint
    
    # Run pipeline
    run_pipeline(
        args.input_file,
        config,
        output_dir=args.output_dir,
        model_kwargs=model_kwargs
    )


if __name__ == "__main__":
    main()

