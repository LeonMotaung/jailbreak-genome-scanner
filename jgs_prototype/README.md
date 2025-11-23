# 🛡️ Jailbreak Genome Scanner - Starter Repository

**This repository provides a clean interface to the fully functional Jailbreak Genome Scanner system.**

All functionality uses the existing, production-ready codebase under the hood. This starter format provides:
- Simple, modular pipeline interface
- Standard input/output formats (JSONL/CSV)
- Easy-to-use CLI and dashboard
- All the power of the full system

## Quick Start

```bash
# Install dependencies (from parent directory)
cd ..
pip install -r requirements.txt

# Run demo
cd jgs_prototype
python demo.py

# View dashboard
streamlit run src/dashboard.py
```

## What This Provides

This starter repo wraps the existing fully functional system:

- ✅ **Prompt Database** - 60+ curated prompts with difficulty levels
- ✅ **Lambda Cloud Integration** - GPU-accelerated model inference
- ✅ **Multiple Model Providers** - OpenAI, Anthropic, Lambda Cloud
- ✅ **Advanced Safety Classifier** - Rule-based + fine-tuned scaffolding
- ✅ **Genome Map Generator** - UMAP + HDBSCAN clustering
- ✅ **JVI Calculator** - Production-ready vulnerability scoring
- ✅ **Web Scraping** - GitHub, Reddit, StackExchange intelligence gathering
- ✅ **Threat Intelligence** - Pattern database and adaptive defenses
- ✅ **Full Dashboard** - Professional arena visualization

## Pipeline Structure

```
Input (prompts.jsonl/CSV)
    ↓
[ingest.py] → Uses existing PromptDatabase
    ↓
[run_model.py] → Uses existing LLMDefender (Mock/OpenAI/Lambda)
    ↓
[classify.py] → Uses existing SafetyClassifier
    ↓
[embed_cluster.py] → Uses existing GenomeMapGenerator
    ↓
[compute_jvi.py] → Uses existing JVICalculator
    ↓
Outputs (JSON/JSONL)
```

## Usage

### Run Pipeline

```bash
# Mock model (testing)
python src/pipeline.py data/input/prompts.jsonl --model-type mock

# Lambda Cloud (your existing instances!)
python src/pipeline.py data/input/prompts.jsonl \
  --model-type lambda \
  --instance-id fe9576a10897449a8311eb667866abe8 \
  --api-endpoint http://localhost:8001/v1/chat/completions

# OpenAI
export OPENAI_API_KEY=your_key
python src/pipeline.py data/input/prompts.jsonl --model-type openai
```

### View Dashboard

```bash
# Simple dashboard
streamlit run src/dashboard.py

# Full existing dashboard (recommended!)
streamlit run ../dashboard/arena_dashboard.py
```

## Input Format

**JSONL:**
```json
{"prompt_id": "p001", "attack_family": "roleplay", "text": "Prompt here"}
```

**CSV (chess-evals style):**
```csv
key,input,attack_family,numeric_eval
p001,"Prompt here",roleplay,3
```

## Outputs

- `responses.jsonl` - Model responses
- `dataset.jsonl` - Classified responses with labels/severity
- `genome_map.json` - Clusters and 2D coordinates
- `jvi_report.json` - JVI score and metrics

## Features

✅ Uses all existing functionality  
✅ Clean starter interface  
✅ Standard input/output formats  
✅ Lambda Cloud integration ready  
✅ Web scraping enabled  
✅ Full dashboard available  

## Next Steps

1. **Use Your Lambda Instances**: Configure with your existing instance IDs
2. **Add Your Prompts**: Use the prompt database or provide your own JSONL/CSV
3. **Run Evaluations**: Use the pipeline or the full arena dashboard
4. **Analyze Results**: View JVI scores, genome maps, and leaderboards

## Full System Access

Want the full features? The existing system includes:
- Competitive arena framework
- Real-time dashboard
- Threat intelligence engine
- Adaptive defense systems
- Evolution engines
- Vector databases

See the parent README.md for full documentation!

