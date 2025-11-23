# 🛡️ Defensive Enhancement with Lambda Cloud Open-Source Models

## Overview

This defensive enhancement system is designed to work with **open-source models deployed on Lambda Cloud**, using only Lambda credits (no API keys for closed models required).

## Why Open-Source Models on Lambda Cloud?

- ✅ **No API costs** - Only pay for compute time
- ✅ **Full control** - Deploy and manage your own models
- ✅ **Privacy** - Data stays on your instances
- ✅ **Flexibility** - Use any open-source model
- ✅ **Credits-based** - Use Lambda Cloud credits only

## Supported Open-Source Models

### Recommended Models for Defense Testing

| Model | Size | Instance Type | Cost/Hour | Best For |
|-------|------|---------------|-----------|----------|
| **phi-2** ⭐ | 2.7B | gpu_1x_a10 | ~$0.50 | Testing & Development |
| **mistral-7b-instruct** | 7B | gpu_1x_a10 | ~$0.50 | High Quality Responses |
| **qwen-7b-chat** | 7B | gpu_1x_a10 | ~$0.50 | Multilingual Support |
| **falcon-7b-instruct** | 7B | gpu_1x_a10 | ~$0.50 | Instruction Following |
| **llama-2-7b-chat** | 7B | gpu_1x_a10 | ~$0.50 | General Purpose |
| **llama-2-13b-chat** | 13B | gpu_1x_a100 | ~$1.10 | Better Capabilities |

## Quick Start with Defense Features

### 1. Deploy Model on Lambda Cloud

```bash
# Deploy recommended model (phi-2)
python deploy_models.py deploy phi-2

# Or use interactive setup
python setup_lambda_models.py
```

This will:
- Launch Lambda Cloud instance
- Download and deploy the model
- Set up API endpoint
- Save configuration to `data/lambda_deployments.json`

### 2. Create Defender with Defense Filter

```python
from src.defenders.llm_defender import LLMDefender
from src.arena.jailbreak_arena import JailbreakArena

# Load deployment config
import json
with open("data/lambda_deployments.json", 'r') as f:
    deployments = json.load(f)
    
instance_id = deployments["deployed_models"]["phi-2"]["instance_id"]
api_endpoint = deployments["deployed_models"]["phi-2"]["api_endpoint"]

# Create defender with defense filter enabled
defender = LLMDefender(
    model_name="microsoft/phi-2",
    model_type="local",
    use_lambda=True,
    lambda_instance_id=instance_id,
    lambda_api_endpoint=api_endpoint,
    enable_defense_filter=True,  # Enable pattern-based defense
    defense_filter_blocking=True  # Block threats automatically
)
```

### 3. Run Evaluations with Pattern Learning

```python
# Initialize arena with pattern database
arena = JailbreakArena(use_pattern_database=True)
arena.add_defender(defender)

# Generate attackers (BIO_HAZARD and CYBER_EXPLOIT always included)
arena.generate_attackers(num_strategies=10)

# Run evaluations (patterns automatically collected)
results = await arena.evaluate(rounds=10)

# Access pattern database
db = arena.pattern_database
analysis = db.analyze_patterns()
print(f"Total exploit patterns learned: {analysis['total_patterns']}")
```

## Defense Features with Open-Source Models

### 1. Pattern-Based Threat Detection

The system learns from successful exploits and blocks similar attacks:

```python
# After running evaluations, similar attacks are automatically blocked
# No configuration needed - it learns from your evaluations
```

### 2. Pre-Processing Filter

Blocks threats before they reach the model:

```python
defender = LLMDefender(
    model_name="microsoft/phi-2",
    model_type="local",
    use_lambda=True,
    lambda_instance_id=instance_id,
    enable_defense_filter=True,  # Enable filter
    defense_filter_blocking=True  # Block threats
)

# Threats similar to known exploits are blocked immediately
response = await defender.generate_response(prompt)
# If threat detected, returns safe response without calling model
```

### 3. Pattern Database

Stores and analyzes exploit patterns:

```python
# Access pattern database
db = arena.pattern_database

# Analyze patterns
analysis = db.analyze_patterns()
print(f"Strategies: {analysis['strategies']}")
print(f"Average attack cost: {analysis['average_attack_cost']}")

# Get vulnerability summary
vulns = db.get_vulnerability_summary()
print(f"Recommendations: {vulns['recommendations']}")

# Find similar patterns
similar = db.find_similar_patterns(embedding, threshold=0.8)
```

## Dashboard Usage

### Enable Defense Filter in Dashboard

1. **Start Dashboard**
   ```bash
   python -m streamlit run dashboard/arena_dashboard.py
   ```

2. **Configure Defender**
   - Select "Lambda Cloud" as defender type
   - Choose your deployed model (e.g., "phi-2")
   - Enter instance ID and API endpoint
   - **Enable "Defense Filter"** checkbox
   - Set "Block Threats" to True

3. **Run Evaluations**
   - Patterns are automatically collected
   - Similar attacks are blocked in real-time
   - View statistics in dashboard

## Cost Optimization

### Tips for Using Lambda Credits Efficiently

1. **Use Smaller Models for Testing**
   - Start with `phi-2` (2.7B) - fastest and cheapest
   - Upgrade to larger models only when needed

2. **Stop Instances When Not in Use**
   ```bash
   # Clean up when done
   python deploy_models.py cleanup phi-2
   ```

3. **Batch Evaluations**
   - Run multiple rounds in one session
   - Reduces instance startup overhead

4. **Monitor Usage**
   - Check Lambda Cloud dashboard regularly
   - Set up billing alerts

## Model Comparison for Defense Testing

### phi-2 (Recommended for Testing)
- ✅ Smallest and fastest
- ✅ Lowest cost
- ✅ Good for initial testing
- ✅ Fast pattern learning

### mistral-7b-instruct (Recommended for Quality)
- ✅ High quality responses
- ✅ Better at following safety guidelines
- ✅ Good for production testing

### qwen-7b-chat (Multilingual)
- ✅ Multilingual support
- ✅ Good for diverse attack patterns
- ✅ Useful for international threat testing

## Troubleshooting

### Model Not Responding

```python
# Check instance status
from src.integrations.lambda_cloud import LambdaCloudClient

client = LambdaCloudClient()
instances = await client.list_instances()
print(instances)
```

### Defense Filter Not Working

```python
# Check if filter is enabled
if defender.preprocessing_filter:
    stats = defender.preprocessing_filter.get_statistics()
    print(f"Filter stats: {stats}")
else:
    print("Filter not enabled - check enable_defense_filter parameter")
```

### Pattern Database Not Collecting

```python
# Check if pattern database is enabled
if arena.pattern_database:
    print(f"Patterns collected: {len(arena.pattern_database.patterns)}")
else:
    print("Pattern database not enabled - check use_pattern_database parameter")
```

## Best Practices

1. **Start Small**
   - Begin with `phi-2` for initial testing
   - Collect patterns and analyze vulnerabilities
   - Upgrade to larger models for production

2. **Enable Defense Filter**
   - Always enable `enable_defense_filter=True`
   - Set `defense_filter_blocking=True` for production
   - Monitor filter statistics

3. **Regular Pattern Analysis**
   - Review pattern database regularly
   - Check vulnerability summaries
   - Implement recommendations

4. **Cost Management**
   - Stop instances when not in use
   - Use smaller models for development
   - Batch evaluations to reduce overhead

## Example: Complete Workflow

```python
import asyncio
from src.defenders.llm_defender import LLMDefender
from src.arena.jailbreak_arena import JailbreakArena
import json

async def main():
    # Load deployment config
    with open("data/lambda_deployments.json", 'r') as f:
        deployments = json.load(f)
    
    model_key = "phi-2"
    instance_id = deployments["deployed_models"][model_key]["instance_id"]
    api_endpoint = deployments["deployed_models"][model_key]["api_endpoint"]
    
    # Create defender with defense filter
    defender = LLMDefender(
        model_name="microsoft/phi-2",
        model_type="local",
        use_lambda=True,
        lambda_instance_id=instance_id,
        lambda_api_endpoint=api_endpoint,
        enable_defense_filter=True,
        defense_filter_blocking=True
    )
    
    # Initialize arena
    arena = JailbreakArena(use_pattern_database=True)
    arena.add_defender(defender)
    arena.generate_attackers(num_strategies=10)
    
    # Run evaluations
    print("Running evaluations with pattern learning...")
    results = await arena.evaluate(rounds=5)
    
    # Analyze patterns
    db = arena.pattern_database
    analysis = db.analyze_patterns()
    print(f"\nPatterns learned: {analysis['total_patterns']}")
    print(f"Strategies: {analysis['strategies']}")
    
    # Get recommendations
    vulns = db.get_vulnerability_summary()
    print(f"\nRecommendations:")
    for rec in vulns['recommendations']:
        print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

**Remember**: All defensive enhancement features work seamlessly with open-source models on Lambda Cloud. No API keys for closed models needed!

