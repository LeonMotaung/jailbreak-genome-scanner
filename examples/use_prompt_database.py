"""Example: Using the structured prompt database."""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attackers.prompt_database import PromptDatabase, DifficultyLevel
from src.arena.jailbreak_arena import JailbreakArena
from src.defenders.llm_defender import LLMDefender
from src.models.jailbreak import AttackStrategy


def example_database_usage():
    """Example of using the prompt database."""
    print("=" * 70)
    print("Prompt Database Usage Example")
    print("=" * 70)
    
    # Load database
    db_path = project_root / "data" / "prompts_database.json"
    db = PromptDatabase.from_json(str(db_path))
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\n[STATS] Database Statistics:")
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Strategies: {', '.join(stats['strategies'])}")
    print(f"  Difficulty categories:")
    for category, count in stats['difficulty_categories'].items():
        print(f"    - {category}: {count} prompts")
    
    # Get prompts by strategy
    print(f"\n[PROMPTS] Roleplay Boundary Pusher prompts:")
    roleplay_prompts = db.get_by_strategy("Roleplay Boundary Pusher")
    print(f"  Found {len(roleplay_prompts)} prompts")
    for prompt in roleplay_prompts[:3]:
        print(f"    - {prompt['prompt_id']} ({prompt['difficulty']}): {prompt['prompt_text'][:60]}...")
    
    # Get prompts by difficulty
    print(f"\n[PROMPTS] High-difficulty prompts (H5):")
    high_prompts = db.get_by_difficulty("H5")
    print(f"  Found {len(high_prompts)} H5 prompts")
    
    # Get random prompts
    print(f"\n[PROMPTS] Random prompts (Medium difficulty):")
    random_prompts = db.get_random(
        difficulty_range=("M1", "M5"),
        count=3
    )
    for prompt in random_prompts:
        print(f"    - {prompt['prompt_id']} ({prompt['strategy']}): {prompt['prompt_text'][:60]}...")


def example_arena_with_database():
    """Example of using database prompts in Arena."""
    print("\n" + "=" * 70)
    print("Arena with Database Prompts Example")
    print("=" * 70)
    
    # Create arena (will auto-load database)
    arena = JailbreakArena()
    
    # Add mock defender
    class SimpleMockDefender(LLMDefender):
        def __init__(self):
            super().__init__(model_name="demo-model", api_endpoint="mock://prompt-db-demo")
        
        async def generate_response(self, prompt, **kwargs):
            return "I can't help with that. I'm designed to be helpful, harmless, and honest."
    
    defender = SimpleMockDefender()
    arena.add_defender(defender)
    
    # Generate attackers (will use database)
    arena.generate_attackers(
        num_strategies=3,
        strategies=[
            AttackStrategy.ROLEPLAY,
            AttackStrategy.EMOTIONAL_COERCION,
            AttackStrategy.FICTIONAL_FRAMING
        ]
    )
    
    print(f"\n[OK] Generated {len(arena.attackers)} attackers using database prompts")
    
    # Run a few rounds
    print("\n[BATTLE] Running evaluation with database prompts...")
    import asyncio
    results = asyncio.run(arena.evaluate(rounds=3))
    
    print(f"\n[RESULTS] Results:")
    print(f"  Total evaluations: {results['statistics']['total_evaluations']}")
    print(f"  Successful exploits: {results['statistics']['total_exploits']}")
    
    # Show some prompts used
    if results.get('evaluation_history'):
        print(f"\n[PROMPTS] Example prompts used:")
        for eval in results['evaluation_history'][:3]:
            print(f"    - {eval.attack_strategy.value}: {eval.prompt[:80]}...")


if __name__ == "__main__":
    example_database_usage()
    example_arena_with_database()

