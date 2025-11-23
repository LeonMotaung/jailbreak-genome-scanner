"""Data models for jailbreak evaluation."""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class AttackStrategy(str, Enum):
    """Attack strategy types."""
    # Semantic attacks
    ROLEPLAY = "roleplay"
    EMOTIONAL_COERCION = "emotional_coercion"
    TRANSLATION_ATTACK = "translation_attack"
    PROMPT_INVERSION = "prompt_inversion"
    CHAIN_OF_COMMAND = "chain_of_command"
    FICTIONAL_FRAMING = "fictional_framing"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    INDIRECT_REQUEST = "indirect_request"
    POLICY_PROBING = "policy_probing"
    HONEYPOT = "honeypot"
    PERMISSION_EXPLOITATION = "permission_exploitation"
    # Specialized defense-focused agents
    BIO_HAZARD = "bio_hazard"  # Bio-Radar: Pathogen synthesis testing
    CYBER_EXPLOIT = "cyber_exploit"  # Cyber-Sentinel: Zero-day hunting

    # Optimization-based attacks (new)
    GCG = "gcg"  # Greedy Coordinate Gradient
    PAIR = "pair"  # Prompt Automatic Iterative Refinement
    CIPHER = "cipher"  # Encoded/obfuscated attacks
    ASCII_ART = "ascii_art"  # ASCII art obfuscation
    GLITCH_TOKENS = "glitch_tokens"  # Tokenizer confusion


class ViolationDomain(str, Enum):
    """Domains of harm."""
    CYBER = "cyber"
    BIOLOGICAL = "biological"
    FRAUD = "fraud"
    VIOLENCE = "violence"
    HARASSMENT = "harassment"
    PRIVACY = "privacy"
    MISINFORMATION = "misinformation"
    ILLEGAL = "illegal"
    OTHER = "other"


class SeverityLevel(int, Enum):
    """Severity levels (0-5)."""
    SAFE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


class EvaluationResult(BaseModel):
    """Result of a single jailbreak evaluation."""
    
    # Identifiers
    id: str = Field(..., description="Unique evaluation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    # Attack details
    attack_strategy: AttackStrategy = Field(..., description="Attack strategy used")
    attacker_id: str = Field(..., description="Attacker identifier")
    prompt: str = Field(..., description="Adversarial prompt")
    
    # Defender details
    defender_id: str = Field(..., description="Defender (model) identifier")
    defender_model: str = Field(..., description="Model name")
    response: str = Field(..., description="Model response")
    
    # Safety classification
    is_jailbroken: bool = Field(..., description="Whether jailbreak was successful")
    severity: SeverityLevel = Field(..., description="Severity level")
    violation_domains: List[ViolationDomain] = Field(
        default_factory=list,
        description="Violation domains"
    )
    harmfulness_score: float = Field(
        ge=0.0, le=1.0,
        description="Harmfulness score (0-1)"
    )
    
    # Embedding and clustering
    embedding: Optional[List[float]] = Field(None, description="Response embedding vector")
    cluster_id: Optional[int] = Field(None, description="Failure cluster ID")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AttackerProfile(BaseModel):
    """Profile of an attacker agent."""
    
    id: str = Field(..., description="Attacker ID")
    name: str = Field(..., description="Attacker name")
    strategy: AttackStrategy = Field(..., description="Primary strategy")
    
    # Performance metrics
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    total_attempts: int = Field(default=0, description="Total attempts")
    successful_exploits: int = Field(default=0, description="Successful exploits")
    total_points: float = Field(default=0.0, description="Total points earned")
    
    # Emotional/cognitive state (Inside Out style)
    emotional_state: Optional[Dict[str, float]] = Field(
        None,
        description="Emotional state components (joy, fear, anger, etc.)"
    )
    
    # Permissions
    permissions: List[str] = Field(default_factory=list, description="Granted permissions")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DefenderProfile(BaseModel):
    """Profile of a defender (model under test)."""
    
    id: str = Field(..., description="Defender ID")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (openai, anthropic, local, etc.)")
    model_path: Optional[str] = Field(None, description="Model path or endpoint")
    
    # Vulnerability metrics
    jvi_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Jailbreak Vulnerability Index")
    exploit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall exploit rate")
    mean_severity: float = Field(default=0.0, ge=0.0, le=5.0, description="Mean severity of exploits")
    high_severity_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Rate of high-severity exploits (>=3)"
    )
    failure_diversity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Failure mode diversity (entropy)"
    )
    
    # Evaluation stats
    total_evaluations: int = Field(default=0, description="Total evaluations run")
    total_exploits: int = Field(default=0, description="Total successful exploits")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GenomeMapPoint(BaseModel):
    """A point on the Genome Map representing a failure cluster."""
    
    cluster_id: int = Field(..., description="Cluster ID")
    x: float = Field(..., description="X coordinate (UMAP)")
    y: float = Field(..., description="Y coordinate (UMAP)")
    
    # Cluster statistics
    size: int = Field(..., description="Number of exploits in cluster")
    mean_severity: float = Field(..., description="Mean severity")
    violation_domains: List[ViolationDomain] = Field(..., description="Violation domains")
    attack_strategies: List[AttackStrategy] = Field(..., description="Attack strategies")
    
    # Representative examples
    representative_prompts: List[str] = Field(..., description="Example prompts")
    representative_responses: List[str] = Field(..., description="Example responses")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ArenaRound(BaseModel):
    """A single round in the Jailbreak Arena."""
    
    round_number: int = Field(..., description="Round number")
    timestamp: datetime = Field(default_factory=datetime.now, description="Round timestamp")
    
    # Participants
    defender_id: str = Field(..., description="Defender ID")
    attacker_ids: List[str] = Field(..., description="Attacker IDs")
    
    # Results
    evaluations: List[EvaluationResult] = Field(..., description="Evaluation results")
    successful_attacks: int = Field(default=0, description="Number of successful attacks")
    
    # Leaderboard updates
    attacker_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Points earned by each attacker"
    )
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ArenaLeaderboard(BaseModel):
    """Arena leaderboard with attacker rankings."""
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Leaderboard timestamp")
    
    # Top attackers
    top_attackers: List[AttackerProfile] = Field(..., description="Top-ranked attackers")
    
    # Defender rankings
    defender_rankings: List[DefenderProfile] = Field(..., description="Defender vulnerability rankings")
    
    # Statistics
    total_rounds: int = Field(default=0, description="Total rounds played")
    total_exploits: int = Field(default=0, description="Total successful exploits")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

