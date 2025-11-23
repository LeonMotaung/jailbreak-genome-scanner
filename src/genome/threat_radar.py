"""Threat Radar: Specialized visualization for dual-use capability detection.

Renamed from "Genome Map" to "Threat Radar" for defense context.
Highlights "Pathogen Synthesis Vectors" and "Zero-Day Exploit Patterns".
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.models.jailbreak import (
    EvaluationResult, GenomeMapPoint, ViolationDomain, AttackStrategy
)
from src.genome.map_generator import GenomeMapGenerator
from src.utils.logger import log
from collections import defaultdict


class ThreatRadar(GenomeMapGenerator):
    """Threat Radar: Defense-focused vulnerability mapping system.
    
    Extends GenomeMapGenerator with specialized visualization for:
    - Pathogen Synthesis Vectors (biological threats)
    - Zero-Day Exploit Patterns (cyber threats)
    - Attack Pattern Fingerprints (for model vaccination)
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Threat Radar.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        super().__init__(embedding_model)
        log.info("Threat Radar initialized (defense-focused mapping)")
    
    def generate(
        self,
        evaluations: List[EvaluationResult],
        min_cluster_size: int = 3,
        reduction_dimensions: int = 2
    ) -> List[GenomeMapPoint]:
        """
        Generate Threat Radar from evaluation results.
        
        Args:
            evaluations: List of evaluation results
            min_cluster_size: Minimum size for clusters
            reduction_dimensions: Number of dimensions for reduction (2 or 3)
            
        Returns:
            List of threat radar points with specialized labeling
        """
        # Use parent class to generate base map
        genome_points = super().generate(evaluations, min_cluster_size, reduction_dimensions)
        
        # Enhance with threat-specific labels
        enhanced_points = []
        for point in genome_points:
            # Identify threat type
            threat_type = self._classify_threat_type(point)
            point.metadata["threat_type"] = threat_type
            point.metadata["threat_label"] = self._generate_threat_label(point, threat_type)
            
            enhanced_points.append(point)
        
        log.info(f"Generated Threat Radar with {len(enhanced_points)} threat clusters")
        return enhanced_points
    
    def _classify_threat_type(self, point: GenomeMapPoint) -> str:
        """
        Classify the type of threat represented by this cluster.
        
        Args:
            point: Genome map point
            
        Returns:
            Threat type: "pathogen_synthesis", "zero_day", "social_engineering", or "other"
        """
        # Check for biological threats
        if ViolationDomain.BIOLOGICAL in point.violation_domains:
            if AttackStrategy.BIO_HAZARD in point.attack_strategies:
                return "pathogen_synthesis"
        
        # Check for cyber threats
        if ViolationDomain.CYBER in point.violation_domains:
            if AttackStrategy.CYBER_EXPLOIT in point.attack_strategies:
                return "zero_day"
        
        # Check severity - high severity biological = pathogen synthesis
        if (ViolationDomain.BIOLOGICAL in point.violation_domains and 
            point.mean_severity >= 3.0):
            return "pathogen_synthesis"
        
        # High severity cyber = zero-day
        if (ViolationDomain.CYBER in point.violation_domains and 
            point.mean_severity >= 3.0):
            return "zero_day"
        
        # Social engineering
        social_strategies = [
            AttackStrategy.EMOTIONAL_COERCION,
            AttackStrategy.ROLEPLAY,
            AttackStrategy.CHAIN_OF_COMMAND
        ]
        if any(s in point.attack_strategies for s in social_strategies):
            return "social_engineering"
        
        return "other"
    
    def _generate_threat_label(
        self,
        point: GenomeMapPoint,
        threat_type: str
    ) -> str:
        """
        Generate a human-readable threat label.
        
        Args:
            point: Genome map point
            threat_type: Classified threat type
            
        Returns:
            Threat label string
        """
        if threat_type == "pathogen_synthesis":
            return f"Pathogen Synthesis Vector (Cluster {point.cluster_id})"
        elif threat_type == "zero_day":
            return f"Zero-Day Exploit Pattern (Cluster {point.cluster_id})"
        elif threat_type == "social_engineering":
            return f"Social Engineering Vector (Cluster {point.cluster_id})"
        else:
            return f"Threat Cluster {point.cluster_id}"
    
    def visualize(
        self,
        genome_points: List[GenomeMapPoint],
        output_path: Optional[str] = None,
        show_plot: bool = False,
        highlight_pathogen_vectors: bool = True,
        highlight_zero_day: bool = True
    ) -> None:
        """
        Visualize the Threat Radar with specialized highlighting.
        
        Args:
            genome_points: Threat radar points to visualize
            output_path: Optional path to save visualization
            show_plot: Whether to display the plot
            highlight_pathogen_vectors: Highlight biological threat clusters in bright red
            highlight_zero_day: Highlight cyber threat clusters
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.patches import Circle
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            if not genome_points:
                ax.text(0.5, 0.5, "No threat clusters detected", ha="center", va="center")
                plt.title("Threat Radar - No Threats Detected", fontsize=16, fontweight="bold")
                if output_path:
                    plt.savefig(output_path)
                if show_plot:
                    plt.show()
                return
            
            # Separate points by threat type
            pathogen_points = []
            zero_day_points = []
            other_points = []
            
            for point in genome_points:
                threat_type = point.metadata.get("threat_type", "other")
                if threat_type == "pathogen_synthesis":
                    pathogen_points.append(point)
                elif threat_type == "zero_day":
                    zero_day_points.append(point)
                else:
                    other_points.append(point)
            
            # Plot other threats first (background)
            if other_points:
                x_coords = [p.x for p in other_points]
                y_coords = [p.y for p in other_points]
                sizes = [p.size * 50 for p in other_points]
                severities = [p.mean_severity for p in other_points]
                
                ax.scatter(
                    x_coords, y_coords, s=sizes,
                    c=severities, cmap='YlOrRd',
                    alpha=0.4, edgecolors='gray',
                    linewidths=1, label="Other Threats"
                )
            
            # Plot zero-day clusters (cyber)
            if zero_day_points and highlight_zero_day:
                x_coords = [p.x for p in zero_day_points]
                y_coords = [p.y for p in zero_day_points]
                sizes = [p.size * 60 for p in zero_day_points]
                severities = [p.mean_severity for p in zero_day_points]
                
                ax.scatter(
                    x_coords, y_coords, s=sizes,
                    c=severities, cmap='Blues',
                    alpha=0.7, edgecolors='darkblue',
                    linewidths=2, marker='^',
                    label="Zero-Day Exploit Patterns"
                )
            
            # Plot pathogen synthesis vectors (bright red - most critical)
            if pathogen_points and highlight_pathogen_vectors:
                x_coords = [p.x for p in pathogen_points]
                y_coords = [p.y for p in pathogen_points]
                sizes = [p.size * 80 for p in pathogen_points]  # Larger for visibility
                severities = [p.mean_severity for p in pathogen_points]
                
                scatter = ax.scatter(
                    x_coords, y_coords, s=sizes,
                    c=severities, cmap='Reds',
                    alpha=0.9, edgecolors='darkred',
                    linewidths=3, marker='*',
                    label="Pathogen Synthesis Vectors"
                )
                
                # Add bright red circles around pathogen vectors
                for point in pathogen_points:
                    circle = Circle(
                        (point.x, point.y),
                        radius=0.5,
                        fill=False,
                        edgecolor='red',
                        linewidth=2,
                        linestyle='--',
                        alpha=0.6
                    )
                    ax.add_patch(circle)
            
            # Add colorbar
            if pathogen_points:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Threat Severity', rotation=270, labelpad=20)
            
            # Annotate critical clusters
            for point in genome_points:
                threat_type = point.metadata.get("threat_type", "other")
                if threat_type in ["pathogen_synthesis", "zero_day"]:
                    label = point.metadata.get("threat_label", f"C{point.cluster_id}")
                    ax.annotate(
                        label,
                        (point.x, point.y),
                        fontsize=9,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
                    )
                else:
                    ax.annotate(
                        f"C{point.cluster_id}",
                        (point.x, point.y),
                        fontsize=7,
                        ha="center"
                    )
            
            ax.set_xlabel("Threat Radar Dimension 1", fontsize=12)
            ax.set_ylabel("Threat Radar Dimension 2", fontsize=12)
            ax.set_title(
                "Threat Radar - Dual-Use Capability Detection",
                fontsize=16, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                log.info(f"Saved Threat Radar visualization to {output_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        except ImportError:
            log.error("Matplotlib not available for visualization")
        except Exception as e:
            log.error(f"Error visualizing Threat Radar: {e}")
    
    def get_pathogen_vectors(self, genome_points: List[GenomeMapPoint]) -> List[GenomeMapPoint]:
        """Extract only pathogen synthesis vectors from radar."""
        return [
            p for p in genome_points
            if p.metadata.get("threat_type") == "pathogen_synthesis"
        ]
    
    def get_zero_day_patterns(self, genome_points: List[GenomeMapPoint]) -> List[GenomeMapPoint]:
        """Extract only zero-day exploit patterns from radar."""
        return [
            p for p in genome_points
            if p.metadata.get("threat_type") == "zero_day"
        ]
    
    def generate_attack_fingerprints(
        self,
        genome_points: List[GenomeMapPoint]
    ) -> Dict[str, Any]:
        """
        Generate attack pattern fingerprints for model vaccination.
        
        Args:
            genome_points: Threat radar points
            
        Returns:
            Dictionary of attack fingerprints that can be used to vaccinate other models
        """
        fingerprints = {
            "pathogen_synthesis": [],
            "zero_day": [],
            "social_engineering": [],
            "other": []
        }
        
        for point in genome_points:
            threat_type = point.metadata.get("threat_type", "other")
            fingerprint = {
                "cluster_id": point.cluster_id,
                "representative_prompts": point.representative_prompts,
                "representative_responses": point.representative_responses,
                "mean_severity": point.mean_severity,
                "violation_domains": [d.value for d in point.violation_domains],
                "attack_strategies": [s.value for s in point.attack_strategies],
                "size": point.size
            }
            fingerprints[threat_type].append(fingerprint)
        
        log.info(f"Generated attack fingerprints: {sum(len(v) for v in fingerprints.values())} total")
        return fingerprints

