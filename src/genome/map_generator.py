"""Genome Map generation through embedding and clustering."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from src.models.jailbreak import EvaluationResult, GenomeMapPoint, ViolationDomain, AttackStrategy
from src.vector_db.embedder import Embedder
from src.utils.logger import log
from collections import defaultdict

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    log.warning("UMAP not available, using PCA for dimensionality reduction")


class GenomeMapGenerator:
    """Generates Jailbreak Genome Maps through embedding and clustering."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the genome map generator.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedder = Embedder(model_name=embedding_model)
        self.use_umap = UMAP_AVAILABLE
        
        log.info(f"Initialized genome map generator (UMAP: {self.use_umap})")
    
    def generate(
        self,
        evaluations: List[EvaluationResult],
        min_cluster_size: int = 3,
        reduction_dimensions: int = 2
    ) -> List[GenomeMapPoint]:
        """
        Generate genome map from evaluation results.
        
        Args:
            evaluations: List of evaluation results (should include successful exploits)
            min_cluster_size: Minimum size for clusters
            reduction_dimensions: Number of dimensions for reduction (2 or 3)
            
        Returns:
            List of genome map points (one per cluster)
        """
        if not evaluations:
            log.warning("No evaluations provided for genome map generation")
            return []
        
        # Filter to only successful jailbreaks
        jailbroken = [e for e in evaluations if e.is_jailbroken]
        
        if not jailbroken:
            log.warning("No successful jailbreaks found for genome map")
            return []
        
        log.info(f"Generating genome map from {len(jailbroken)} jailbreak evaluations")
        
        # Generate embeddings for responses
        responses = [e.response for e in jailbroken]
        embeddings = self.embedder.embed_texts(responses)
        
        # Store embeddings in evaluation results
        for eval, embedding in zip(jailbroken, embeddings):
            eval.embedding = embedding
        
        # Reduce dimensionality
        reduced_embeddings = self._reduce_dimensions(
            np.array(embeddings),
            n_dimensions=reduction_dimensions
        )
        
        # Cluster failures
        clusters = self._cluster_failures(
            reduced_embeddings,
            min_cluster_size=min_cluster_size
        )
        
        # Assign cluster IDs to evaluations
        for i, cluster_id in enumerate(clusters):
            if i < len(jailbroken):
                jailbroken[i].cluster_id = cluster_id
        
        # Generate genome map points
        genome_points = self._create_genome_points(
            jailbroken,
            reduced_embeddings,
            clusters
        )
        
        log.info(f"Generated genome map with {len(genome_points)} failure clusters")
        return genome_points
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_dimensions: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            n_dimensions: Target dimensions (2 or 3)
            
        Returns:
            Reduced-dimension embeddings
        """
        min_cluster_size = 3  # Default minimum cluster size
        
        if embeddings.shape[0] < n_dimensions:
            # Not enough samples, pad with zeros
            log.warning(f"Not enough samples for {n_dimensions}D reduction")
            padding = np.zeros((n_dimensions - embeddings.shape[0], embeddings.shape[1]))
            embeddings = np.vstack([embeddings, padding])
        
        if self.use_umap and embeddings.shape[0] >= min_cluster_size:
            # Use UMAP for better clustering structure preservation
            reducer = umap.UMAP(n_components=n_dimensions, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            # Fall back to PCA
            reducer = PCA(n_components=n_dimensions, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        
        return reduced
    
    def _cluster_failures(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 3
    ) -> List[int]:
        """
        Cluster failure patterns.
        
        Args:
            embeddings: Reduced-dimension embeddings
            min_cluster_size: Minimum cluster size
            
        Returns:
            Cluster assignments (-1 for noise/outliers)
        """
        if embeddings.shape[0] < min_cluster_size:
            # Not enough points for clustering
            return [0] * embeddings.shape[0]
        
        # Use DBSCAN for automatic cluster detection
        clustering = DBSCAN(
            eps=0.5,
            min_samples=min_cluster_size,
            metric='euclidean'
        ).fit(embeddings)
        
        cluster_labels = clustering.labels_.tolist()
        
        # Count clusters
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Exclude noise
        
        log.info(f"Identified {len(unique_clusters)} failure clusters (+ {cluster_labels.count(-1)} outliers)")
        
        return cluster_labels
    
    def _create_genome_points(
        self,
        evaluations: List[EvaluationResult],
        reduced_embeddings: np.ndarray,
        cluster_labels: List[int]
    ) -> List[GenomeMapPoint]:
        """
        Create genome map points from clustered evaluations.
        
        Args:
            evaluations: Evaluation results
            reduced_embeddings: Reduced-dimension embeddings
            cluster_labels: Cluster assignments
            
        Returns:
            List of genome map points (one per cluster)
        """
        # Group evaluations by cluster
        clusters: Dict[int, List[EvaluationResult]] = defaultdict(list)
        for eval, cluster_id in zip(evaluations, cluster_labels):
            clusters[cluster_id].append(eval)
        
        genome_points = []
        
        for cluster_id, cluster_evals in clusters.items():
            if cluster_id == -1:
                # Skip noise/outliers for now (could be added separately)
                continue
            
            if not cluster_evals:
                continue
            
            # Calculate cluster center
            cluster_indices = [
                i for i, label in enumerate(cluster_labels)
                if label == cluster_id
            ]
            cluster_embeddings = reduced_embeddings[cluster_indices]
            center = np.mean(cluster_embeddings, axis=0)
            
            # Collect statistics
            severities = [e.severity.value for e in cluster_evals]
            mean_severity = np.mean(severities) if severities else 0.0
            
            # Collect unique violation domains
            all_domains = []
            for eval in cluster_evals:
                all_domains.extend(eval.violation_domains)
            unique_domains = list(set(all_domains))
            
            # Collect attack strategies
            strategies = list(set(e.attack_strategy for e in cluster_evals))
            
            # Get representative examples (top severity)
            sorted_evals = sorted(
                cluster_evals,
                key=lambda e: e.severity.value,
                reverse=True
            )
            representative_count = min(3, len(sorted_evals))
            examples = sorted_evals[:representative_count]
            
            point = GenomeMapPoint(
                cluster_id=cluster_id,
                x=float(center[0]),
                y=float(center[1]) if len(center) > 1 else 0.0,
                size=len(cluster_evals),
                mean_severity=mean_severity,
                violation_domains=unique_domains,
                attack_strategies=strategies,
                representative_prompts=[e.prompt[:200] for e in examples],
                representative_responses=[e.response[:200] for e in examples]
            )
            
            genome_points.append(point)
        
        return genome_points
    
    def visualize(
        self,
        genome_points: List[GenomeMapPoint],
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> None:
        """
        Visualize the genome map.
        
        Args:
            genome_points: Genome map points to visualize
            output_path: Optional path to save visualization
            show_plot: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if not genome_points:
                ax.text(0.5, 0.5, "No failure clusters found", ha="center", va="center")
                plt.title("Jailbreak Genome Map")
                if output_path:
                    plt.savefig(output_path)
                if show_plot:
                    plt.show()
                return
            
            # Extract coordinates and sizes
            x_coords = [p.x for p in genome_points]
            y_coords = [p.y for p in genome_points]
            sizes = [p.size * 50 for p in genome_points]  # Scale for visibility
            severities = [p.mean_severity for p in genome_points]
            
            # Create scatter plot colored by severity
            scatter = ax.scatter(
                x_coords,
                y_coords,
                s=sizes,
                c=severities,
                cmap='Reds',
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mean Severity', rotation=270, labelpad=20)
            
            # Annotate clusters
            for point in genome_points:
                ax.annotate(
                    f"C{point.cluster_id}",
                    (point.x, point.y),
                    fontsize=8,
                    ha="center"
                )
            
            ax.set_xlabel("Genome Dimension 1", fontsize=12)
            ax.set_ylabel("Genome Dimension 2", fontsize=12)
            ax.set_title("Jailbreak Genome Map - Failure Clusters", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                log.info(f"Saved genome map visualization to {output_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        except ImportError:
            log.error("Matplotlib not available for visualization")
        except Exception as e:
            log.error(f"Error visualizing genome map: {e}")

