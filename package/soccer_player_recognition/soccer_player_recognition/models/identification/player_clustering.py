"""
Player Clustering Module for Soccer Player Identification

This module provides clustering algorithms for grouping players based on their
visual embeddings, enabling team clustering, similar player grouping, and
anomaly detection in player datasets.

Supports multiple clustering algorithms:
- K-means clustering
- DBSCAN clustering  
- Hierarchical clustering
- Agglomerative clustering
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import logging
from PIL import Image
import os
import json
import pickle

from .siglip_model import SigLIPPlayerIdentification, create_siglip_model
from ....utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for player clustering"""
    algorithm: str = "kmeans"  # kmeans, dbscan, hierarchical, agglomerative
    n_clusters: Optional[int] = None
    max_clusters: int = 20
    min_cluster_size: int = 2
    similarity_threshold: float = 0.8
    eps: float = 0.5  # For DBSCAN
    min_samples: int = 5  # For DBSCAN
    n_components: int = 2  # For PCA
    random_state: int = 42


class PlayerClusterer:
    """Main class for player clustering based on visual embeddings"""
    
    def __init__(self, config: ClusteringConfig = None, 
                 embedding_model: SigLIPPlayerIdentification = None):
        """
        Initialize player clusterer
        
        Args:
            config: Clustering configuration
            embedding_model: Pre-trained SigLIP model for embeddings
        """
        self.config = config or ClusteringConfig()
        self.embedding_model = embedding_model or create_siglip_model()
        self.scaler = StandardScaler()
        
        # Clustering results
        self.embeddings = None
        self.labels = None
        self.cluster_centers = None
        self.similarity_matrix = None
        self.silhouette_score = None
        self.calinski_harabasz_score = None
        self.davies_bouldin_score = None
        
    def extract_player_embeddings(self, player_images: Dict[str, List[str]], 
                                 team_context: str = None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for players from their images
        
        Args:
            player_images: Dictionary mapping player names to list of image paths
            team_context: Optional team context for text prompts
            
        Returns:
            Dictionary mapping player names to their average embeddings
        """
        player_embeddings = {}
        
        for player_name, image_paths in player_images.items():
            embeddings_list = []
            
            for image_path in image_paths:
                try:
                    # Preprocess image
                    image_tensor = self.embedding_model.preprocess_image(image_path)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = self.embedding_model.model.encode_image(image_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embeddings_list.append(embedding.cpu().numpy())
                        
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    continue
            
            if embeddings_list:
                # Average embeddings across all images of this player
                avg_embedding = np.mean(embeddings_list, axis=0)
                player_embeddings[player_name] = avg_embedding.flatten()
                logger.info(f"Extracted embedding for {player_name} from {len(embeddings_list)} images")
            else:
                logger.warning(f"No valid embeddings found for {player_name}")
        
        return player_embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray, 
                                player_names: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between player embeddings
        
        Args:
            embeddings: Embedding matrix (n_players x embedding_dim)
            player_names: List of player names
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Convert to distance matrix (1 - similarity for easier clustering)
        distance_matrix = 1 - similarity_matrix
        
        self.similarity_matrix = similarity_matrix
        
        logger.info(f"Computed similarity matrix for {len(player_names)} players")
        return similarity_matrix
    
    def find_optimal_clusters(self, embeddings: np.ndarray, 
                             method: str = "silhouette") -> int:
        """
        Find optimal number of clusters using various methods
        
        Args:
            embeddings: Embedding matrix
            method: Method for finding optimal clusters (silhouette, elbow, calinski_harabasz)
            
        Returns:
            Optimal number of clusters
        """
        if self.config.algorithm == "dbscan":
            logger.info("DBSCAN doesn't require predefined number of clusters")
            return self.config.n_clusters or 0
        
        max_clusters = min(self.config.max_clusters, len(embeddings) - 1)
        
        if method == "silhouette":
            scores = []
            cluster_range = range(2, max_clusters + 1)
            
            for n_clusters in cluster_range:
                if self.config.algorithm == "kmeans":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                elif self.config.algorithm == "agglomerative":
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    continue
                
                labels = clusterer.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                scores.append(score)
            
            optimal_clusters = cluster_range[np.argmax(scores)]
            
        elif method == "calinski_harabasz":
            scores = []
            cluster_range = range(2, max_clusters + 1)
            
            for n_clusters in cluster_range:
                if self.config.algorithm == "kmeans":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                elif self.config.algorithm == "agglomerative":
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    continue
                
                labels = clusterer.fit_predict(embeddings)
                score = calinski_harabasz_score(embeddings, labels)
                scores.append(score)
            
            optimal_clusters = cluster_range[np.argmax(scores)]
            
        elif method == "elbow":
            # Use elbow method for K-means
            inertias = []
            cluster_range = range(1, max_clusters + 1)
            
            for n_clusters in cluster_range:
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
                clusterer.fit(embeddings)
                inertias.append(clusterer.inertia_)
            
            # Find elbow point (simplified)
            optimal_clusters = cluster_range[np.argmax(np.diff(np.diff(inertias))) + 2]
            optimal_clusters = min(optimal_clusters, max_clusters)
            
        else:
            optimal_clusters = self.config.n_clusters or 4
            logger.warning(f"Unknown method {method}, using default: {optimal_clusters}")
        
        logger.info(f"Optimal number of clusters: {optimal_clusters} (method: {method})")
        return optimal_clusters
    
    def cluster_players(self, player_embeddings: Dict[str, np.ndarray], 
                       player_names: List[str] = None) -> Dict[str, Any]:
        """
        Cluster players based on their embeddings
        
        Args:
            player_embeddings: Dictionary mapping player names to embeddings
            player_names: List of player names (optional, defaults to keys of player_embeddings)
            
        Returns:
            Clustering results
        """
        if player_names is None:
            player_names = list(player_embeddings.keys())
        
        # Convert to embedding matrix
        embeddings = np.array([player_embeddings[name] for name in player_names])
        
        logger.info(f"Clustering {len(player_names)} players with algorithm: {self.config.algorithm}")
        
        # Scale embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Find optimal number of clusters if not specified
        if self.config.n_clusters is None:
            self.config.n_clusters = self.find_optimal_clusters(embeddings_scaled)
        
        # Apply clustering algorithm
        if self.config.algorithm == "kmeans":
            clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state)
            labels = clusterer.fit_predict(embeddings_scaled)
            self.cluster_centers = clusterer.cluster_centers_
            
        elif self.config.algorithm == "dbscan":
            clusterer = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
            labels = clusterer.fit_predict(embeddings_scaled)
            # DBSCAN might identify noise points (-1)
            
        elif self.config.algorithm == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=self.config.n_clusters)
            labels = clusterer.fit_predict(embeddings_scaled)
            
        elif self.config.algorithm == "hierarchical":
            # Use scipy for hierarchical clustering
            condensed_distance = pdist(embeddings_scaled, metric='euclidean')
            linkage_matrix = linkage(condensed_distance, method='ward')
            labels = np.arange(len(embeddings_scaled))
            
            # Cut dendrogram to get specified number of clusters
            from scipy.cluster.hierarchy import fcluster
            labels = fcluster(linkage_matrix, self.config.n_clusters, criterion='maxclust') - 1
            
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.config.algorithm}")
        
        # Store results
        self.embeddings = embeddings_scaled
        self.labels = labels
        
        # Filter out noise points if using DBSCAN
        valid_indices = labels != -1 if self.config.algorithm == "dbscan" else labels >= 0
        
        # Compute clustering quality metrics
        if len(np.unique(labels[valid_indices])) > 1:
            self.silhouette_score = silhouette_score(embeddings_scaled[valid_indices], labels[valid_indices])
            self.calinski_harabasz_score = calinski_harabasz_score(embeddings_scaled[valid_indices], labels[valid_indices])
            self.davies_bouldin_score = davies_bouldin_score(embeddings_scaled[valid_indices], labels[valid_indices])
        
        # Compute similarity matrix
        self.compute_similarity_matrix(embeddings, player_names)
        
        # Format results
        results = self._format_clustering_results(player_names, labels)
        
        logger.info(f"Clustering completed. Found {len(np.unique(labels[valid_indices]))} clusters")
        logger.info(f"Silhouette Score: {self.silhouette_score:.3f}")
        
        return results
    
    def _format_clustering_results(self, player_names: List[str], 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Format clustering results into dictionary"""
        
        # Create player-to-cluster mapping
        player_clusters = {}
        cluster_players = {}
        
        for player, label in zip(player_names, labels):
            player_clusters[player] = int(label)
            if label not in cluster_players:
                cluster_players[label] = []
            cluster_players[label].append(player)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, players in cluster_players.items():
            cluster_stats[int(cluster_id)] = {
                'players': players,
                'size': len(players),
                'quality_score': 0.0
            }
            
            # Calculate intra-cluster similarity
            if len(players) > 1:
                cluster_indices = [i for i, p in enumerate(player_names) if p in players]
                cluster_similarities = []
                for i in cluster_indices:
                    for j in cluster_indices[i+1:], cluster_indices:
                        if i < j:
                            sim = self.similarity_matrix[i, j]
                            cluster_similarities.append(sim)
                
                if cluster_similarities:
                    avg_similarity = np.mean(cluster_similarities)
                    cluster_stats[int(cluster_id)]['quality_score'] = avg_similarity
        
        results = {
            'player_clusters': player_clusters,
            'cluster_players': cluster_players,
            'cluster_stats': cluster_stats,
            'metrics': {
                'silhouette_score': self.silhouette_score,
                'calinski_harabasz_score': self.calinski_harabasz_score,
                'davies_bouldin_score': self.davies_bouldin_score
            },
            'n_clusters': len(np.unique(labels[labels >= 0])),
            'noise_points': np.sum(labels == -1) if self.config.algorithm == "dbscan" else 0
        }
        
        return results
    
    def find_similar_players(self, target_player: str, 
                           player_embeddings: Dict[str, np.ndarray],
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find players most similar to a target player
        
        Args:
            target_player: Name of target player
            player_embeddings: Dictionary of player embeddings
            top_k: Number of similar players to return
            
        Returns:
            List of similar players with similarity scores
        """
        if target_player not in player_embeddings:
            raise ValueError(f"Player {target_player} not found in embeddings")
        
        target_embedding = player_embeddings[target_player]
        similarities = []
        
        for player_name, embedding in player_embeddings.items():
            if player_name != target_player:
                # Compute cosine similarity
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append({
                    'player': player_name,
                    'similarity': float(similarity)
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {min(top_k, len(similarities))} similar players to {target_player}")
        return similarities[:top_k]
    
    def detect_anomalies(self, player_embeddings: Dict[str, np.ndarray],
                        threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect anomalous players using clustering
        
        Args:
            player_embeddings: Dictionary of player embeddings
            threshold: Distance threshold for anomaly detection
            
        Returns:
            List of anomalous players
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        embeddings = np.array(list(player_embeddings.values()))
        player_names = list(player_embeddings.keys())
        
        if len(embeddings) < 3:
            logger.warning("Not enough players for anomaly detection")
            return []
        
        # Use DBSCAN for anomaly detection
        clusterer = DBSCAN(eps=1-threshold, min_samples=2)
        labels = clusterer.fit_predict(embeddings)
        
        anomalies = []
        for i, (player, label) in enumerate(zip(player_names, labels)):
            if label == -1:  # Noise points are anomalies
                # Find distance to nearest cluster center
                distances_to_centers = []
                for cluster_id in np.unique(labels[labels != -1]):
                    cluster_mask = labels == cluster_id
                    if np.any(cluster_mask):
                        cluster_center = np.mean(embeddings[cluster_mask], axis=0)
                        distance = np.linalg.norm(embeddings[i] - cluster_center)
                        distances_to_centers.append(distance)
                
                min_distance = min(distances_to_centers) if distances_to_centers else float('inf')
                
                anomalies.append({
                    'player': player,
                    'anomaly_score': float(min_distance),
                    'cluster': -1
                })
        
        # Sort by anomaly score
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        logger.info(f"Detected {len(anomalies)} anomalous players")
        return anomalies
    
    def visualize_clusters(self, player_names: List[str], 
                          output_path: str = None, 
                          method: str = "pca") -> str:
        """
        Create visualization of player clusters
        
        Args:
            player_names: List of player names
            output_path: Path to save visualization
            method: Dimensionality reduction method (pca, tsne)
            
        Returns:
            Path to saved visualization
        """
        if self.embeddings is None or self.labels is None:
            raise ValueError("No clustering results to visualize")
        
        if output_path is None:
            output_path = f"player_clusters_{method}_{self.config.algorithm}.png"
        
        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=self.config.random_state)
            coords_2d = reducer.fit_transform(self.embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            title = f"Player Clusters (PCA, explained variance: {explained_var:.2%})"
            
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=self.config.random_state, perplexity=min(30, len(player_names)-1))
            coords_2d = reducer.fit_transform(self.embeddings)
            title = "Player Clusters (t-SNE)"
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot points
        unique_labels = np.unique(self.labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            if label == -1:
                label_name = "Noise"
                alpha = 0.5
                marker = 'x'
            else:
                label_name = f"Cluster {label}"
                alpha = 0.8
                marker = 'o'
            
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=alpha, marker=marker, s=50)
        
        # Add player name annotations
        for i, player in enumerate(player_names):
            plt.annotate(player, (coords_2d[i, 0], coords_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cluster visualization to {output_path}")
        return output_path
    
    def save_clustering_results(self, results: Dict[str, Any], 
                               output_path: str):
        """
        Save clustering results to file
        
        Args:
            results: Clustering results dictionary
            output_path: Path to save results
        """
        # Prepare serializable results
        serializable_results = {
            'config': self.config.__dict__,
            'metrics': results['metrics'],
            'n_clusters': results['n_clusters'],
            'noise_points': results['noise_points'],
            'player_clusters': results['player_clusters'],
            'cluster_stats': results['cluster_stats']
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved clustering results to {output_path}")
    
    def load_clustering_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load clustering results from file
        
        Args:
            input_path: Path to clustering results file
            
        Returns:
            Clustering results dictionary
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded clustering results from {input_path}")
        return results


class TeamClusterer(PlayerClusterer):
    """Specialized clusterer for team-based player grouping"""
    
    def cluster_by_team(self, player_images: Dict[str, List[str]], 
                       team_contexts: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Cluster players primarily by team association
        
        Args:
            player_images: Dictionary mapping player names to image paths
            team_contexts: Dictionary mapping player names to team contexts
            
        Returns:
            Clustering results with team information
        """
        # Extract embeddings with team context
        player_embeddings = self.extract_player_embeddings(
            player_images, team_contexts=team_contexts
        )
        
        # Perform clustering
        results = self.cluster_players(player_embeddings)
        
        # Analyze teams
        teams_analysis = self._analyze_teams(results, team_contexts)
        results['team_analysis'] = teams_analysis
        
        return results
    
    def _analyze_teams(self, clustering_results: Dict[str, Any], 
                      team_contexts: Dict[str, str] = None) -> Dict[str, Any]:
        """Analyze clustering results by teams"""
        
        if team_contexts is None:
            return {}
        
        player_clusters = clustering_results['player_clusters']
        
        # Group players by cluster and team
        cluster_team_analysis = {}
        
        for cluster_id, players in clustering_results['cluster_players'].items():
            team_counts = {}
            team_players = {}
            
            for player in players:
                if player in team_contexts:
                    team = team_contexts[player]
                    if team not in team_counts:
                        team_counts[team] = 0
                        team_players[team] = []
                    team_counts[team] += 1
                    team_players[team].append(player)
            
            # Determine dominant team in cluster
            dominant_team = max(team_counts.items(), key=lambda x: x[1]) if team_counts else None
            purity = dominant_team[1] / len(players) if dominant_team else 0.0
            
            cluster_team_analysis[cluster_id] = {
                'team_counts': team_counts,
                'dominant_team': dominant_team[0] if dominant_team else None,
                'dominant_team_count': dominant_team[1] if dominant_team else 0,
                'purity': purity,
                'team_players': team_players
            }
        
        return cluster_team_analysis


def create_player_clusterer(algorithm: str = "kmeans", 
                          config: ClusteringConfig = None,
                          embedding_model: SigLIPPlayerIdentification = None) -> PlayerClusterer:
    """
    Factory function to create player clusterer
    
    Args:
        algorithm: Clustering algorithm
        config: Clustering configuration
        embedding_model: Pre-trained embedding model
        
    Returns:
        PlayerClusterer instance
    """
    if config is None:
        config = ClusteringConfig(algorithm=algorithm)
    else:
        config.algorithm = algorithm
    
    return PlayerClusterer(config=config, embedding_model=embedding_model)


def create_team_clusterer(algorithm: str = "kmeans", 
                         config: ClusteringConfig = None,
                         embedding_model: SigLIPPlayerIdentification = None) -> TeamClusterer:
    """
    Factory function to create team clusterer
    
    Args:
        algorithm: Clustering algorithm
        config: Clustering configuration
        embedding_model: Pre-trained embedding model
        
    Returns:
        TeamClusterer instance
    """
    if config is None:
        config = ClusteringConfig(algorithm=algorithm)
    else:
        config.algorithm = algorithm
    
    return TeamClusterer(config=config, embedding_model=embedding_model)


if __name__ == "__main__":
    # Example usage
    config = ClusteringConfig(algorithm="kmeans", n_clusters=4)
    clusterer = create_player_clusterer(algorithm="kmeans", config=config)
    
    # Example player embeddings (would come from actual extraction)
    sample_embeddings = {
        "Messi": np.random.randn(768),
        "Ronaldo": np.random.randn(768),
        "Neymar": np.random.randn(768),
        "Mbappé": np.random.randn(768),
        "Haaland": np.random.randn(768),
        "Lewandowski": np.random.randn(768)
    }
    
    # Perform clustering
    results = clusterer.cluster_players(sample_embeddings)
    print(f"Found {results['n_clusters']} clusters")
    
    # Visualize clusters
    player_names = list(sample_embeddings.keys())
    clusterer.visualize_clusters(player_names, "cluster_visualization.png")