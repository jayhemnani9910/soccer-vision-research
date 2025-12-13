"""
SigLIP Utilities for Text Processing and Similarity Search

This module provides utility functions for text prompt processing, similarity search,
player name normalization, team context handling, and embedding operations
used in the SigLIP-based player identification system.
"""

import re
import string
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import difflib
import json
import os
from scipy.spatial.distance import cdist, pdist
from scipy.stats import cosine
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlayerName:
    """Represents a normalized player name"""
    full_name: str
    first_name: str
    last_name: str
    common_names: List[str]
    aliases: List[str]
    team: Optional[str] = None
    position: Optional[str] = None
    nationality: Optional[str] = None


@dataclass
class SimilarityResult:
    """Result of similarity search"""
    query: str
    target: str
    similarity: float
    distance: float
    rank: int


class PlayerNameNormalizer:
    """Normalize and standardize player names for consistent matching"""
    
    def __init__(self):
        # Common name variations and abbreviations
        self.name_mappings = {
            'lionel messi': ['lionel andres messi', 'lionel messi', 'messi'],
            'cristiano ronaldo': ['cristiano ronaldo dos santos aveiro', 'cristiano ronaldo', 'c ronaldo', 'ronaldo'],
            'neymar jr': ['neymar da silva santos júnior', 'neymar jr', 'neymar', 'neymar jr.'],
            'kylian mbappé': ['kylian mbappé', 'k mbappé', 'mbappé'],
            'erling haaland': ['erling braut haaland', 'erling haaland', 'e haaland', 'haaland'],
            'robert lewandowski': ['robert lewandowski', 'r lewandowski', 'lewandowski'],
            'harry kane': ['harry edward kane', 'harry kane', 'h kane', 'kane'],
            'zlatan ibrahimović': ['zlatan ibrahimovic', 'zlatan ibrahimović', 'z ibrahimovic', 'ibrahimovic'],
            'sadio mané': ['sadio mane', 'sadio mané', 's mane', 'mane'],
            'vinicius jr': ['vinicius jose Paixao de Oliveira Junior', 'vinicius jr', 'vinicius jr.', 'vinicius'],
            'kevin de bruyne': ['kevin de bruyne', 'k de bruyne', 'de bruyne'],
        }
        
        # Common name suffixes
        self.suffixes = {
            'jr': ['jr', 'jr.', 'junior'],
            'sr': ['sr', 'sr.', 'senior'],
            'iii': ['iii', '3rd'],
            'ii': ['ii', '2nd'],
            'iv': ['iv', '4th'],
        }
        
        # Remove common prefixes
        self.prefixes = ['mr.', 'mrs.', 'dr.', 'prof.', 'sir']
        
    def normalize_name(self, name: str) -> str:
        """
        Normalize player name for consistent matching
        
        Args:
            name: Raw player name
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove prefixes
        for prefix in self.prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s\-\']', '', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def extract_player_name(self, text: str) -> Optional[PlayerName]:
        """
        Extract player name from text
        
        Args:
            text: Text containing player name
            
        Returns:
            PlayerName object or None
        """
        text = self.normalize_name(text)
        
        # Try to match known player names
        for canonical_name, variations in self.name_mappings.items():
            if text == canonical_name or text in variations:
                parts = canonical_name.split()
                return PlayerName(
                    full_name=canonical_name.title(),
                    first_name=parts[0].title(),
                    last_name=parts[-1].title(),
                    common_names=[canonical_name.title()],
                    aliases=variations
                )
        
        # Extract name from patterns
        # Look for "player named X" or "X player" patterns
        patterns = [
            r'player named ([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+) player',
            r'player ([a-zA-Z\s]+)',
            r'([a-zA-Z\s]{2,30})',  # General name pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name_part = match.group(1).strip()
                name_part = self.normalize_name(name_part)
                
                # Split into parts
                parts = name_part.split()
                if len(parts) >= 2:
                    return PlayerName(
                        full_name=name_part.title(),
                        first_name=parts[0].title(),
                        last_name=parts[-1].title(),
                        common_names=[name_part.title()],
                        aliases=[name_part]
                    )
        
        return None
    
    def find_similar_names(self, query: str, candidate_names: List[str], 
                          threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find similar names using fuzzy string matching
        
        Args:
            query: Query name
            candidate_names: List of candidate names
            threshold: Similarity threshold
            
        Returns:
            List of (name, similarity) tuples
        """
        query_normalized = self.normalize_name(query)
        similarities = []
        
        for candidate in candidate_names:
            candidate_normalized = self.normalize_name(candidate)
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, query_normalized, candidate_normalized).ratio()
            
            # Also check for substring matches
            if query_normalized in candidate_normalized or candidate_normalized in query_normalized:
                similarity = max(similarity, 0.7)
            
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


class TextPromptProcessor:
    """Process and generate text prompts for SigLIP model"""
    
    def __init__(self):
        self.team_contexts = {
            'barcelona': 'blue and red Barcelona jersey',
            'real madrid': 'white Real Madrid jersey', 
            'manchester united': 'red Manchester United jersey',
            'manchester city': 'blue Manchester City jersey',
            'liverpool': 'red Liverpool jersey',
            'chelsea': 'blue Chelsea jersey',
            'arsenal': 'red Arsenal jersey',
            'tottenham': 'white Tottenham jersey',
            'juventus': 'black and white Juventus jersey',
            'ac milan': 'red and black AC Milan jersey',
            'inter milan': 'blue and black Inter Milan jersey',
            'bayern munich': 'red Bayern Munich jersey',
            'psg': 'blue Paris Saint-Germain jersey',
            'atletico madrid': 'red and white Atletico Madrid jersey',
        }
        
        self.position_contexts = {
            'forward': 'attacking player',
            'winger': 'wide attacking player',
            'striker': 'central attacking player',
            'midfielder': 'midfield player',
            'defender': 'defensive player',
            'goalkeeper': 'goalkeeper',
            'center back': 'central defender',
            'left back': 'left-side defender',
            'right back': 'right-side defender',
            'attacking midfielder': 'advanced midfielder',
            'defensive midfielder': 'holding midfielder',
        }
    
    def create_player_prompt(self, player_name: str, 
                           team_context: str = None,
                           position_context: str = None,
                           additional_context: str = None) -> str:
        """
        Create detailed prompt for player identification
        
        Args:
            player_name: Player name
            team_context: Team or jersey colors
            position_context: Player position
            additional_context: Additional descriptive context
            
        Returns:
            Generated prompt
        """
        base_prompt = f"a soccer player named {player_name}"
        
        contexts = []
        
        # Add team context
        if team_context:
            team_lower = team_context.lower()
            if team_lower in self.team_contexts:
                contexts.append(self.team_contexts[team_lower])
            else:
                contexts.append(f"{team_context} team jersey")
        
        # Add position context
        if position_context:
            position_lower = position_context.lower()
            if position_lower in self.position_contexts:
                contexts.append(self.position_contexts[position_lower])
            else:
                contexts.append(f"{position_context}")
        
        # Add additional context
        if additional_context:
            contexts.append(additional_context)
        
        if contexts:
            prompt = f"{base_prompt} wearing " + " and ".join(contexts)
        else:
            prompt = base_prompt
        
        return prompt
    
    def create_team_prompt(self, team_name: str, 
                          players: List[str] = None) -> str:
        """
        Create prompt for team identification
        
        Args:
            team_name: Team name
            players: Optional list of players
            
        Returns:
            Team prompt
        """
        base_prompt = f"a soccer team called {team_name}"
        
        if players:
            player_list = ", ".join(players[:3])  # Limit to first 3 players
            if len(players) > 3:
                player_list += " and others"
            prompt = f"{base_prompt} with players like {player_list}"
        else:
            prompt = base_prompt
        
        return prompt
    
    def generate_variations(self, player_name: str, 
                          num_variations: int = 5) -> List[str]:
        """
        Generate variations of player prompts
        
        Args:
            player_name: Player name
            num_variations: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        variations = [
            f"a soccer player named {player_name}",
            f"soccer player {player_name}",
            f"footballer {player_name}",
            f"a {player_name} playing soccer",
            f"professional soccer player {player_name}",
            f"an athlete named {player_name}",
            f"player {player_name} on the field",
            f"{player_name} in soccer uniform",
        ]
        
        # Add team-specific variations
        team_variations = [
            f"{player_name} wearing team uniform",
            f"{player_name} in soccer gear",
            f"{player_name} during match",
            f"{player_name} on soccer field",
            f"{player_name} with soccer ball",
        ]
        
        variations.extend(team_variations)
        
        return variations[:num_variations]
    
    def extract_team_from_text(self, text: str) -> Optional[str]:
        """
        Extract team name from text
        
        Args:
            text: Text containing team information
            
        Returns:
            Team name or None
        """
        text_lower = text.lower()
        
        # Check for known teams
        for team in self.team_contexts.keys():
            if team in text_lower:
                return team.title()
        
        # Look for common team patterns
        patterns = [
            r'team ([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+) team',
            r'from ([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+) football club',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                team_name = match.group(1).strip()
                if len(team_name) > 2:  # Avoid matching single words
                    return team_name.title()
        
        return None
    
    def extract_position_from_text(self, text: str) -> Optional[str]:
        """
        Extract player position from text
        
        Args:
            text: Text containing position information
            
        Returns:
            Position or None
        """
        text_lower = text.lower()
        
        # Check for known positions
        for position in self.position_contexts.keys():
            if position in text_lower:
                return position
        
        # Look for common position patterns
        patterns = [
            r'([a-zA-Z]+)er',  # Striker, Midfielder, Defender
            r'([a-zA-Z]+) back',  # Center back, Left back, etc.
            r'winger',
            r'goalkeeper',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                position = match.group(1) + "er" if len(match.groups()) > 0 else match.group(0)
                return position
        
        return None


class SimilaritySearch:
    """Perform similarity searches on embeddings"""
    
    def __init__(self, embeddings: np.ndarray = None, 
                 labels: List[str] = None):
        """
        Initialize similarity search
        
        Args:
            embeddings: Embedding matrix
            labels: Labels for embeddings
        """
        self.embeddings = embeddings
        self.labels = labels
        self.index = None
        
        if embeddings is not None:
            self._build_index()
    
    def _build_index(self):
        """Build search index for embeddings"""
        if self.embeddings is None:
            raise ValueError("No embeddings provided")
        
        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Build simple index (for large datasets, consider FAISS or Annoy)
        logger.info(f"Built similarity search index with {len(self.embeddings)} embeddings")
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10,
               threshold: float = 0.0) -> List[SimilarityResult]:
        """
        Search for most similar embeddings
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SimilarityResult objects
        """
        if self.normalized_embeddings is None:
            raise ValueError("No index built")
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(self.normalized_embeddings, query_normalized)
        distances = 1 - similarities  # Convert to distances
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            
            if similarity >= threshold:
                result = SimilarityResult(
                    query="query_embedding",
                    target=self.labels[idx] if self.labels else f"embedding_{idx}",
                    similarity=float(similarity),
                    distance=float(distances[idx]),
                    rank=i + 1
                )
                results.append(result)
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray,
                    top_k: int = 10,
                    threshold: float = 0.0) -> List[List[SimilarityResult]]:
        """
        Perform batch similarity search
        
        Args:
            query_embeddings: Multiple query embeddings
            top_k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists
        """
        results = []
        
        for query_embedding in query_embeddings:
            query_results = self.search(query_embedding, top_k, threshold)
            results.append(query_results)
        
        return results
    
    def find_duplicates(self, threshold: float = 0.95) -> List[List[int]]:
        """
        Find duplicate or very similar embeddings
        
        Args:
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of groups of duplicate indices
        """
        if self.normalized_embeddings is None:
            raise ValueError("No index built")
        
        # Compute similarity matrix
        similarity_matrix = np.dot(
            self.normalized_embeddings, 
            self.normalized_embeddings.T
        )
        
        # Find groups of similar embeddings
        groups = []
        visited = set()
        
        for i in range(len(similarity_matrix)):
            if i in visited:
                continue
            
            # Find all embeddings similar to current one
            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx != i]
            
            if len(similar_indices) > 0:
                group = [i] + similar_indices
                groups.append(group)
                visited.update(group)
        
        return groups
    
    def compute_pairwise_similarities(self) -> np.ndarray:
        """
        Compute pairwise similarities between all embeddings
        
        Returns:
            Similarity matrix
        """
        if self.normalized_embeddings is None:
            raise ValueError("No index built")
        
        return np.dot(self.normalized_embeddings, self.normalized_embeddings.T)


class EmbeddingUtils:
    """Utility functions for embedding operations"""
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    @staticmethod
    def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
        """Compute centroid of embeddings"""
        return np.mean(embeddings, axis=0)
    
    @staticmethod
    def compute_variance(embeddings: np.ndarray) -> float:
        """Compute variance of embeddings"""
        centroid = EmbeddingUtils.compute_centroid(embeddings)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return np.var(distances)
    
    @staticmethod
    def filter_outliers(embeddings: np.ndarray, 
                       labels: List[str] = None,
                       threshold: float = 2.0) -> Tuple[np.ndarray, List[str]]:
        """
        Filter outlier embeddings
        
        Args:
            embeddings: Embedding matrix
            labels: Labels corresponding to embeddings
            threshold: Standard deviation threshold
            
        Returns:
            Filtered embeddings and labels
        """
        centroid = EmbeddingUtils.compute_centroid(embeddings)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Filter outliers
        outlier_mask = distances < (mean_distance + threshold * std_distance)
        filtered_embeddings = embeddings[outlier_mask]
        
        if labels is not None:
            filtered_labels = [labels[i] for i in range(len(labels)) if outlier_mask[i]]
            return filtered_embeddings, filtered_labels
        else:
            return filtered_embeddings, None
    
    @staticmethod
    def compute_inter_cluster_distance(cluster1: np.ndarray, 
                                      cluster2: np.ndarray) -> float:
        """Compute distance between two clusters"""
        centroid1 = EmbeddingUtils.compute_centroid(cluster1)
        centroid2 = EmbeddingUtils.compute_centroid(cluster2)
        return np.linalg.norm(centroid1 - centroid2)
    
    @staticmethod
    def compute_intra_cluster_distance(embeddings: np.ndarray) -> float:
        """Compute average intra-cluster distance"""
        if len(embeddings) < 2:
            return 0.0
        
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(distance)
        
        return np.mean(distances)
    
    @staticmethod
    def apply_pca(embeddings: np.ndarray, 
                  n_components: int = 50,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA dimensionality reduction
        
        Args:
            embeddings: Input embeddings
            n_components: Number of components
            random_state: Random state
            
        Returns:
            Reduced embeddings and explained variance ratio
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_embeddings = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_
        
        return reduced_embeddings, explained_variance
    
    @staticmethod
    def cluster_quality_metrics(embeddings: np.ndarray, 
                               labels: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering quality metrics
        
        Args:
            embeddings: Embedding matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of quality metrics
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # Compute metrics
        silhouette = silhouette_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        }


class MatchScoreAggregator:
    """Aggregate multiple match scores for robust identification"""
    
    def __init__(self):
        self.weights = {
            'visual_similarity': 0.4,
            'text_similarity': 0.3,
            'team_context': 0.2,
            'position_match': 0.1
        }
    
    def aggregate_scores(self, 
                        visual_scores: Dict[str, float],
                        text_scores: Dict[str, float] = None,
                        team_matches: Dict[str, float] = None,
                        position_matches: Dict[str, float] = None) -> Dict[str, float]:
        """
        Aggregate multiple score types
        
        Args:
            visual_scores: Visual similarity scores
            text_scores: Text similarity scores
            team_matches: Team context match scores
            position_matches: Position match scores
            
        Returns:
            Aggregated scores
        """
        all_players = set()
        
        # Collect all player names
        all_players.update(visual_scores.keys())
        if text_scores:
            all_players.update(text_scores.keys())
        if team_matches:
            all_players.update(team_matches.keys())
        if position_matches:
            all_players.update(position_matches.keys())
        
        aggregated_scores = {}
        
        for player in all_players:
            score = 0.0
            
            # Visual similarity (always present)
            visual_score = visual_scores.get(player, 0.0)
            score += visual_score * self.weights['visual_similarity']
            
            # Text similarity
            if text_scores:
                text_score = text_scores.get(player, 0.0)
                score += text_score * self.weights['text_similarity']
            
            # Team context
            if team_matches:
                team_score = team_matches.get(player, 0.0)
                score += team_score * self.weights['team_context']
            
            # Position match
            if position_matches:
                position_score = position_matches.get(player, 0.0)
                score += position_score * self.weights['position_match']
            
            aggregated_scores[player] = score
        
        return aggregated_scores
    
    def rank_players(self, aggregated_scores: Dict[str, float], 
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rank players by aggregated scores
        
        Args:
            aggregated_scores: Aggregated scores for players
            top_k: Number of top players to return
            
        Returns:
            List of (player, score) tuples
        """
        sorted_players = sorted(
            aggregated_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_players[:top_k]


# Factory functions
def create_name_normalizer() -> PlayerNameNormalizer:
    """Create player name normalizer instance"""
    return PlayerNameNormalizer()


def create_prompt_processor() -> TextPromptProcessor:
    """Create text prompt processor instance"""
    return TextPromptProcessor()


def create_similarity_search(embeddings: np.ndarray, 
                           labels: List[str] = None) -> SimilaritySearch:
    """Create similarity search instance"""
    return SimilaritySearch(embeddings=embeddings, labels=labels)


def create_match_aggregator() -> MatchScoreAggregator:
    """Create match score aggregator instance"""
    return MatchScoreAggregator()


if __name__ == "__main__":
    # Example usage
    normalizer = create_name_normalizer()
    processor = create_prompt_processor()
    
    # Test name normalization
    test_name = "Lionel Andrés Messi"
    normalized = normalizer.normalize_name(test_name)
    print(f"Normalized name: {normalized}")
    
    # Test prompt generation
    prompt = processor.create_player_prompt(
        "Lionel Messi", 
        team_context="Barcelona",
        position_context="forward"
    )
    print(f"Generated prompt: {prompt}")
    
    # Test similarity search
    sample_embeddings = np.random.randn(100, 768)
    search = create_similarity_search(sample_embeddings, [f"player_{i}" for i in range(100)])
    
    query_embedding = np.random.randn(768)
    results = search.search(query_embedding, top_k=5)
    print(f"Found {len(results)} similar players")
    
    # Test score aggregation
    aggregator = create_match_aggregator()
    visual_scores = {"Messi": 0.9, "Ronaldo": 0.7, "Neymar": 0.6}
    text_scores = {"Messi": 0.8, "Ronaldo": 0.9, "Neymar": 0.5}
    
    aggregated = aggregator.aggregate_scores(visual_scores, text_scores)
    ranked = aggregator.rank_players(aggregated)
    print(f"Top player: {ranked[0]}")