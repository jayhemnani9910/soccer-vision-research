"""Identification module for player recognition and matching.

This module handles player identification using:
- SigLIP zero-shot identification
- Multimodal text-image matching
- Player clustering algorithms
"""

# SigLIP zero-shot identification
from ..models.identification.siglip_model import (
    SigLIPPlayerIdentification,
    SigLIPConfig,
    SigLIPModel,
    create_siglip_model
)

from ..models.identification.player_clustering import (
    PlayerClusterer,
    TeamClusterer,
    ClusteringConfig,
    create_player_clusterer,
    create_team_clusterer
)

from ..utils.siglip_utils import (
    PlayerNameNormalizer,
    TextPromptProcessor,
    SimilaritySearch,
    EmbeddingUtils,
    MatchScoreAggregator,
    create_name_normalizer,
    create_prompt_processor,
    create_similarity_search,
    create_match_aggregator
)

__all__ = [
    # SigLIP identification
    "SigLIPPlayerIdentification",
    "SigLIPConfig",
    "SigLIPModel",
    "create_siglip_model",
    
    # Clustering
    "PlayerClusterer",
    "TeamClusterer",
    "ClusteringConfig",
    "create_player_clusterer",
    "create_team_clusterer",
    
    # Utilities
    "PlayerNameNormalizer",
    "TextPromptProcessor",
    "SimilaritySearch",
    "EmbeddingUtils",
    "MatchScoreAggregator",
    "create_name_normalizer",
    "create_prompt_processor",
    "create_similarity_search",
    "create_match_aggregator",
]