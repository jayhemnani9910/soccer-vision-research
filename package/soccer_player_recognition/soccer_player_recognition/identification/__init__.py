"""Identification module for player recognition and matching.

This module handles player identification using:
- SigLIP zero-shot identification
- Traditional facial features and jersey analysis
- Multimodal text-image matching
- Player clustering algorithms
"""

# Traditional identification methods
from .face_identifier import FaceIdentifier
from .jersey_analyzer import JerseyAnalyzer
from .identification_engine import IdentificationEngine
from .embedding_extractor import EmbeddingExtractor

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
    # Traditional identification
    "FaceIdentifier",
    "JerseyAnalyzer",
    "IdentificationEngine", 
    "EmbeddingExtractor",
    
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