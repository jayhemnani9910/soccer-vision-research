"""
Identification Models Package

This package contains models for player identification including:
- SigLIP model for zero-shot identification
- Player clustering algorithms
- Text processing utilities
"""

from .siglip_model import (
    SigLIPPlayerIdentification,
    SigLIPConfig,
    SigLIPModel,
    create_siglip_model
)

from .player_clustering import (
    PlayerClusterer,
    TeamClusterer,
    ClusteringConfig,
    create_player_clusterer,
    create_team_clusterer
)

__all__ = [
    "SigLIPPlayerIdentification",
    "SigLIPConfig", 
    "SigLIPModel",
    "create_siglip_model",
    "PlayerClusterer",
    "TeamClusterer",
    "ClusteringConfig",
    "create_player_clusterer",
    "create_team_clusterer",
]