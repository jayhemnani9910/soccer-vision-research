"""
Demo script for SigLIP Player Identification System

This script demonstrates the complete SigLIP zero-shot player identification system
including model usage, clustering, and text processing utilities.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import json

# Add the soccer_player_recognition package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from soccer_player_recognition.identification import (
    SigLIPPlayerIdentification,
    SigLIPConfig,
    create_siglip_model,
    PlayerClusterer,
    ClusteringConfig,
    create_player_clusterer,
    PlayerNameNormalizer,
    TextPromptProcessor,
    SimilaritySearch,
    create_name_normalizer,
    create_prompt_processor,
    create_similarity_search,
    create_match_aggregator
)

def demo_basic_identification():
    """Demo basic player identification with SigLIP"""
    print("=" * 60)
    print("SIGLIP ZERO-SHOT PLAYER IDENTIFICATION DEMO")
    print("=" * 60)
    
    # Initialize SigLIP model
    print("\n1. Initializing SigLIP Model...")
    config = SigLIPConfig(
        model_name="siglip-base-patch16-224",
        image_size=224,
        temperature=0.07
    )
    
    model = create_siglip_model(config=config)
    print(f"✓ Model initialized on {model.device}")
    
    # Define player candidates
    players = [
        "Lionel Messi",
        "Cristiano Ronaldo", 
        "Neymar Jr",
        "Kylian Mbappé",
        "Erling Haaland",
        "Robert Lewandowski",
        "Harry Kane",
        "Sadio Mané"
    ]
    
    print(f"\n2. Player Candidates: {len(players)} players")
    for i, player in enumerate(players, 1):
        print(f"   {i}. {player}")
    
    # Create text prompts
    print("\n3. Creating Text Prompts...")
    prompt_processor = create_prompt_processor()
    
    team_contexts = {
        "Lionel Messi": "Barcelona",
        "Cristiano Ronaldo": "Real Madrid", 
        "Neymar Jr": "PSG",
        "Kylian Mbappé": "PSG"
    }
    
    for player in players[:4]:  # Show first 4
        prompt = prompt_processor.create_player_prompt(
            player, 
            team_context=team_contexts.get(player, "Soccer team"),
            position_context="forward"
        )
        print(f"   {player}: {prompt}")
    
    # Simulate embedding extraction (in real usage, this would use actual images)
    print("\n4. Simulating Player Embeddings...")
    dummy_embeddings = {
        player: np.random.randn(768) for player in players
    }
    
    # Demonstrate similarity search
    print("\n5. Similarity Search Demo...")
    query_player = "Lionel Messi"
    search = create_similarity_search(
        embeddings=np.array([dummy_embeddings[p] for p in players]),
        labels=players
    )
    
    query_embedding = dummy_embeddings[query_player]
    results = search.search(query_embedding, top_k=3)
    
    print(f"   Similar players to {query_player}:")
    for result in results:
        print(f"   - {result.target}: {result.similarity:.3f}")
    
    print("\n✓ Basic identification demo completed!\n")


def demo_clustering():
    """Demo player clustering functionality"""
    print("=" * 60)
    print("PLAYER CLUSTERING DEMO")
    print("=" * 60)
    
    # Initialize clusterer
    print("\n1. Initializing Player Clusterer...")
    cluster_config = ClusteringConfig(
        algorithm="kmeans",
        n_clusters=3,
        random_state=42
    )
    
    clusterer = create_player_clusterer(
        algorithm="kmeans", 
        config=cluster_config
    )
    print(f"✓ Clusterer initialized with {cluster_config.algorithm} algorithm")
    
    # Create sample player data (in real usage, this would come from actual images)
    print("\n2. Creating Sample Player Embeddings...")
    sample_players = {
        "Lionel Messi": np.random.randn(768) + np.array([1.0, 0.5, -0.3] * 256),  # Distinct pattern
        "Cristiano Ronaldo": np.random.randn(768) + np.array([0.8, -0.2, 1.1] * 256),
        "Neymar Jr": np.random.randn(768) + np.array([-0.5, 1.2, 0.4] * 256),
        "Kylian Mbappé": np.random.randn(768) + np.array([1.3, 0.1, -0.8] * 256),
        "Erling Haaland": np.random.randn(768) + np.array([-0.2, 0.9, 1.5] * 256),
        "Robert Lewandowski": np.random.randn(768) + np.array([0.3, -1.0, 0.7] * 256),
        "Harry Kane": np.random.randn(768) + np.array([1.1, 0.4, -0.1] * 256),
        "Sadio Mané": np.random.randn(768) + np.array([-0.8, 1.3, 0.2] * 256),
    }
    
    # Perform clustering
    print("\n3. Performing Clustering...")
    results = clusterer.cluster_players(sample_players)
    
    print(f"   Found {results['n_clusters']} clusters")
    print(f"   Silhouette Score: {results['metrics']['silhouette_score']:.3f}")
    
    # Display cluster results
    print("\n4. Cluster Results:")
    for cluster_id, players in results['cluster_players'].items():
        if cluster_id != -1:  # Skip noise points
            print(f"   Cluster {cluster_id}: {', '.join(players)}")
    
    # Find similar players
    print("\n5. Finding Similar Players...")
    target = "Lionel Messi"
    similar_players = clusterer.find_similar_players(target, sample_players, top_k=3)
    
    print(f"   Players most similar to {target}:")
    for player in similar_players:
        print(f"   - {player['player']}: {player['similarity']:.3f}")
    
    # Detect anomalies
    print("\n6. Anomaly Detection...")
    anomalies = clusterer.detect_anomalies(sample_players)
    
    if anomalies:
        print("   Detected anomalous players:")
        for anomaly in anomalies[:3]:
            print(f"   - {anomaly['player']}: anomaly score {anomaly['anomaly_score']:.3f}")
    else:
        print("   No anomalies detected")
    
    print("\n✓ Clustering demo completed!\n")


def demo_text_processing():
    """Demo text processing utilities"""
    print("=" * 60)
    print("TEXT PROCESSING UTILITIES DEMO")
    print("=" * 60)
    
    # Initialize normalizer
    print("\n1. Player Name Normalization...")
    normalizer = create_name_normalizer()
    
    test_names = [
        "Lionel Andrés Messi",
        "Cristiano Ronaldo dos Santos Aveiro", 
        "Neymar Jr.",
        "Kylian Mbappé"
    ]
    
    for name in test_names:
        normalized = normalizer.normalize_name(name)
        print(f"   '{name}' → '{normalized}'")
    
    # Name extraction
    print("\n2. Name Extraction from Text...")
    texts = [
        "a soccer player named Lionel Messi",
        "player Messi in Barcelona jersey",
        "forward named Cristiano Ronaldo"
    ]
    
    for text in texts:
        extracted = normalizer.extract_player_name(text)
        if extracted:
            print(f"   '{text}' → {extracted.full_name}")
    
    # Similar name finding
    print("\n3. Finding Similar Names...")
    query = "messi"
    candidates = ["lionel messi", "cristiano ronaldo", "neymar jr", "lionel andres"]
    
    similar = normalizer.find_similar_names(query, candidates, threshold=0.6)
    print(f"   Similar to '{query}': {similar}")
    
    # Prompt generation
    print("\n4. Text Prompt Generation...")
    processor = create_prompt_processor()
    
    player_prompt = processor.create_player_prompt(
        "Lionel Messi",
        team_context="Barcelona",
        position_context="forward",
        additional_context="professional soccer player"
    )
    
    print(f"   Player prompt: {player_prompt}")
    
    # Team prompt
    team_prompt = processor.create_team_prompt(
        "Barcelona",
        players=["Lionel Messi", "Sergio Busquets", "Xavi Hernandez"]
    )
    
    print(f"   Team prompt: {team_prompt}")
    
    # Extract team and position from text
    print("\n5. Extracting Context from Text...")
    test_texts = [
        "Player from Barcelona wearing red and blue jersey",
        "Forward playing for Manchester United",
        "Midfielder named Kevin De Bruyne from Manchester City"
    ]
    
    for text in test_texts:
        team = processor.extract_team_from_text(text)
        position = processor.extract_position_from_text(text)
        print(f"   '{text}'")
        print(f"      Team: {team or 'None'}")
        print(f"      Position: {position or 'None'}")
    
    print("\n✓ Text processing demo completed!\n")


def demo_model_configuration():
    """Demo different model configurations"""
    print("=" * 60)
    print("MODEL CONFIGURATION DEMO")
    print("=" * 60)
    
    # Show different configurations
    configs = [
        SigLIPConfig(
            model_name="siglip-base-patch16-224",
            vision_embed_dim=768,
            text_embed_dim=768,
            temperature=0.07,
            batch_size=32
        ),
        SigLIPConfig(
            model_name="siglip-large-patch16-384",
            vision_embed_dim=1024,
            text_embed_dim=1024,
            temperature=0.05,
            batch_size=16
        ),
        SigLIPConfig(
            model_name="siglip-base-patch16-224",
            vision_embed_dim=512,
            text_embed_dim=512,
            temperature=0.1,
            batch_size=64
        )
    ]
    
    print("\n1. Different SigLIP Configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\n   Configuration {i}:")
        print(f"   - Model: {config.model_name}")
        print(f"   - Vision Embedding Dim: {config.vision_embed_dim}")
        print(f"   - Text Embedding Dim: {config.text_embed_dim}")
        print(f"   - Temperature: {config.temperature}")
        print(f"   - Batch Size: {config.batch_size}")
    
    # Clustering configurations
    print("\n2. Different Clustering Configurations:")
    cluster_configs = [
        ClusteringConfig(algorithm="kmeans", n_clusters=4),
        ClusteringConfig(algorithm="dbscan", eps=0.5, min_samples=3),
        ClusteringConfig(algorithm="agglomerative", n_clusters=3)
    ]
    
    for i, config in enumerate(cluster_configs, 1):
        print(f"\n   Clustering Config {i}:")
        print(f"   - Algorithm: {config.algorithm}")
        if config.n_clusters:
            print(f"   - Number of Clusters: {config.n_clusters}")
        if config.algorithm == "dbscan":
            print(f"   - Epsilon: {config.eps}")
            print(f"   - Min Samples: {config.min_samples}")
    
    print("\n✓ Configuration demo completed!\n")


def main():
    """Run all demos"""
    print("SIGLIP PLAYER IDENTIFICATION SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo showcases the complete SigLIP zero-shot player identification")
    print("system including model usage, clustering, and text processing utilities.")
    print("=" * 80)
    
    try:
        # Check if PyTorch is available
        if torch.cuda.is_available():
            print(f"✓ PyTorch available with CUDA support: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ PyTorch available but using CPU (CUDA not available)")
        
        # Run demos
        demo_basic_identification()
        demo_clustering()  
        demo_text_processing()
        demo_model_configuration()
        
        print("=" * 80)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("• SigLIP model initialization and configuration")
        print("• Zero-shot player identification")
        print("• Text prompt generation and processing")
        print("• Player clustering with multiple algorithms")
        print("• Similarity search and anomaly detection")
        print("• Name normalization and extraction")
        print("• Team and position context handling")
        print("\nNext Steps:")
        print("• Load real player images for embedding extraction")
        print("• Train or fine-tune the SigLIP model")
        print("• Integrate with video analysis pipeline")
        print("• Deploy for real-time player identification")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please ensure all required dependencies are installed:")
        print("- torch")
        print("- torchvision")
        print("- scikit-learn")
        print("- matplotlib")
        print("- seaborn")
        print("- numpy")
        print("- PIL")
        print("- scipy")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()