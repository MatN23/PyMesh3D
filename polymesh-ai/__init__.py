"""
Mesh Transformers Library

A comprehensive library for processing 3D meshes using transformer architectures.
Includes tokenization strategies, attention mechanisms, training pipelines, and utilities.
"""

__version__ = "0.3.4"
__author__ = "Matias Nielsen"

# Core mesh library components
from .mesh_library import (
    Vertex,
    Face, 
    Mesh,
    MeshGenerator,
    MeshLoader,
    MeshDataset
)

# Tokenization components
from .mesh_transformers import (
    MeshToken,
    MeshTokenizer,
    VertexTokenizer,
    FaceTokenizer,
    PatchTokenizer,
    MeshPositionalEncoding,
    MeshTransformerEmbedding,
    MeshTransformer,
    MeshTransformerTrainer
)

# Advanced attention mechanisms
from .mesh_attention import (
    GeometricAttention,
    GraphAttention,
    MultiScaleAttention,
    SparseAttention,
    MeshTransformerLayer,
    AdaptiveMeshTransformer,
    MeshTransformerPreTrainer
)

# Training pipeline components
from .mesh_training_pipeline import (
    MeshTransformerDataset,
    MeshTransformerTrainingPipeline,
    MeshAugmentation,
    MeshDatasetBuilder,
    train_mesh_classifier,
    train_mesh_autoencoder
)

# Convenience imports for common use cases
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core mesh data structures
    "Vertex",
    "Face",
    "Mesh",
    "MeshGenerator",
    "MeshLoader", 
    "MeshDataset",
    
    # Tokenization
    "MeshToken",
    "MeshTokenizer",
    "VertexTokenizer",
    "FaceTokenizer", 
    "PatchTokenizer",
    
    # Model components
    "MeshPositionalEncoding",
    "MeshTransformerEmbedding",
    "MeshTransformer",
    "MeshTransformerTrainer",
    
    # Advanced attention mechanisms
    "GeometricAttention",
    "GraphAttention", 
    "MultiScaleAttention",
    "SparseAttention",
    "MeshTransformerLayer",
    "AdaptiveMeshTransformer",
    "MeshTransformerPreTrainer",
    
    # Training pipeline
    "MeshTransformerDataset",
    "MeshTransformerTrainingPipeline",
    "MeshAugmentation",
    "MeshDatasetBuilder",
    "train_mesh_classifier",
    "train_mesh_autoencoder",
]

# Utility functions for quick setup
def create_vertex_tokenizer(include_normals=True, include_colors=False, quantize=False):
    """
    Create a vertex tokenizer with common settings.
    
    Args:
        include_normals (bool): Include vertex normals in features
        include_colors (bool): Include vertex colors in features  
        quantize (bool): Quantize vertex positions
        
    Returns:
        VertexTokenizer: Configured tokenizer
    """
    return VertexTokenizer(
        include_normals=include_normals,
        include_colors=include_colors,
        quantize_positions=quantize
    )

def create_mesh_transformer(feature_dim, num_classes=None, task='classification'):
    """
    Create a mesh transformer model with sensible defaults.
    
    Args:
        feature_dim (int): Dimension of input features
        num_classes (int, optional): Number of output classes for classification
        task (str): Task type - 'classification', 'reconstruction', or 'generation'
        
    Returns:
        MeshTransformer: Configured model
    """
    model = MeshTransformer(
        feature_dim=feature_dim,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        use_3d_pos_encoding=True
    )
    
    # Adjust output head for number of classes if specified
    if task == 'classification' and num_classes is not None:
        import torch.nn as nn
        model.classification_head = nn.Linear(512, num_classes)
    
    return model

def create_adaptive_mesh_transformer(d_model=512, num_layers=6):
    """
    Create an adaptive mesh transformer with multiple attention mechanisms.
    
    Args:
        d_model (int): Model dimension
        num_layers (int): Number of transformer layers
        
    Returns:
        AdaptiveMeshTransformer: Configured adaptive model
    """
    return AdaptiveMeshTransformer(
        d_model=d_model,
        nhead=8,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=0.1
    )

def create_training_pipeline(config):
    """
    Create a training pipeline with the given configuration.
    
    Args:
        config (dict): Training configuration dictionary
        
    Returns:
        MeshTransformerTrainingPipeline: Configured training pipeline
    """
    return MeshTransformerTrainingPipeline(config)

def load_mesh_obj(filepath):
    """
    Quick utility to load a mesh from OBJ file.
    
    Args:
        filepath (str): Path to OBJ file
        
    Returns:
        Mesh: Loaded mesh object
    """
    return MeshLoader.load_obj(filepath)

def generate_sample_mesh(mesh_type='cube', **kwargs):
    """
    Generate a sample mesh for testing.
    
    Args:
        mesh_type (str): Type of mesh - 'cube', 'sphere', 'cylinder', 'plane'
        **kwargs: Additional arguments for mesh generation
        
    Returns:
        Mesh: Generated mesh
    """
    if mesh_type == 'cube':
        return MeshGenerator.cube(kwargs.get('size', 1.0))
    elif mesh_type == 'sphere':
        return MeshGenerator.sphere(
            radius=kwargs.get('radius', 1.0),
            subdivisions=kwargs.get('subdivisions', 2)
        )
    elif mesh_type == 'cylinder':
        return MeshGenerator.cylinder(
            radius=kwargs.get('radius', 1.0),
            height=kwargs.get('height', 2.0),
            segments=kwargs.get('segments', 16)
        )
    elif mesh_type == 'plane':
        return MeshGenerator.plane(
            width=kwargs.get('width', 2.0),
            height=kwargs.get('height', 2.0),
            width_segments=kwargs.get('width_segments', 1),
            height_segments=kwargs.get('height_segments', 1)
        )
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

# Example configurations for common use cases
CLASSIFICATION_CONFIG = {
    'model_type': 'standard',
    'tokenizer_type': 'vertex',
    'feature_dim': 6,  # position + normal
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'max_epochs': 100,
    'task_type': 'classification',
    'include_normals': True,
    'include_colors': False
}

AUTOENCODER_CONFIG = {
    'model_type': 'standard',
    'tokenizer_type': 'vertex', 
    'feature_dim': 6,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'learning_rate': 5e-5,
    'batch_size': 16,
    'max_epochs': 200,
    'task_type': 'reconstruction',
    'include_normals': True,
    'include_colors': False
}

ADAPTIVE_CONFIG = {
    'model_type': 'adaptive',
    'tokenizer_type': 'vertex',
    'feature_dim': 6,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'max_epochs': 100,
    'task_type': 'classification',
    'include_normals': True,
    'include_colors': False
}

def get_config(config_name):
    """
    Get a predefined configuration.
    
    Args:
        config_name (str): Name of configuration - 'classification', 'autoencoder', 'adaptive'
        
    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'classification': CLASSIFICATION_CONFIG.copy(),
        'autoencoder': AUTOENCODER_CONFIG.copy(), 
        'adaptive': ADAPTIVE_CONFIG.copy()
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

# Library information
def get_library_info():
    """Get information about the mesh transformers library."""
    info = {
        'version': __version__,
        'author': __author__,
        'description': __doc__.strip(),
        'components': {
            'mesh_library': 'Core mesh data structures and utilities',
            'mesh_transformers': 'Basic transformer models and tokenization',
            'mesh_attention': 'Advanced attention mechanisms',
            'mesh_training_pipeline': 'Training utilities and pipelines'
        },
        'supported_tasks': [
            'Mesh classification',
            'Mesh reconstruction', 
            'Mesh generation',
            'Mesh segmentation',
            'Shape completion'
        ],
        'tokenization_strategies': [
            'Vertex-based tokenization',
            'Face-based tokenization',
            'Patch-based tokenization'
        ],
        'attention_mechanisms': [
            'Geometric attention (distance-based)',
            'Graph attention (connectivity-based)', 
            'Multi-scale attention',
            'Sparse attention'
        ]
    }
    return info

def print_library_info():
    """Print library information in a formatted way."""
    info = get_library_info()
    print(f"Mesh Transformers Library v{info['version']}")
    print("=" * 50)
    print(f"Author: {info['author']}")
    print(f"\n{info['description']}")
    print("\nSupported Tasks:")
    for task in info['supported_tasks']:
        print(f"  • {task}")
    print("\nTokenization Strategies:")
    for strategy in info['tokenization_strategies']:
        print(f"  • {strategy}")
    print("\nAttention Mechanisms:")
    for mechanism in info['attention_mechanisms']:
        print(f"  • {mechanism}")

# Quick start example
def quick_start_example():
    """
    Demonstrate basic library usage.
    
    Returns:
        dict: Results from the quick start example
    """
    print("Mesh Transformers Quick Start Example")
    print("=" * 40)
    
    # 1. Generate a sample mesh
    print("1. Generating sample sphere mesh...")
    mesh = generate_sample_mesh('sphere', radius=1.0, subdivisions=1)
    mesh.compute_vertex_normals()
    print(f"   Created mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # 2. Create tokenizer
    print("2. Creating vertex tokenizer...")
    tokenizer = create_vertex_tokenizer(include_normals=True)
    tokens = tokenizer.tokenize(mesh)
    print(f"   Generated {len(tokens)} tokens")
    
    # 3. Create model
    print("3. Creating mesh transformer model...")
    model = create_mesh_transformer(feature_dim=6, num_classes=10, task='classification')
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {param_count:,} parameters")
    
    # 4. Test forward pass
    print("4. Testing forward pass...")
    import torch
    with torch.no_grad():
        output = model(tokens, task='classification')
        print(f"   Output shape: {output.shape}")
    
    results = {
        'mesh_vertices': len(mesh.vertices),
        'mesh_faces': len(mesh.faces),
        'tokens_generated': len(tokens),
        'model_parameters': param_count,
        'output_shape': tuple(output.shape)
    }
    
    print("Quick start completed successfully!")
    return results

# Error handling for imports
try:
    import torch
    import torch.nn as nn
    import numpy as np
    _DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    _DEPENDENCIES_AVAILABLE = False
    _MISSING_DEPENDENCY = str(e)

def check_dependencies():
    """Check if all required dependencies are available."""
    if not _DEPENDENCIES_AVAILABLE:
        raise ImportError(f"Missing required dependencies: {_MISSING_DEPENDENCY}")
    return True

# Initialize library
def initialize():
    """Initialize the mesh transformers library."""
    try:
        check_dependencies()
        print(f"Mesh Transformers Library v{__version__} initialized successfully!")
        return True
    except ImportError as e:
        print(f"Failed to initialize library: {e}")
        return False