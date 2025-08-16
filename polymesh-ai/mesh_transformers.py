# Copyright (c) 2025 Matias Nielsen. All rights reserved.
# Licensed under the Custom License below.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

# Assuming the base mesh library is imported
# from mesh_library import Mesh, Vertex, MeshDataset

@dataclass
class MeshToken:
    """A tokenized representation of mesh elements"""
    token_id: int
    features: np.ndarray
    position: np.ndarray
    token_type: str  # 'vertex', 'face', 'edge', 'patch'
    metadata: Optional[Dict] = None

class MeshTokenizer(ABC):
    """Abstract base class for mesh tokenization strategies"""
    
    @abstractmethod
    def tokenize(self, mesh) -> List[MeshToken]:
        pass
    
    @abstractmethod
    def detokenize(self, tokens: List[MeshToken]):
        pass

class VertexTokenizer(MeshTokenizer):
    """Tokenizes mesh by treating each vertex as a token"""
    
    def __init__(self, feature_dim: int = 3, include_normals: bool = True, 
                 include_colors: bool = False, quantize_positions: bool = False,
                 quantization_levels: int = 256):
        self.feature_dim = feature_dim
        self.include_normals = include_normals
        self.include_colors = include_colors
        self.quantize_positions = quantize_positions
        self.quantization_levels = quantization_levels
        
    def tokenize(self, mesh) -> List[MeshToken]:
        """Convert mesh vertices to tokens"""
        tokens = []
        
        for i, vertex in enumerate(mesh.vertices):
            # Base features: position
            features = vertex.position.copy()
            
            # Add normals if available and requested
            if self.include_normals and vertex.normal is not None:
                features = np.concatenate([features, vertex.normal])
            elif self.include_normals:
                features = np.concatenate([features, np.array([0, 0, 1], dtype=np.float32)])
            
            # Add colors if available and requested
            if self.include_colors and vertex.color is not None:
                color = vertex.color[:3] if len(vertex.color) > 3 else vertex.color
                features = np.concatenate([features, color])
            elif self.include_colors:
                features = np.concatenate([features, np.array([0.5, 0.5, 0.5], dtype=np.float32)])
            
            # Quantize positions if requested
            position = vertex.position
            if self.quantize_positions:
                position = np.round(position * self.quantization_levels) / self.quantization_levels
            
            token = MeshToken(
                token_id=i,
                features=features.astype(np.float32),
                position=position,
                token_type='vertex',
                metadata={'vertex_index': i}
            )
            tokens.append(token)
        
        return tokens
    
    def detokenize(self, tokens: List[MeshToken]):
        """Convert tokens back to mesh vertices"""
        from mesh_library import Vertex  # Assuming this import works
        
        vertices = []
        for token in tokens:
            if token.token_type != 'vertex':
                continue
                
            features = token.features
            position = features[:3]
            
            normal = None
            if self.include_normals and len(features) >= 6:
                normal = features[3:6]
            
            color = None
            if self.include_colors and len(features) >= 9:
                color = features[6:9]
            elif self.include_colors and len(features) >= 6 and not self.include_normals:
                color = features[3:6]
            
            vertex = Vertex(position=position, normal=normal, color=color)
            vertices.append(vertex)
        
        return vertices

class FaceTokenizer(MeshTokenizer):
    """Tokenizes mesh by treating each face as a token"""
    
    def __init__(self, max_face_vertices: int = 4, include_face_normal: bool = True):
        self.max_face_vertices = max_face_vertices
        self.include_face_normal = include_face_normal
    
    def tokenize(self, mesh) -> List[MeshToken]:
        """Convert mesh faces to tokens"""
        tokens = []
        vertex_positions = mesh.vertex_positions
        face_normals = mesh.compute_face_normals() if self.include_face_normal else None
        
        for i, face in enumerate(mesh.faces):
            # Compute face center
            face_vertices = vertex_positions[face]
            face_center = np.mean(face_vertices, axis=0)
            
            # Create feature vector
            features = []
            
            # Add face vertex indices (padded to max_face_vertices)
            padded_face = np.pad(face, (0, max(0, self.max_face_vertices - len(face))), 
                               mode='constant', constant_values=-1)[:self.max_face_vertices]
            features.extend(padded_face.astype(np.float32))
            
            # Add face normal if requested
            if self.include_face_normal and face_normals is not None:
                features.extend(face_normals[i])
            
            # Add face area
            if len(face) >= 3:
                v0, v1, v2 = face_vertices[0], face_vertices[1], face_vertices[2]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            else:
                area = 0.0
            features.append(area)
            
            token = MeshToken(
                token_id=i,
                features=np.array(features, dtype=np.float32),
                position=face_center,
                token_type='face',
                metadata={'face_index': i, 'face_size': len(face)}
            )
            tokens.append(token)
        
        return tokens
    
    def detokenize(self, tokens: List[MeshToken]):
        """Convert face tokens back to face indices"""
        faces = []
        for token in tokens:
            if token.token_type != 'face':
                continue
                
            # Extract face indices from features
            face_indices = token.features[:self.max_face_vertices].astype(int)
            # Remove padding (-1 values)
            face_indices = face_indices[face_indices >= 0]
            faces.append(face_indices.tolist())
        
        return faces

class PatchTokenizer(MeshTokenizer):
    """Tokenizes mesh by grouping vertices into local patches"""
    
    def __init__(self, patch_size: int = 16, overlap: int = 4, 
                 feature_aggregation: str = 'mean'):
        self.patch_size = patch_size
        self.overlap = overlap
        self.feature_aggregation = feature_aggregation
    
    def tokenize(self, mesh) -> List[MeshToken]:
        """Group vertices into patches and create tokens"""
        tokens = []
        vertex_positions = mesh.vertex_positions
        adjacency = mesh.get_adjacency_matrix()
        
        visited = set()
        patch_id = 0
        
        for start_vertex in range(len(mesh.vertices)):
            if start_vertex in visited:
                continue
            
            # BFS to find patch vertices
            patch_vertices = self._bfs_patch(start_vertex, adjacency, self.patch_size)
            
            # Mark vertices as visited (with overlap consideration)
            for v in patch_vertices[:-self.overlap]:
                visited.add(v)
            
            # Aggregate patch features
            patch_positions = vertex_positions[patch_vertices]
            patch_center = np.mean(patch_positions, axis=0)
            
            if self.feature_aggregation == 'mean':
                patch_features = np.mean(patch_positions, axis=0)
            elif self.feature_aggregation == 'max':
                patch_features = np.max(patch_positions, axis=0)
            elif self.feature_aggregation == 'concat':
                # Pad or truncate to fixed size
                padded_positions = np.zeros((self.patch_size, 3), dtype=np.float32)
                n_vertices = min(len(patch_positions), self.patch_size)
                padded_positions[:n_vertices] = patch_positions[:n_vertices]
                patch_features = padded_positions.flatten()
            else:
                patch_features = np.mean(patch_positions, axis=0)
            
            token = MeshToken(
                token_id=patch_id,
                features=patch_features,
                position=patch_center,
                token_type='patch',
                metadata={
                    'patch_vertices': patch_vertices,
                    'patch_size': len(patch_vertices)
                }
            )
            tokens.append(token)
            patch_id += 1
        
        return tokens
    
    def _bfs_patch(self, start_vertex: int, adjacency: np.ndarray, max_size: int) -> List[int]:
        """BFS to find connected patch of vertices"""
        visited = {start_vertex}
        queue = [start_vertex]
        patch = [start_vertex]
        
        while queue and len(patch) < max_size:
            current = queue.pop(0)
            neighbors = np.where(adjacency[current])[0]
            
            for neighbor in neighbors:
                if neighbor not in visited and len(patch) < max_size:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    patch.append(neighbor)
        
        return patch
    
    def detokenize(self, tokens: List[MeshToken]):
        """Reconstruct patches from tokens"""
        patches = []
        for token in tokens:
            if token.token_type != 'patch':
                continue
            patches.append({
                'vertices': token.metadata.get('patch_vertices', []),
                'center': token.position,
                'features': token.features
            })
        return patches

class MeshPositionalEncoding(nn.Module):
    """Positional encoding for mesh tokens based on 3D coordinates"""
    
    def __init__(self, d_model: int, max_freq: float = 10.0, num_freq_bands: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        
        # Create frequency bands
        freq_bands = torch.linspace(1.0, max_freq, num_freq_bands)
        self.register_buffer('freq_bands', freq_bands)
        
        # Linear projection to d_model
        encoding_dim = 3 * 2 * num_freq_bands  # 3 coords * 2 (sin/cos) * num_bands
        self.projection = nn.Linear(encoding_dim, d_model)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch_size, seq_len, 3] - 3D positions
        Returns:
            [batch_size, seq_len, d_model] - positional encodings
        """
        batch_size, seq_len, _ = positions.shape
        
        # Create sinusoidal encodings
        encodings = []
        for coord_idx in range(3):  # x, y, z
            coord = positions[:, :, coord_idx:coord_idx+1]  # [B, L, 1]
            
            # Apply frequency bands
            freqs = coord * self.freq_bands.view(1, 1, -1)  # [B, L, num_freq_bands]
            
            # Sin and cos
            encodings.append(torch.sin(freqs))
            encodings.append(torch.cos(freqs))
        
        # Concatenate all encodings
        encoding = torch.cat(encodings, dim=-1)  # [B, L, 3 * 2 * num_freq_bands]
        
        # Project to d_model
        return self.projection(encoding)

class MeshTransformerEmbedding(nn.Module):
    """Embedding layer for mesh tokens with positional encoding"""
    
    def __init__(self, feature_dim: int, d_model: int, max_seq_len: int = 1024,
                 use_3d_pos_encoding: bool = True, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.use_3d_pos_encoding = use_3d_pos_encoding
        
        # Feature embedding
        self.feature_embedding = nn.Linear(feature_dim, d_model)
        
        # Token type embeddings
        self.token_type_embedding = nn.Embedding(4, d_model)  # vertex, face, edge, patch
        
        # Positional encoding
        if use_3d_pos_encoding:
            self.pos_encoding = MeshPositionalEncoding(d_model)
        else:
            # Standard 1D positional encoding
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(dropout)
        
        # Token type mapping
        self.token_type_map = {'vertex': 0, 'face': 1, 'edge': 2, 'patch': 3}
    
    def forward(self, tokens: List[MeshToken], device: torch.device = None) -> torch.Tensor:
        """
        Convert mesh tokens to transformer embeddings
        
        Args:
            tokens: List of MeshToken objects
            device: Target device for tensors
            
        Returns:
            [seq_len, d_model] - token embeddings
        """
        if device is None:
            device = next(self.parameters()).device
        
        seq_len = len(tokens)
        
        # Extract features and positions
        features = torch.tensor([token.features for token in tokens], 
                               dtype=torch.float32, device=device)
        positions = torch.tensor([token.position for token in tokens], 
                                dtype=torch.float32, device=device)
        token_types = torch.tensor([self.token_type_map.get(token.token_type, 0) 
                                  for token in tokens], dtype=torch.long, device=device)
        
        # Feature embeddings
        feature_emb = self.feature_embedding(features)
        
        # Token type embeddings
        type_emb = self.token_type_embedding(token_types)
        
        # Positional encoding
        if self.use_3d_pos_encoding:
            pos_emb = self.pos_encoding(positions.unsqueeze(0)).squeeze(0)
        else:
            pos_emb = self.pe[:seq_len]
        
        # Combine embeddings
        embeddings = feature_emb + type_emb + pos_emb
        
        return self.dropout(embeddings)

class MeshTransformer(nn.Module):
    """Transformer model for mesh processing"""
    
    def __init__(self, feature_dim: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_seq_len: int = 1024,
                 use_3d_pos_encoding: bool = True):
        super().__init__()
        
        self.embedding = MeshTransformerEmbedding(
            feature_dim, d_model, max_seq_len, use_3d_pos_encoding, dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads (can be customized for different tasks)
        self.classification_head = nn.Linear(d_model, 1000)  # For mesh classification
        self.reconstruction_head = nn.Linear(d_model, feature_dim)  # For reconstruction
        self.generation_head = nn.Linear(d_model, feature_dim)  # For generation
        
    def forward(self, tokens: List[MeshToken], task: str = 'classification',
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tokens: List of MeshToken objects
            task: 'classification', 'reconstruction', or 'generation'
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Task-specific output tensor
        """
        # Convert tokens to embeddings
        embeddings = self.embedding(tokens).unsqueeze(0)  # Add batch dimension
        
        # Apply transformer
        output = self.transformer_encoder(embeddings, mask=mask)
        
        # Apply task-specific head
        if task == 'classification':
            # Global average pooling + classification
            pooled = torch.mean(output, dim=1)  # [1, d_model]
            return self.classification_head(pooled)
        elif task == 'reconstruction':
            return self.reconstruction_head(output.squeeze(0))  # [seq_len, feature_dim]
        elif task == 'generation':
            return self.generation_head(output.squeeze(0))  # [seq_len, feature_dim]
        else:
            return output.squeeze(0)  # [seq_len, d_model]

class MeshTransformerTrainer:
    """Training utilities for mesh transformers"""
    
    def __init__(self, model: MeshTransformer, tokenizer: MeshTokenizer,
                 device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def prepare_batch(self, meshes: List) -> Dict[str, torch.Tensor]:
        """Prepare a batch of meshes for training"""
        batch_tokens = []
        batch_lengths = []
        
        for mesh in meshes:
            tokens = self.tokenizer.tokenize(mesh)
            batch_tokens.append(tokens)
            batch_lengths.append(len(tokens))
        
        # For now, process each mesh separately (can be extended for true batching)
        return {
            'tokens': batch_tokens,
            'lengths': torch.tensor(batch_lengths, device=self.device)
        }
    
    def train_step(self, batch: Dict, task: str = 'classification',
                   labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single training step"""
        self.model.train()
        total_loss = 0
        
        for i, tokens in enumerate(batch['tokens']):
            # Forward pass
            output = self.model(tokens, task=task)
            
            # Compute loss based on task
            if task == 'classification' and labels is not None:
                loss = F.cross_entropy(output, labels[i:i+1])
            elif task == 'reconstruction':
                # Reconstruct original features
                target_features = torch.tensor(
                    [token.features for token in tokens], 
                    dtype=torch.float32, device=self.device
                )
                loss = F.mse_loss(output, target_features)
            else:
                # Default: use dummy loss
                loss = torch.tensor(0.0, device=self.device)
            
            total_loss += loss
        
        return total_loss / len(batch['tokens'])

# Example usage and demonstration
def example_usage():
    """Demonstrate the mesh transformer pipeline"""
    # Create sample mesh (assuming MeshGenerator is available)
    # mesh = MeshGenerator.sphere(radius=1.0, subdivisions=2)
    
    # Initialize tokenizer
    tokenizer = VertexTokenizer(include_normals=True, include_colors=False)
    
    # Create transformer model
    model = MeshTransformer(
        feature_dim=6,  # 3 position + 3 normal
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    print("Mesh Transformer Model:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example tokenization (would use real mesh in practice)
    from mesh_library import Vertex
    dummy_vertices = [
        Vertex([0, 0, 0], [0, 0, 1]),
        Vertex([1, 0, 0], [1, 0, 0]),
        Vertex([0, 1, 0], [0, 1, 0])
    ]
    
    # Create dummy mesh object
    class DummyMesh:
        def __init__(self, vertices):
            self.vertices = vertices
    
    dummy_mesh = DummyMesh(dummy_vertices)
    
    # Tokenize
    tokens = tokenizer.tokenize(dummy_mesh)
    print(f"Generated {len(tokens)} tokens")
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens, task='classification')
        print(f"Classification output shape: {output.shape}")

if __name__ == "__main__":
    example_usage()