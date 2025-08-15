# PyMesh3D: Transformer Library for 3D Mesh Processing

A comprehensive PyTorch library that brings transformer architectures to 3D mesh data, enabling advanced geometric deep learning for computer vision and AI applications.

## 🚀 Features

- **Multiple Tokenization Strategies**: Vertex, face, and patch-based mesh tokenization
- **3D-Aware Attention Mechanisms**: Geometric, graph, multi-scale, and sparse attention
- **Adaptive Architecture**: Dynamically selects optimal attention types per layer
- **Complete Training Pipeline**: End-to-end training with augmentation and validation
- **Pre-training Tasks**: Masked mesh modeling, completion, and autoregressive generation
- **Mesh Utilities**: Primitives generation, transformations, and I/O operations


## 🛠️ Installation

```bash
# Required dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Mesh Classification

```python
from mesh_transformers import *
from mesh_attention import *

# Generate sample mesh
mesh = MeshGenerator.sphere(radius=1.0, subdivisions=2)
mesh.compute_vertex_normals()

# Initialize tokenizer and model
tokenizer = VertexTokenizer(include_normals=True)
model = MeshTransformer(feature_dim=6, d_model=256, nhead=8)

# Tokenize and classify
tokens = tokenizer.tokenize(mesh)
output = model(tokens, task='classification')
```

### Advanced Training Pipeline

```python
# Configuration
config = {
    'model_type': 'adaptive',
    'tokenizer_type': 'vertex',
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'max_epochs': 100,
    'task_type': 'classification'
}

# Initialize and train
pipeline = MeshTransformerTrainingPipeline(config)
pipeline.train(train_dataset, val_dataset)
```

## 🔧 Core Components

### Tokenization Strategies

- **`VertexTokenizer`**: Treats each vertex as a token with position, normal, and color features
- **`FaceTokenizer`**: Converts faces to tokens with connectivity and geometric properties
- **`PatchTokenizer`**: Groups vertices into local patches for hierarchical processing

### Attention Mechanisms

- **`GeometricAttention`**: Distance-aware attention with geometric bias
- **`GraphAttention`**: Graph neural network style attention for mesh connectivity
- **`MultiScaleAttention`**: Hierarchical attention at multiple resolutions
- **`SparseAttention`**: Efficient attention for large meshes using local neighborhoods
- **`AdaptiveMeshTransformer`**: Dynamically switches attention types based on input

### Model Architectures

- **`MeshTransformer`**: Standard transformer with 3D positional encoding
- **`AdaptiveMeshTransformer`**: Advanced model with multiple attention mechanisms
- **`MeshTransformerLayer`**: Configurable transformer layer with mesh-aware attention

## 🎨 Use Cases

### 3D Shape Classification
```python
model = MeshTransformer(feature_dim=6, d_model=512)
output = model(tokens, task='classification')
```

### Mesh Generation & Completion
```python
model = MeshTransformer(feature_dim=6, d_model=512)
generated = model(partial_tokens, task='generation')
```

### Mesh Segmentation
```python
model = MeshTransformer(feature_dim=6, d_model=512)
segment_labels = model(tokens, task='segmentation')
```

## 📊 Training Features

### Data Augmentation
```python
# Automatic mesh augmentation
augmented_mesh = MeshAugmentation.random_augment(mesh)

# Individual augmentations
rotated = MeshAugmentation.random_rotation(mesh)
scaled = MeshAugmentation.random_scale(mesh)
noisy = MeshAugmentation.add_noise(mesh)
```

### Pre-training Tasks
```python
trainer = MeshTransformerPreTrainer(model, tokenizer)

# Masked mesh modeling
masked_input, positions, targets = trainer.masked_mesh_modeling(tokens)

# Mesh completion
partial_input, complete_target = trainer.mesh_completion_task(partial_tokens, complete_tokens)
```

## 🔬 Advanced Features

### 3D Positional Encoding
- Sinusoidal encoding for 3D coordinates
- Frequency-based spatial embeddings
- Supports both local and global positional information

### Sparse Processing
- Efficient handling of large meshes (1000+ vertices)
- Neighborhood-based sparse attention
- Configurable sparsity patterns

### Multi-Scale Processing
- Hierarchical mesh analysis
- Multiple resolution attention
- Adaptive scale selection

## 📈 Performance

- **Memory Efficient**: Sparse attention reduces memory complexity
- **Scalable**: Handles meshes from 100 to 10,000+ vertices
- **Fast Training**: Optimized batching and caching
- **GPU Accelerated**: Full CUDA support

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by advances in geometric deep learning and transformer architectures
- Built on PyTorch framework
- Thanks to the 3D computer vision and AI research community

## 📚 Citation

```bibtex
@software{meshnet2024,
  title={MeshNet: Transformer Library for 3D Mesh Processing},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/meshnet}
}
```

---

**Made with ❤️ for the 3D AI community**