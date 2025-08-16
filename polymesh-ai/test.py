#!/usr/bin/env python3
"""
Complete demonstration and testing suite for the Mesh Transformers Library
Includes practical examples, benchmarks, and visualization utilities
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your library modules (assuming they're in the same directory)
from mesh_library import Vertex, Face, Mesh, MeshGenerator, MeshLoader, MeshDataset
from mesh_transformers import (MeshToken, VertexTokenizer, FaceTokenizer, PatchTokenizer, 
                               MeshPositionalEncoding, MeshTransformerEmbedding, MeshTransformer)
from mesh_attention import (GeometricAttention, GraphAttention, MultiScaleAttention, 
                           SparseAttention, AdaptiveMeshTransformer)
from mesh_training_pipeline import MeshTransformerDataset, MeshAugmentation, train_mesh_classifier

class MeshTransformersDemo:
    """Complete demonstration suite for mesh transformers"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.results = {}
    
    def generate_sample_meshes(self, n_samples: int = 10) -> Dict[str, List[Mesh]]:
        """Generate diverse sample meshes for testing"""
        print("\n🔧 Generating Sample Meshes...")
        
        meshes = {
            'cubes': [],
            'spheres': [],
            'cylinders': [],
            'planes': []
        }
        
        for i in range(n_samples):
            # Generate cubes with varying sizes
            cube = MeshGenerator.cube(size=np.random.uniform(0.5, 2.0))
            cube.normalize().compute_vertex_normals()
            cube.name = f"cube_{i}"
            meshes['cubes'].append(cube)
            
            # Generate spheres with varying subdivisions
            sphere = MeshGenerator.sphere(
                radius=np.random.uniform(0.8, 1.5),
                subdivisions=np.random.randint(1, 3)
            )
            sphere.normalize().compute_vertex_normals()
            sphere.name = f"sphere_{i}"
            meshes['spheres'].append(sphere)
            
            # Generate cylinders with varying proportions
            cylinder = MeshGenerator.cylinder(
                radius=np.random.uniform(0.5, 1.2),
                height=np.random.uniform(1.0, 3.0),
                segments=np.random.randint(8, 32)
            )
            cylinder.normalize().compute_vertex_normals()
            cylinder.name = f"cylinder_{i}"
            meshes['cylinders'].append(cylinder)
            
            # Generate planes with varying subdivisions
            plane = MeshGenerator.plane(
                width=np.random.uniform(1.0, 3.0),
                height=np.random.uniform(1.0, 3.0),
                width_segments=np.random.randint(1, 4),
                height_segments=np.random.randint(1, 4)
            )
            plane.normalize().compute_vertex_normals()
            plane.name = f"plane_{i}"
            meshes['planes'].append(plane)
        
        total_meshes = sum(len(mesh_list) for mesh_list in meshes.values())
        print(f"✅ Generated {total_meshes} sample meshes")
        
        return meshes
    
    def test_tokenization_strategies(self, meshes: Dict[str, List[Mesh]]):
        """Test different tokenization strategies"""
        print("\n🔍 Testing Tokenization Strategies...")
        
        # Get a sample mesh
        test_mesh = meshes['spheres'][0]
        print(f"Testing with mesh: {test_mesh}")
        
        tokenizers = {
            'vertex': VertexTokenizer(include_normals=True, include_colors=False),
            'face': FaceTokenizer(max_face_vertices=4, include_face_normal=True),
            'patch': PatchTokenizer(patch_size=8, overlap=2)
        }
        
        tokenization_results = {}
        
        for name, tokenizer in tokenizers.items():
            start_time = time.time()
            tokens = tokenizer.tokenize(test_mesh)
            tokenization_time = time.time() - start_time
            
            if tokens:
                avg_feature_dim = np.mean([len(token.features) for token in tokens])
                token_positions = np.array([token.position for token in tokens])
                position_variance = np.var(token_positions, axis=0)
            else:
                avg_feature_dim = 0
                position_variance = np.zeros(3)
            
            tokenization_results[name] = {
                'num_tokens': len(tokens),
                'avg_feature_dim': avg_feature_dim,
                'tokenization_time': tokenization_time,
                'position_variance': position_variance,
                'sample_token': tokens[0] if tokens else None
            }
            
            print(f"  {name.upper()}: {len(tokens)} tokens, "
                  f"avg feature dim: {avg_feature_dim:.1f}, "
                  f"time: {tokenization_time:.4f}s")
        
        self.results['tokenization'] = tokenization_results
        return tokenization_results
    
    def test_attention_mechanisms(self):
        """Test different attention mechanisms"""
        print("\n⚡ Testing Attention Mechanisms...")
        
        # Create sample data
        batch_size, seq_len, d_model = 2, 64, 256
        x = torch.randn(batch_size, seq_len, d_model)
        positions = torch.randn(batch_size, seq_len, 3)
        
        attention_mechanisms = {
            'geometric': GeometricAttention(d_model, nhead=8),
            'multiscale': MultiScaleAttention(d_model, scales=[1, 2, 4]),
            'sparse': SparseAttention(d_model, nhead=8, neighborhood_size=16)
        }
        
        attention_results = {}
        
        for name, attention in attention_mechanisms.items():
            attention.to(self.device)
            x_device = x.to(self.device)
            positions_device = positions.to(self.device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                if name == 'geometric':
                    output = attention(x_device, x_device, x_device, positions_device)
                else:
                    output = attention(x_device, positions_device)
            inference_time = time.time() - start_time
            
            # Calculate memory usage (parameters)
            num_params = sum(p.numel() for p in attention.parameters())
            
            attention_results[name] = {
                'output_shape': tuple(output.shape),
                'inference_time': inference_time,
                'num_parameters': num_params,
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
            
            print(f"  {name.upper()}: {tuple(output.shape)}, "
                  f"params: {num_params:,}, time: {inference_time:.4f}s")
        
        self.results['attention'] = attention_results
        return attention_results
    
    def test_model_architectures(self):
        """Test different model architectures"""
        print("\n🏗️ Testing Model Architectures...")
        
        models = {
            'standard': MeshTransformer(
                feature_dim=6, d_model=256, nhead=8, num_layers=4
            ),
            'adaptive': AdaptiveMeshTransformer(
                d_model=256, nhead=8, num_layers=4
            )
        }
        
        model_results = {}
        
        for name, model in models.items():
            model.to(self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            if name == 'standard':
                # Create sample tokens
                sample_tokens = []
                for i in range(32):
                    token = MeshToken(
                        token_id=i,
                        features=np.random.randn(6).astype(np.float32),
                        position=np.random.randn(3).astype(np.float32),
                        token_type='vertex'
                    )
                    sample_tokens.append(token)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(sample_tokens, task='classification')
                inference_time = time.time() - start_time
                
            else:  # adaptive
                x = torch.randn(1, 32, 256).to(self.device)
                positions = torch.randn(1, 32, 3).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(x, positions)
                inference_time = time.time() - start_time
            
            model_results[name] = {
                'num_parameters': num_params,
                'output_shape': tuple(output.shape),
                'inference_time': inference_time,
                'memory_mb': num_params * 4 / (1024 * 1024)  # Rough estimate
            }
            
            print(f"  {name.upper()}: {num_params:,} params, "
                  f"output: {tuple(output.shape)}, time: {inference_time:.4f}s")
        
        self.results['models'] = model_results
        return model_results
    
    def test_mesh_augmentation(self, meshes: Dict[str, List[Mesh]]):
        """Test mesh augmentation techniques"""
        print("\n🔄 Testing Mesh Augmentation...")
        
        test_mesh = meshes['cubes'][0].copy()
        original_vertices = len(test_mesh.vertices)
        original_center = test_mesh.get_center().copy()
        original_scale = test_mesh.get_scale()
        
        augmentations = {
            'rotation': lambda m: MeshAugmentation.random_rotation(m, angle_range=45),
            'scaling': lambda m: MeshAugmentation.random_scale(m, scale_range=(0.7, 1.3)),
            'translation': lambda m: MeshAugmentation.random_translation(m, translation_range=0.2),
            'noise': lambda m: MeshAugmentation.add_noise(m, noise_std=0.02),
            'combined': lambda m: MeshAugmentation.random_augment(m)
        }
        
        augmentation_results = {}
        
        for name, aug_fn in augmentations.items():
            test_mesh_copy = test_mesh.copy()
            
            start_time = time.time()
            augmented_mesh = aug_fn(test_mesh_copy)
            aug_time = time.time() - start_time
            
            new_center = augmented_mesh.get_center()
            new_scale = augmented_mesh.get_scale()
            center_change = np.linalg.norm(new_center - original_center)
            scale_change = abs(new_scale - original_scale) / original_scale
            
            augmentation_results[name] = {
                'vertices_preserved': len(augmented_mesh.vertices) == original_vertices,
                'center_change': center_change,
                'scale_change': scale_change,
                'augmentation_time': aug_time
            }
            
            print(f"  {name.upper()}: center Δ={center_change:.3f}, "
                  f"scale Δ={scale_change:.3f}, time={aug_time:.4f}s")
        
        self.results['augmentation'] = augmentation_results
        return augmentation_results
    
    def benchmark_performance(self, meshes: Dict[str, List[Mesh]]):
        """Benchmark model performance on different mesh sizes"""
        print("\n⚡ Benchmarking Performance...")
        
        # Test with different mesh complexities
        test_meshes = [
            meshes['cubes'][0],      # Simple
            meshes['spheres'][1],    # Medium
            meshes['cylinders'][0]   # Complex (more vertices)
        ]
        
        tokenizer = VertexTokenizer(include_normals=True)
        model = MeshTransformer(feature_dim=6, d_model=256, nhead=8, num_layers=4)
        model.to(self.device)
        
        benchmark_results = []
        
        for i, mesh in enumerate(test_meshes):
            tokens = tokenizer.tokenize(mesh)
            complexity = len(tokens)
            
            # Warm-up run
            with torch.no_grad():
                _ = model(tokens, task='classification')
            
            # Timed runs
            times = []
            for _ in range(5):
                start_time = time.time()
                with torch.no_grad():
                    output = model(tokens, task='classification')
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = complexity / avg_time  # tokens per second
            
            result = {
                'mesh_name': mesh.name,
                'complexity': complexity,
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'output_shape': tuple(output.shape)
            }
            benchmark_results.append(result)
            
            print(f"  {mesh.name}: {complexity} tokens, "
                  f"{avg_time:.4f}±{std_time:.4f}s, "
                  f"{throughput:.1f} tokens/s")
        
        self.results['benchmark'] = benchmark_results
        return benchmark_results
    
    def visualize_mesh(self, mesh: Mesh, title: str = "Mesh", save_path: Optional[str] = None):
        """Visualize a 3D mesh"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get vertex positions
        positions = mesh.vertex_positions
        
        # Plot vertices
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=20, alpha=0.6, label='Vertices')
        
        # Plot faces as triangles
        for face in mesh.faces:
            if len(face.vertex_indices) >= 3:
                face_verts = positions[face.vertex_indices[:3]]
                # Create triangle
                triangle = np.vstack([face_verts, face_verts[0]])
                ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                       'b-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}\n{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([positions.max()-positions.min()]).max() / 2.0
        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_report(self):
        """Generate a comprehensive performance report"""
        print("\n📊 Generating Performance Report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'results': self.results
        }
        
        # Save report
        report_path = Path('mesh_transformers_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("MESH TRANSFORMERS PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'tokenization' in self.results:
            print("\nTOKENIZATION:")
            for name, data in self.results['tokenization'].items():
                print(f"  {name}: {data['num_tokens']} tokens, "
                      f"{data['tokenization_time']:.4f}s")
        
        if 'models' in self.results:
            print("\nMODELS:")
            for name, data in self.results['models'].items():
                print(f"  {name}: {data['num_parameters']:,} params, "
                      f"{data['memory_mb']:.1f}MB")
        
        if 'benchmark' in self.results:
            print("\nBENCHMARK:")
            for data in self.results['benchmark']:
                print(f"  {data['mesh_name']}: {data['throughput']:.1f} tokens/s")
        
        return report
    
    def run_comprehensive_demo(self):
        """Run the complete demonstration suite"""
        print("🚀 Starting Mesh Transformers Comprehensive Demo")
        print("=" * 60)
        
        # Generate test data
        meshes = self.generate_sample_meshes(n_samples=5)
        
        # Test tokenization
        self.test_tokenization_strategies(meshes)
        
        # Test attention mechanisms
        self.test_attention_mechanisms()
        
        # Test model architectures
        self.test_model_architectures()
        
        # Test augmentation
        self.test_mesh_augmentation(meshes)
        
        # Benchmark performance
        self.benchmark_performance(meshes)
        
        # Visualize sample mesh
        print("\n🎨 Visualizing Sample Mesh...")
        self.visualize_mesh(meshes['spheres'][0], "Sample Sphere")
        
        # Generate report
        self.create_performance_report()
        
        print("\n✅ Demo completed successfully!")

class MeshClassificationExample:
    """Example of training a mesh classifier"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_synthetic_dataset(self, n_per_class: int = 50):
        """Create a synthetic mesh classification dataset"""
        print(f"\n📚 Creating synthetic dataset ({n_per_class} per class)...")
        
        dataset = []
        classes = ['cube', 'sphere', 'cylinder']
        
        for class_idx, mesh_type in enumerate(classes):
            for i in range(n_per_class):
                if mesh_type == 'cube':
                    mesh = MeshGenerator.cube(size=np.random.uniform(0.8, 1.5))
                elif mesh_type == 'sphere':
                    mesh = MeshGenerator.sphere(
                        radius=np.random.uniform(0.8, 1.2),
                        subdivisions=np.random.randint(1, 3)
                    )
                else:  # cylinder
                    mesh = MeshGenerator.cylinder(
                        radius=np.random.uniform(0.6, 1.0),
                        height=np.random.uniform(1.5, 2.5),
                        segments=np.random.randint(12, 24)
                    )
                
                # Normalize and compute normals
                mesh.normalize().compute_vertex_normals()
                
                # Add to dataset
                dataset.append({
                    'mesh': mesh,
                    'label': class_idx,
                    'class_name': mesh_type,
                    'id': f"{mesh_type}_{i}"
                })
        
        print(f"✅ Created dataset with {len(dataset)} samples")
        return dataset, classes
    
    def train_simple_classifier(self):
        """Train a simple mesh classifier"""
        print("\n🎯 Training Simple Mesh Classifier...")
        
        # Create dataset
        dataset, classes = self.create_synthetic_dataset(n_per_class=20)
        
        # Split into train/val
        np.random.shuffle(dataset)
        split_idx = int(0.8 * len(dataset))
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Create simple model and tokenizer
        tokenizer = VertexTokenizer(include_normals=True, include_colors=False)
        model = MeshTransformer(
            feature_dim=6,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256
        ).to(self.device)
        
        # Add proper classification head
        model.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(classes))
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Simple training loop
        model.train()
        for epoch in range(10):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for item in train_data:
                optimizer.zero_grad()
                
                # Tokenize mesh
                tokens = tokenizer.tokenize(item['mesh'])
                if not tokens:
                    continue
                
                # Forward pass
                output = model(tokens, task='classification')
                label = torch.tensor([item['label']], device=self.device)
                
                # Compute loss
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = torch.argmax(output, dim=1)
                correct += (pred == label).sum().item()
                total += 1
            
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_data):.4f}, "
                  f"Acc={accuracy:.1f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for item in val_data:
                tokens = tokenizer.tokenize(item['mesh'])
                if not tokens:
                    continue
                
                output = model(tokens, task='classification')
                label = torch.tensor([item['label']], device=self.device)
                
                pred = torch.argmax(output, dim=1)
                val_correct += (pred == label).sum().item()
                val_total += 1
        
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"✅ Validation Accuracy: {val_accuracy:.1f}%")
        
        return model, tokenizer, classes

def main():
    """Main execution function"""
    print("Mesh Transformers Library - Complete Demo")
    print("=" * 50)
    
    # Run comprehensive demo
    demo = MeshTransformersDemo()
    demo.run_comprehensive_demo()
    
    # Run classification example
    classifier_demo = MeshClassificationExample()
    model, tokenizer, classes = classifier_demo.train_simple_classifier()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("\nNext steps:")
    print("- Experiment with different tokenization strategies")
    print("- Try advanced attention mechanisms")
    print("- Scale up to larger datasets")
    print("- Implement custom mesh processing tasks")

if __name__ == "__main__":
    main()