# complete_training_example.py
# A complete working example for training mesh transformers

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict

# Import your existing modules
from mesh_library import MeshGenerator, Mesh
from mesh_transformers import (
    MeshTransformer, AdaptiveMeshTransformer, VertexTokenizer,
    create_mesh_classifier, create_adaptive_classifier
)

class SimpleMeshDataset:
    """Simplified dataset for testing"""
    
    def __init__(self, num_samples_per_class: int = 100, num_classes: int = 3):
        self.data = []
        self.tokenizer = VertexTokenizer(include_normals=True, include_colors=False)
        
        print(f"Generating dataset with {num_samples_per_class} samples per class...")
        
        # Generate different mesh types
        mesh_generators = {
            0: lambda: MeshGenerator.cube(size=np.random.uniform(0.5, 2.0)),
            1: lambda: MeshGenerator.sphere(radius=np.random.uniform(0.5, 1.5), 
                                         subdivisions=np.random.randint(1, 3)),
            2: lambda: MeshGenerator.cylinder(radius=np.random.uniform(0.5, 1.2),
                                            height=np.random.uniform(1.0, 3.0))
        }
        
        for class_id in range(num_classes):
            for sample_id in range(num_samples_per_class):
                # Generate mesh
                mesh = mesh_generators[class_id]()
                mesh.normalize().compute_vertex_normals()
                
                # Apply random augmentation
                mesh = self._augment_mesh(mesh)
                
                # Tokenize
                tokens = self.tokenizer.tokenize(mesh)
                
                # Limit sequence length
                if len(tokens) > 512:
                    tokens = tokens[:512]
                
                self.data.append({
                    'tokens': tokens,
                    'label': class_id,
                    'mesh': mesh
                })
    
    def _augment_mesh(self, mesh: Mesh) -> Mesh:
        """Simple augmentation"""
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30) * np.pi / 180
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = (cos_angle * np.eye(3) + 
                              sin_angle * np.array([[0, -axis[2], axis[1]],
                                                  [axis[2], 0, -axis[0]],
                                                  [-axis[1], axis[0], 0]]) +
                              (1 - cos_angle) * np.outer(axis, axis))
            
            mesh.rotate(rotation_matrix)
        
        # Random scale
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            mesh.scale(scale)
        
        # Random noise
        if np.random.rand() > 0.3:
            for vertex in mesh.vertices:
                noise = np.random.normal(0, 0.01, 3)
                vertex.position += noise
            mesh._invalidate_caches()
        
        return mesh
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for batching mesh data"""
    batch_tokens = [item['tokens'] for item in batch]
    batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    if len(batch_tokens) == 0:
        return {
            'features': torch.empty(0, 0, 6),
            'positions': torch.empty(0, 0, 3),
            'attention_mask': torch.empty(0, 0, dtype=torch.bool),
            'labels': torch.empty(0, dtype=torch.long)
        }
    
    # Get max sequence length
    max_len = max(len(tokens) for tokens in batch_tokens)
    if max_len == 0:
        max_len = 1
    
    # Pad sequences
    padded_features = []
    padded_positions = []
    attention_masks = []
    
    for tokens in batch_tokens:
        if len(tokens) == 0:
            # Handle empty token sequences
            padded_feat = np.zeros((max_len, 6), dtype=np.float32)
            padded_pos = np.zeros((max_len, 3), dtype=np.float32)
            mask = np.zeros(max_len, dtype=bool)
        else:
            # Extract features and positions
            features = np.array([token.features for token in tokens])
            positions = np.array([token.position for token in tokens])
            
            seq_len = len(tokens)
            
            # Ensure features have correct dimension (position + normal = 6)
            if features.shape[1] < 6:
                # Pad with zeros if missing features
                padding = np.zeros((seq_len, 6 - features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > 6:
                # Truncate if too many features
                features = features[:, :6]
            
            # Pad sequences to max length
            padded_feat = np.zeros((max_len, 6), dtype=np.float32)
            padded_pos = np.zeros((max_len, 3), dtype=np.float32)
            mask = np.zeros(max_len, dtype=bool)
            
            padded_feat[:seq_len] = features
            padded_pos[:seq_len] = positions
            mask[:seq_len] = True
        
        padded_features.append(padded_feat)
        padded_positions.append(padded_pos)
        attention_masks.append(mask)
    
    return {
        'features': torch.tensor(np.array(padded_features), dtype=torch.float32),
        'positions': torch.tensor(np.array(padded_positions), dtype=torch.float32),
        'attention_mask': torch.tensor(np.array(attention_masks), dtype=torch.bool),
        'labels': batch_labels
    }

def train_model():
    """Complete training function"""
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'num_classes': 3,
        'feature_dim': 6,  # 3D position + 3D normal
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'weight_decay': 0.01,
        'grad_clip': 1.0
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = SimpleMeshDataset(num_samples_per_class=50, num_classes=3)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_data = full_dataset.data[:train_size]
    val_data = full_dataset.data[train_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = create_adaptive_classifier(
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs']
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for batch in train_pbar:
            optimizer.zero_grad()
            
            # Move to device
            features = batch['features'].to(device)
            positions = batch['positions'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Skip empty batches
            if features.size(0) == 0:
                continue
            
            # Forward pass
            try:
                outputs = model(features, positions, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            for batch in val_pbar:
                # Move to device
                features = batch['features'].to(device)
                positions = batch['positions'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Skip empty batches
                if features.size(0) == 0:
                    continue
                
                try:
                    # Forward pass
                    outputs = model(features, positions, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    # Predictions
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Update metrics
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Update progress bar
                    current_acc = 100 * val_correct / val_total
                    val_pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'config': config
            }, 'best_mesh_model.pth')
            print(f"  New best model saved! (Accuracy: {val_accuracy:.2f}%)")
        
        print("-" * 60)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }, 'final_mesh_model.pth')
    
    return model, train_losses, val_accuracies

def test_single_prediction():
    """Test the trained model on a single mesh"""
    
    print("\nTesting single prediction...")
    
    # Load the best model
    if not os.path.exists('best_mesh_model.pth'):
        print("No trained model found. Please run training first.")
        return
    
    checkpoint = torch.load('best_mesh_model.pth', map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = create_adaptive_classifier(
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device(config['device'])
    model.to(device)
    
    # Create test meshes
    test_meshes = {
        'cube': MeshGenerator.cube(size=1.5),
        'sphere': MeshGenerator.sphere(radius=1.0, subdivisions=2),
        'cylinder': MeshGenerator.cylinder(radius=0.8, height=2.0)
    }
    
    tokenizer = VertexTokenizer(include_normals=True, include_colors=False)
    class_names = ['Cube', 'Sphere', 'Cylinder']
    
    print("\nPredictions:")
    print("-" * 40)
    
    with torch.no_grad():
        for mesh_name, mesh in test_meshes.items():
            # Preprocess mesh
            mesh.normalize().compute_vertex_normals()
            
            # Tokenize
            tokens = tokenizer.tokenize(mesh)
            if len(tokens) > 512:
                tokens = tokens[:512]
            
            if len(tokens) == 0:
                print(f"{mesh_name}: No tokens generated")
                continue
            
            # Convert to batch format
            features = np.array([token.features for token in tokens])
            positions = np.array([token.position for token in tokens])
            
            # Ensure correct feature dimension
            if features.shape[1] < 6:
                padding = np.zeros((len(tokens), 6 - features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > 6:
                features = features[:, :6]
            
            # Add batch dimension
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            positions = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).to(device)
            mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)
            
            # Predict
            outputs = model(features, positions, mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            print(f"{mesh_name:10} -> {class_names[predicted_class]:8} (confidence: {confidence:.3f})")
            
            # Show all probabilities
            print(f"{'':12} Probabilities: ", end="")
            for i, prob in enumerate(probabilities[0]):
                print(f"{class_names[i]}: {prob.item():.3f} ", end="")
            print()

def visualize_training_progress():
    """Plot training progress if matplotlib is available"""
    try:
        import matplotlib.pyplot as plt
        
        if not os.path.exists('final_mesh_model.pth'):
            print("No training data found.")
            return
        
        checkpoint = torch.load('final_mesh_model.pth', map_location='cpu')
        train_losses = checkpoint['train_losses']
        val_accuracies = checkpoint['val_accuracies']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
        
        print("Training progress saved as 'training_progress.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

if __name__ == "__main__":
    print("Mesh Transformer Training Example")
    print("=" * 50)
    
    # Train the model
    try:
        model, train_losses, val_accuracies = train_model()
        
        # Test predictions
        test_single_prediction()
        
        # Visualize progress
        visualize_training_progress()
        
        print("\nTraining and testing completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()