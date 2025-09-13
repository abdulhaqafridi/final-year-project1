# import numpy as np
# import pandas as pd
# import pickle
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# import joblib
# import time
# import warnings
# warnings.filterwarnings('ignore')

# # Check GPU availability
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# class EEGNeuralNetwork(nn.Module):
#     """Deep Neural Network for EEG classification"""
#     def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
#         super(EEGNeuralNetwork, self).__init__()
        
#         layers = []
#         prev_size = input_size
        
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.BatchNorm1d(hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_rate)
#             ])
#             prev_size = hidden_size
        
#         # Output layer
#         layers.append(nn.Linear(prev_size, 1))
#         layers.append(nn.Sigmoid())
        
#         self.network = nn.Sequential(*layers)
        
#     def forward(self, x):
#         return self.network(x)

# class GPUEEGModelTrainer:
#     def __init__(self, features_df):
#         self.features_df = features_df
#         self.X = None
#         self.y = None
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
#         self.scaler = None
#         self.models = {}
#         self.results = {}
#         self.feature_names = None
#         self.device = device
        
#     def prepare_data(self, test_size=0.2, random_state=42):
#         """Prepare data for training"""
        
#         # Select feature columns (exclude metadata)
#         metadata_cols = ['patient', 'file', 'channel']
#         feature_cols = [col for col in self.features_df.columns 
#                        if col not in metadata_cols + ['label']]
        
#         self.feature_names = feature_cols
        
#         # Prepare features and labels
#         self.X = self.features_df[feature_cols].values
#         self.y = self.features_df['label'].values
        
#         # Handle any NaN or infinite values
#         self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Split data
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=test_size, random_state=random_state, 
#             stratify=self.y
#         )
        
#         # Scale features
#         self.scaler = StandardScaler()
#         self.X_train_scaled = self.scaler.fit_transform(self.X_train)
#         self.X_test_scaled = self.scaler.transform(self.X_test)
        
#         print(f"Data prepared:")
#         print(f"Training set: {self.X_train.shape[0]} samples")
#         print(f"Test set: {self.X_test.shape[0]} samples")
#         print(f"Features: {self.X_train.shape[1]}")
#         print(f"Class distribution in training set:")
#         print(f"  Non-seizure (0): {np.sum(self.y_train == 0)} ({np.mean(self.y_train == 0)*100:.1f}%)")
#         print(f"  Seizure (1): {np.sum(self.y_train == 1)} ({np.mean(self.y_train == 1)*100:.1f}%)")
        
#     def train_neural_network(self, epochs=100, batch_size=64, lr=0.001):
#         """Train deep neural network on GPU"""
        
#         print(f"\nTraining Neural Network on {self.device}...")
        
#         # Convert to tensors
#         X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(self.device)
#         y_train_tensor = torch.FloatTensor(self.y_train.reshape(-1, 1)).to(self.device)
#         X_test_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
#         y_test_tensor = torch.FloatTensor(self.y_test.reshape(-1, 1)).to(self.device)
        
#         # Create datasets
#         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
#         # Initialize model
#         model = EEGNeuralNetwork(input_size=self.X_train_scaled.shape[1]).to(self.device)
#         criterion = nn.BCELoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
#         # Training loop
#         train_losses = []
#         start_time = time.time()
        
#         for epoch in range(epochs):
#             model.train()
#             epoch_loss = 0.0
            
#             for batch_X, batch_y in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
            
#             avg_loss = epoch_loss / len(train_loader)
#             train_losses.append(avg_loss)
#             scheduler.step(avg_loss)
            
#             if (epoch + 1) % 20 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
#         training_time = time.time() - start_time
#         print(f"Neural Network training completed in {training_time:.2f} seconds")
        
#         # Evaluation
#         model.eval()
#         with torch.no_grad():
#             y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
#             y_pred = (y_pred_proba > 0.5).astype(int)
        
#         # Calculate metrics
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred)
#         recall = recall_score(self.y_test, y_pred)
#         f1 = f1_score(self.y_test, y_pred)
#         roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
#         # Store results
#         self.results['Neural Network'] = {
#             'model': model,
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'roc_auc': roc_auc,
#             'y_pred': y_pred,
#             'y_pred_proba': y_pred_proba,
#             'confusion_matrix': confusion_matrix(self.y_test, y_pred),
#             'training_time': training_time
#         }
        
#         print(f"Neural Network Results:")
#         print(f"  Accuracy: {accuracy:.4f}")
#         print(f"  Precision: {precision:.4f}")
#         print(f"  Recall: {recall:.4f}")
#         print(f"  F1-Score: {f1:.4f}")
#         print(f"  ROC AUC: {roc_auc:.4f}")
#         print(f"  Training Time: {training_time:.2f}s")
        
#     def initialize_sklearn_models(self):
#         """Initialize sklearn models with GPU-optimized parameters"""
        
#         self.models = {
#             'Random Forest': RandomForestClassifier(
#                 n_estimators=200,
#                 max_depth=15,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 random_state=42,
#                 n_jobs=-1  # Use all CPU cores
#             ),
#             'SVM': SVC(
#                 kernel='rbf',
#                 C=10.0,
#                 gamma='scale',
#                 probability=True,
#                 random_state=42
#             ),
#             'Logistic Regression': LogisticRegression(
#                 C=1.0,
#                 max_iter=2000,
#                 random_state=42,
#                 n_jobs=-1
#             )
#         }
        
#         print("Initialized sklearn models:", list(self.models.keys()))
        
#     def train_sklearn_models(self):
#         """Train sklearn models with timing"""
        
#         for name, model in self.models.items():
#             print(f"\nTraining {name}...")
#             start_time = time.time()
            
#             # Train model
#             model.fit(self.X_train_scaled, self.y_train)
#             training_time = time.time() - start_time
            
#             # Predictions
#             y_pred = model.predict(self.X_test_scaled)
#             y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
#             # Calculate metrics
#             accuracy = accuracy_score(self.y_test, y_pred)
#             precision = precision_score(self.y_test, y_pred)
#             recall = recall_score(self.y_test, y_pred)
#             f1 = f1_score(self.y_test, y_pred)
#             roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            
#             # Store results
#             self.results[name] = {
#                 'model': model,
#                 'accuracy': accuracy,
#                 'precision': precision,
#                 'recall': recall,
#                 'f1_score': f1,
#                 'roc_auc': roc_auc,
#                 'y_pred': y_pred,
#                 'y_pred_proba': y_pred_proba,
#                 'confusion_matrix': confusion_matrix(self.y_test, y_pred),
#                 'training_time': training_time
#             }
            
#             print(f"{name} Results:")
#             print(f"  Accuracy: {accuracy:.4f}")
#             print(f"  Precision: {precision:.4f}")
#             print(f"  Recall: {recall:.4f}")
#             print(f"  F1-Score: {f1:.4f}")
#             if roc_auc:
#                 print(f"  ROC AUC: {roc_auc:.4f}")
#             print(f"  Training Time: {training_time:.2f}s")
            
#     def cross_validate_neural_network(self, k_folds=5, epochs=50):
#         """Perform k-fold cross-validation for neural network"""
        
#         print(f"\nPerforming {k_folds}-fold cross-validation for Neural Network...")
        
#         kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
#         cv_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_scaled, self.y_train)):
#             print(f"Fold {fold + 1}/{k_folds}")
            
#             # Split data
#             X_train_fold = torch.FloatTensor(self.X_train_scaled[train_idx]).to(self.device)
#             y_train_fold = torch.FloatTensor(self.y_train[train_idx].reshape(-1, 1)).to(self.device)
#             X_val_fold = torch.FloatTensor(self.X_train_scaled[val_idx]).to(self.device)
#             y_val_fold = self.y_train[val_idx]
            
#             # Create data loader
#             fold_dataset = TensorDataset(X_train_fold, y_train_fold)
#             fold_loader = DataLoader(fold_dataset, batch_size=64, shuffle=True)
            
#             # Initialize model
#             model = EEGNeuralNetwork(input_size=self.X_train_scaled.shape[1]).to(self.device)
#             criterion = nn.BCELoss()
#             optimizer = optim.Adam(model.parameters(), lr=0.001)
            
#             # Train
#             model.train()
#             for epoch in range(epochs):
#                 for batch_X, batch_y in fold_loader:
#                     optimizer.zero_grad()
#                     outputs = model(batch_X)
#                     loss = criterion(outputs, batch_y)
#                     loss.backward()
#                     optimizer.step()
            
#             # Evaluate
#             model.eval()
#             with torch.no_grad():
#                 val_pred_proba = model(X_val_fold).cpu().numpy().flatten()
#                 val_pred = (val_pred_proba > 0.5).astype(int)
            
#             fold_f1 = f1_score(y_val_fold, val_pred)
#             cv_scores.append(fold_f1)
#             print(f"  Fold {fold + 1} F1-Score: {fold_f1:.4f}")
        
#         cv_mean = np.mean(cv_scores)
#         cv_std = np.std(cv_scores)
        
#         print(f"Neural Network CV Results:")
#         print(f"  Mean F1-Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
#         # Update neural network results with CV scores
#         if 'Neural Network' in self.results:
#             self.results['Neural Network']['cv_mean'] = cv_mean
#             self.results['Neural Network']['cv_std'] = cv_std
        
#         return cv_mean, cv_std
    
#     def save_models(self, output_dir="models"):
#         """Save all trained models"""
        
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save scaler
#         joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
#         # Save feature names
#         with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
#             pickle.dump(self.feature_names, f)
        
#         # Save sklearn models
#         for name, result in self.results.items():
#             if name != 'Neural Network':
#                 model_filename = f'{name.lower().replace(" ", "_")}_model.pkl'
#                 joblib.dump(result['model'], os.path.join(output_dir, model_filename))
        
#         # Save neural network
#         if 'Neural Network' in self.results:
#             torch.save(self.results['Neural Network']['model'].state_dict(), 
#                       os.path.join(output_dir, 'neural_network_model.pth'))
            
#             # Save model architecture info
#             model_info = {
#                 'input_size': self.X_train_scaled.shape[1],
#                 'hidden_sizes': [256, 128, 64],
#                 'dropout_rate': 0.3
#             }
#             with open(os.path.join(output_dir, 'neural_network_info.pkl'), 'wb') as f:
#                 pickle.dump(model_info, f)
        
#         # Find and save best model
#         best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
#         best_f1_score = self.results[best_model_name]['f1_score']
        
#         if best_model_name != 'Neural Network':
#             joblib.dump(self.results[best_model_name]['model'], 
#                        os.path.join(output_dir, 'best_model.pkl'))
#         else:
#             torch.save(self.results[best_model_name]['model'].state_dict(),
#                        os.path.join(output_dir, 'best_model.pth'))
        
#         # Save results summary
#         results_summary = {}
#         for name, result in self.results.items():
#             results_summary[name] = {
#                 'accuracy': result['accuracy'],
#                 'precision': result['precision'],
#                 'recall': result['recall'],
#                 'f1_score': result['f1_score'],
#                 'roc_auc': result['roc_auc'],
#                 'training_time': result['training_time']
#             }
#             if 'cv_mean' in result:
#                 results_summary[name]['cv_mean'] = result['cv_mean']
#                 results_summary[name]['cv_std'] = result['cv_std']
        
#         with open(os.path.join(output_dir, 'results_summary.pkl'), 'wb') as f:
#             pickle.dump(results_summary, f)
        
#         print(f"\nModels saved to '{output_dir}/' directory")
#         print(f"Best model: {best_model_name} (F1-Score: {best_f1_score:.4f})")
        
#         return best_model_name, best_f1_score
        
#     def generate_training_report(self, output_dir="processed_data"):
#         """Generate comprehensive training report"""
        
#         os.makedirs(output_dir, exist_ok=True)
        
#         report_text = "EEG SEIZURE DETECTION - GPU TRAINING REPORT\n"
#         report_text += "=" * 60 + "\n\n"
        
#         # System information
#         report_text += f"System Information:\n"
#         report_text += f"  Device: {self.device}\n"
#         if torch.cuda.is_available():
#             report_text += f"  GPU: {torch.cuda.get_device_name(0)}\n"
#             report_text += f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
#         report_text += "\n"
        
#         # Dataset summary
#         report_text += f"Dataset Summary:\n"
#         report_text += f"  Total samples: {len(self.features_df)}\n"
#         report_text += f"  Training samples: {len(self.y_train)}\n"
#         report_text += f"  Test samples: {len(self.y_test)}\n"
#         report_text += f"  Features: {self.X_train.shape[1]}\n"
#         report_text += f"  Class distribution (test set):\n"
#         report_text += f"    Non-seizure: {np.sum(self.y_test == 0)} ({np.mean(self.y_test == 0)*100:.1f}%)\n"
#         report_text += f"    Seizure: {np.sum(self.y_test == 1)} ({np.mean(self.y_test == 1)*100:.1f}%)\n\n"
        
#         # Model results
#         report_text += "Model Performance:\n"
#         report_text += "-" * 50 + "\n"
        
#         # Sort models by F1 score
#         sorted_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
#         for name, result in sorted_models:
#             report_text += f"\n{name}:\n"
#             report_text += f"  Accuracy:      {result['accuracy']:.4f}\n"
#             report_text += f"  Precision:     {result['precision']:.4f}\n"
#             report_text += f"  Recall:        {result['recall']:.4f}\n"
#             report_text += f"  F1-Score:      {result['f1_score']:.4f}\n"
#             if result['roc_auc']:
#                 report_text += f"  ROC AUC:       {result['roc_auc']:.4f}\n"
#             report_text += f"  Training Time: {result['training_time']:.2f}s\n"
#             if 'cv_mean' in result:
#                 report_text += f"  CV F1:         {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n"
        
#         # Best model
#         best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
#         report_text += f"\nBest Model: {best_model_name}\n"
#         report_text += f"Best F1-Score: {self.results[best_model_name]['f1_score']:.4f}\n"
        
#         # Performance summary table
#         report_text += f"\nPerformance Summary:\n"
#         report_text += f"{'Model':<20} {'F1-Score':<10} {'Accuracy':<10} {'Time(s)':<10}\n"
#         report_text += f"{'-'*50}\n"
#         for name, result in sorted_models:
#             report_text += f"{name:<20} {result['f1_score']:<10.4f} {result['accuracy']:<10.4f} {result['training_time']:<10.2f}\n"
        
#         # Save report
#         with open(os.path.join(output_dir, 'gpu_training_report.txt'), 'w') as f:
#             f.write(report_text)
        
#         print("\nTraining Report:")
#         print(report_text)
        
#         return report_text

# def main():
#     """Main function to run GPU-optimized training pipeline"""
    
#     try:
#         # Create output directories
#         os.makedirs('processed_data', exist_ok=True)
#         os.makedirs('models', exist_ok=True)
        
#         # Load features
#         print("Loading features...")
#         if os.path.exists('processed_data/eeg_features.pkl'):
#             with open('processed_data/eeg_features.pkl', 'rb') as f:
#                 features_df = pickle.load(f)
#         elif os.path.exists('processed_data/eeg_features.csv'):
#             features_df = pd.read_csv('processed_data/eeg_features.csv')
#         else:
#             raise FileNotFoundError("No feature file found. Please run feature extraction first.")
        
#         print(f"Loaded features: {features_df.shape}")
        
#         # Check if label column exists
#         if 'label' not in features_df.columns:
#             raise ValueError("No 'label' column found in features dataframe")
        
#         # Initialize trainer
#         trainer = GPUEEGModelTrainer(features_df)
        
#         # Prepare data
#         print("\n" + "="*50)
#         print("PREPARING DATA")
#         print("="*50)
#         trainer.prepare_data(test_size=0.2, random_state=42)
        
#         # Train neural network on GPU
#         print("\n" + "="*50)
#         print("TRAINING NEURAL NETWORK ON GPU")
#         print("="*50)
#         trainer.train_neural_network(epochs=100, batch_size=64, lr=0.001)
        
#         # Cross-validate neural network
#         print("\n" + "="*50)
#         print("CROSS-VALIDATING NEURAL NETWORK")
#         print("="*50)
#         trainer.cross_validate_neural_network(k_folds=5, epochs=50)
        
#         # Train sklearn models
#         print("\n" + "="*50)
#         print("TRAINING SKLEARN MODELS")
#         print("="*50)
#         trainer.initialize_sklearn_models()
#         trainer.train_sklearn_models()
        
#         # Save models
#         print("\n" + "="*50)
#         print("SAVING MODELS")
#         print("="*50)
#         best_model_name, best_score = trainer.save_models()
        
#         # Generate report
#         print("\n" + "="*50)
#         print("GENERATING REPORT")
#         print("="*50)
#         trainer.generate_training_report()
        
#         print("\n" + "="*60)
#         print("GPU TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
#         print("="*60)
#         print(f"Best Model: {best_model_name}")
#         print(f"Best F1-Score: {best_score:.4f}")
#         print("\nOutputs saved to:")
#         print("  - processed_data/ (reports)")
#         print("  - models/ (trained models)")
#         print("="*60)
        
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Please ensure that feature extraction has been completed first.")
        
#     except Exception as e:
#         print(f"An error occurred during training: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# import numpy as np
# import pickle
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # 1. Load and Prepare Raw Segments
# def load_raw_segments():
#     with open('processed_data/seizure_segments.pkl', 'rb') as f:
#         seizure_segments = pickle.load(f)
#     with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
#         non_seizure_segments = pickle.load(f)

#     all_segments = seizure_segments + non_seizure_segments
#     labels = [1] * len(seizure_segments) + [0] * len(non_seizure_segments)

#     X = np.array([s['data'] for s in all_segments])  # (n_samples, time_steps)
#     y = np.array(labels)  # (n_samples,)

#     X = X[:, :, np.newaxis]  # (n_samples, time_steps, 1)

#     return X, y

# # 2. Define CNN-LSTM Model
# class CNNLSTM(nn.Module):
#     def __init__(self, input_shape):
#         super(CNNLSTM, self).__init__()
#         time_steps, features = input_shape

#         self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=7, padding=3)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2)

#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2)

#         self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

#         self.fc = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (batch, features, time)
#         x = self.pool1(self.relu1(self.conv1(x)))  # -> (batch, 32, time/2)
#         x = self.pool2(self.relu2(self.conv2(x)))  # -> (batch, 64, time/4)

#         x = x.permute(0, 2, 1)  # (batch, time, features)
#         self.lstm.flatten_parameters()
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Last timestep
#         x = self.fc(x)
#         return x

# # 3. Training and Evaluation

# def train_cnn_lstm(X, y, epochs=20, batch_size=64, lr=0.001):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

#     train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

#     model = CNNLSTM(input_shape=(X.shape[1], X.shape[2])).to(device)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     print("\nStarting training...\n")
#     start_time = time.time()
#     train_losses = []
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch_x, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         train_losses.append(avg_loss)
#         print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

#     training_time = time.time() - start_time
#     print(f"\n✅ Training completed in {training_time:.2f} seconds.")

#     # Plot training loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('processed_data/training_loss_curve.png')
#     plt.close()

#     # Evaluation with mini-batches
#     model.eval()
#     y_pred_proba = []
#     with torch.no_grad():
#         for xb, _ in test_loader:
#             out = model(xb).cpu().numpy().flatten()
#             y_pred_proba.extend(out)

#     y_pred_proba = np.array(y_pred_proba)
#     y_pred = (y_pred_proba > 0.5).astype(int)

#     print("\n📊 Evaluation Results:")
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
#     cm = confusion_matrix(y_test, y_pred)

#     print(f"Accuracy:  {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1 Score:  {f1:.4f}")
#     print(f"ROC AUC:   {roc_auc:.4f}")
#     print(f"Confusion Matrix:\n{cm}")

#     # Plot confusion matrix
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Seizure', 'Seizure'], yticklabels=['Non-Seizure', 'Seizure'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig('processed_data/confusion_matrix.png')
#     plt.close()

#     return model

# if __name__ == "__main__":
#     print("Loading raw EEG segments...")
#     X, y = load_raw_segments()
#     print(f"Loaded {X.shape[0]} segments with shape {X.shape[1:]}\n")

#     os.makedirs('processed_data', exist_ok=True)
#     model = train_cnn_lstm(X, y, epochs=20, batch_size=64, lr=0.001)



import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and Prepare Raw Segments
def load_raw_segments():
    with open('processed_data/seizure_segments.pkl', 'rb') as f:
        seizure_segments = pickle.load(f)
    with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
        non_seizure_segments = pickle.load(f)

    all_segments = seizure_segments + non_seizure_segments
    labels = [1] * len(seizure_segments) + [0] * len(non_seizure_segments)

    X = np.array([s['data'] for s in all_segments])  # (n_samples, time_steps)
    y = np.array(labels)  # (n_samples,)

    X = X[:, :, np.newaxis]  # (n_samples, time_steps, 1)

    return X, y

class CNNLSTM(nn.Module):
    def __init__(self, input_shape):
        super(CNNLSTM, self).__init__()
        time_steps, features = input_shape

        self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Output shape will be (batch, 64, time_steps / 4)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # More regularization
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)                      # (batch, features, time)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))  # -> (batch, 32, time/2)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))  # -> (batch, 64, time/4)

        x = x.permute(0, 2, 1)                      # (batch, time, features)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]                             # Use last hidden state
        x = self.fc(x)
        return x

# 3. Training and Evaluation

def train_cnn_lstm(X, y, epochs=20, batch_size=64, lr=0.001):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    model = CNNLSTM(input_shape=(X.shape[1], X.shape[2])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nStarting training...\n")
    start_time = time.time()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_out = model(val_x)
                val_loss = criterion(val_out, val_y)
                total_val_loss += val_loss.item()

                preds = (val_out > 0.5).float()
                correct_val += (preds == val_y).sum().item()
                total_val += val_y.size(0)

        val_acc = correct_val / total_val
        val_loss = total_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds.")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('processed_data/loss_accuracy_curve.png')
    plt.close()

    # Evaluation with mini-batches
    model.eval()
    y_pred_proba = []
    with torch.no_grad():
        for xb, _ in val_loader:
            out = model(xb).cpu().numpy().flatten()
            y_pred_proba.extend(out)

    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\n📊 Evaluation Results:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Seizure', 'Seizure'], yticklabels=['Non-Seizure', 'Seizure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('processed_data/confusion_matrix.png')
    plt.close()

    return model

if __name__ == "__main__":
    print("Loading raw EEG segments...")
    X, y = load_raw_segments()
    print(f"Loaded {X.shape[0]} segments with shape {X.shape[1:]}\n")

    os.makedirs('processed_data', exist_ok=True)
    model = train_cnn_lstm(X, y, epochs=50, batch_size=64, lr=0.001)
     
   
    
      