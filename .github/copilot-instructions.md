# Copilot Instructions for Image Classification Project

## Project Overview
This project compares three classification models on a 3-class subset of CIFAR-10 (Airplane, Automobile, Bird). The codebase is a **single-file monolithic script** (`image_classification.py`) that sequentially loads data, trains three models, and compares performance.

## Architecture & Data Flow

**Data Pipeline:**
1. Load full CIFAR-10 dataset using `torchvision`
2. Filter to 3 classes (indices 0, 1, 2)
3. Reshape 2D images (32×32×3) to 1D vectors for sklearn models
4. Keep tensor format for PyTorch neural network

**Model Implementations:**
- **SVM** (`SVC` from sklearn): Linear kernel, trains on flattened data
- **Softmax** (`LogisticRegression`): Multinomial with LBFGS solver, max_iter=1000
- **Neural Network**: Custom `TwoLayerNN` class (input→100 hidden→3 output) with ReLU activation and softmax output, trained with Adam optimizer (lr=0.001) for 20 epochs

## Key Code Patterns

### Dataset Filtering
```python
selected_classes = [0, 1, 2]
train_indices = [i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]
trainset.data = trainset.data[train_indices]
trainset.targets = [trainset.targets[i] for i in train_indices]
```
Both train and test sets follow this same pattern. When modifying dataset logic, update both consistently.

### Data Preparation for Models
- **sklearn models**: Require flattened data: `X_train.reshape(len(X_train), -1)`
- **PyTorch**: Requires tensor conversion and `dtype=torch.float32` for inputs, `dtype=torch.long` for labels

### Hyperparameter Locations
- SVM: `kernel='linear'` in `SVC()` call
- Softmax: `max_iter=1000` in `LogisticRegression()`
- NN: Hidden size=100, learning rate=0.001, epochs=20 (all hardcoded, no config file)

## Critical Workflows

**Running the Project:**
```bash
pip install numpy matplotlib scikit-learn pandas torch torchvision
python image_classification.py
```

**Dataset Auto-download:** CIFAR-10 downloads automatically to `./data/` on first run. Subsequent runs use cached data.

**Output:** 
- Console prints accuracy scores for each model
- Matplotlib displays sample training image and final bar chart comparing SVM vs Softmax accuracies
- Neural network accuracy is printed but NOT included in final comparison chart (bug or intentional?)

## Common Modifications

**Change Classes:** Modify `selected_classes = [0, 1, 2]` (lines ~24). Requires updating `output_size = len(selected_classes)` automatically works.

**Adjust NN Hyperparameters:** Hidden size at line 77, learning rate at line 80, epochs at line 87.

**Add NN to Comparison:** Extend final bar chart (line 95) to include neural network accuracy—requires computing accuracy on test set (currently only trains, doesn't evaluate).

## Dependencies & Versions
- PyTorch/torchvision: For CIFAR-10 dataset and NN implementation
- scikit-learn: For SVM and Softmax classifiers
- NumPy: Implicit (used by sklearn and torch)
- Matplotlib: For visualization only, non-critical to model training

## Known Gaps & TODOs
- Neural network accuracy not evaluated or displayed in final comparison
- Hyperparameters hardcoded throughout (no configuration system)
- No train/validation split for NN (trains on full training set)
- SVM/Softmax trained in single batch (no epoch-based monitoring like NN)
