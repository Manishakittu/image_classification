import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Step 1: Load and Prepare Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select 3 classes (Airplane=0, Automobile=1, Bird=2)
selected_classes = [0, 1, 2]

train_indices = [i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]
test_indices = [i for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]

trainset.data = trainset.data[train_indices]
trainset.targets = [trainset.targets[i] for i in train_indices]

testset.data = testset.data[test_indices]
testset.targets = [testset.targets[i] for i in test_indices]

# Display a sample image
plt.imshow(trainset.data[0])
plt.title(f"Sample Image - Class {trainset.targets[0]}")
plt.axis("off")
plt.show()

# -----------------------------
# Step 2: Prepare Data for Classifiers
# -----------------------------
X_train = trainset.data.reshape(len(trainset.data), -1)
y_train = trainset.targets

X_test = testset.data.reshape(len(testset.data), -1)
y_test = testset.targets

# -----------------------------
# Step 3: Multiclass SVM
# -----------------------------
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# -----------------------------
# Step 4: Softmax Classifier
# -----------------------------
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax.fit(X_train, y_train)
y_pred_softmax = softmax.predict(X_test)
softmax_accuracy = accuracy_score(y_test, y_pred_softmax)
print(f"Softmax Accuracy: {softmax_accuracy:.4f}")

# -----------------------------
# Step 5: Two-layer Neural Network
# -----------------------------
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

input_size = 32 * 32 * 3
hidden_size = 100
output_size = len(selected_classes)

model = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

for epoch in range(20):  # keep epochs small for demo
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Neural Network Training Completed!")

# -----------------------------
# Step 6: Compare Performance
# -----------------------------
plt.bar(["SVM", "Softmax"], [svm_accuracy, softmax_accuracy])
plt.ylabel("Accuracy")
plt.title("Classifier Comparison")
plt.show()