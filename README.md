# image_classificationImage Classification Project
This project implements and compares three image classification models — **Multiclass SVM**, **Softmax Classifier**, and a **Two-layer Neural Network** — using a simplified subset of the CIFAR-10 dataset (3 classes: Airplane, Automobile, Bird). The goal is to understand model differences, strengths, weaknesses, and the impact of hyperparameter tuning.
Project Objectives
- Implement and evaluate three classifiers:
  - Support Vector Machine (SVM)
  - Softmax Classifier (Multinomial Logistic Regression)
  - Two-layer Neural Network (PyTorch)
- Work with a simplified CIFAR-10 dataset (3 classes).
- Compare performance across models.
- Practice GitHub version control and submission workflow.
Tools & Libraries
- **Python** – Core programming language
- **VS Code** – Development environment
- **GitHub Copilot** – Assisted coding and debugging
- **GitHub** – Version control and submission
- **NumPy, Pandas, Matplotlib** – Data manipulation & visualization
- **scikit-learn** – SVM and Softmax classifiers
- **PyTorch** – Neural network implementation
Project Structure
Image_Classification_Project/ │ ├── image_classification.py   # Main script with all models ├── data/                     # CIFAR-10 dataset (downloaded automatically) ├── README.md                 # Project documentation
How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Image_Classification_Project.git
   cd Image_Classification_Project
   
2. Install dependencies:
pip install numpy matplotlib scikit-learn pandas torch torchvision
3. Run the script:
python image_classification.py
Results
•	SVM: Margin-based classifier, performs well on small datasets but computationally heavy
•	Softmax: Probabilistic outputs, interpretable, efficient for multi-class problems.
•	Neural Network: Captures non-linear features, generally achieves higher accuracy with proper tuning.
•	Performance comparison is visualized with a bar chart at the end of the script
Deliverables
- GitHub Repository: Contains all source code and documentation.
- PowerPoint Presentation: Summarizes methodology, results, and insights.
- Video Walkthrough: Demonstrates code execution and explains results.
- ZIP File Submission: Downloadable repo for evaluation.
Key Learnings
- SVM and Softmax are strong baselines for classification tasks.
- Neural networks outperform linear models when tuned correctly.
- Hyperparameter tuning (learning rate, batch size, hidden units) is critical for performance.
- GitHub provides a professional workflow for version control and submission.


Author
Pavan Kumar
