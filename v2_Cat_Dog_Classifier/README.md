# 🚀 Binary Image Classification with a Perceptron From Scratch

A modular, analytical implementation of a single-layer Perceptron built entirely from scratch using pure **Python** and **NumPy**, applied to a binary image classification task (Cats vs. Dogs).

---

## 📁 Project Structure

```text
├── data/                    # HDF5 dataset files (train/test sets)
├── src/
│   ├── utilities.py         # HDF5 data loading, reshaping, and pixel normalization
│   └── perceptron.py        # Core model, log-loss (with numerical stability), and gradients
├── notebooks/
│   └── Cat_Or_Dog_Classifier.ipynb  # Experimentation and training visualization
├── requirements.txt         # Python dependencies
└── README.md
```

## 📊 Dataset & Preprocessing
Format: The data is stored in HDF5 format (.hdf5), containing high-dimensional image arrays and binary labels (trainset.hdf5, testset.hdf5). (Note: The raw dataset was provided as part of the curriculum).

Preprocessing (utilities.py):

Reshaping: Flattening 2D image matrices into 1D feature vectors to match the Perceptron's input requirements.

Normalization: Pixel values are divided by 255.0 to scale them between 0.0 and 1.0, ensuring faster and stable training convergence.

## 📈 Results & Visualizations
During training, metrics are tracked dynamically using tqdm and plotted using Matplotlib:

Loss Curve: Demonstrates a smooth, asymptotic decrease, proving the mathematical correctness of the gradient descent.

Accuracy Plateau: Training accuracy peaks around 56%, while test accuracy hovers near 44-46%.

## ⚠️ Theoretical Insights & Limitations
1. The Curse of Dimensionality
By flattening $64 \times 64$ images into 1D arrays, each data point enters a 4096-dimensional input space. Operating directly on raw pixels in such a high-dimensional space causes data sparsity and prevents the linear model from capturing robust patterns.

2. Limitations of Linear Separability
A single-layer Perceptron can only construct a linear decision boundary. Complex visual datasets (like cats vs. dogs) require spatial hierarchies and non-linear feature representations.

Next Steps: Transitioning from a single neuron to a Multi-Layer Perceptron (MLP) or integrating a Convolutional Feature Extractor or first of all integrating a MultiLayer Perceptron(MLP) to overcome these structural limitations.

## 🛠️ Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/ton-nom-d-utilisateur/nom-de-ton-repo.git](https://github.com/clementlekane7/perceptron-numpy-from-scratch.git)
   cd nom-de-ton-repo
   
2. Install the required dependencies:
    pip install -r requirements.txt

3. Launch Jupyter Lab to explore the training and evaluation notebook:
    jupyter lab notebooks/Cat_Or_Dog_Classifier.ipynb
    
## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
