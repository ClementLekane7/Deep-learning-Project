# 🧠 Perceptron From Scratch with NumPy

A clean, from-scratch implementation of a single-neuron Perceptron using Python and NumPy. This project is built to demonstrate the fundamental mathematical and algorithmic concepts behind Machine Learning, specifically binary classification, logistic loss, and gradient descent optimization.

## 🚀 Features

- **Synthetic Data Generation**: Utilizes `scikit-learn`'s `make_blobs` to generate linearly separable clusters.
- **Pure NumPy Implementation**: No black-box machine learning libraries used for the core model logic—everything from weight initialization to gradient calculation is implemented explicitly.
- **Optimization**: Implements Gradient Descent using Binary Cross-Entropy Loss (Log Loss) and the Sigmoid activation function.
- **Performance Evaluation**: Computes model accuracy and visualizes the convergence via training loss curves.

## 🛠️ Tech Stack

- **Python** (>= 3.8)
- **NumPy** (Matrix operations and vectorized math)
- **Matplotlib** (Loss evolution visualization)
- **Scikit-learn** (Dataset generation and metrics)

## 📦 Getting Started

### Prerequisites

Make sure you have Python installed along with the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Running the Code
 ```Clone the repository and open the Jupyter Notebook or run the script:
git clone [https://github.com/ClementLekane7/perceptron-numpy-from-scratch.git](https://github.com/ClementLekane7/perceptron-numpy-from-scratch.git)
cd perceptron-numpy-from-scratch
```
📈 Results
During training over 100 iterations, the model successfully minimizes the log loss and achieves high classification accuracy on the synthetic dataset.   
Loss curve visualization and decision boundary analysis included in the notebook[cite: 7].
