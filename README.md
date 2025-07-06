# All-in-AI: AI Model Practice Project

Chinese version available: [README_zh.md](README_zh.md)

This project serves as a practice repository for understanding various AI models through simple code implementations. Each model is accompanied by minimal, explanatory code to demonstrate core concepts.

## Project Structure
- Each model has its own directory with implementation code
- Code includes comments explaining key concepts and algorithms
- Simple examples demonstrating model usage

## 🚀 Featured Models
Explore our collection of hands-on AI implementations, each designed to teach core concepts through practical code:

### 1. linear_nn_fit-01: Configurable Linear Neural Network
- **Key Concepts**: Linear regression, sigmoid activation, dual-layer configurationsor different activation configurations
- **What You'll Learn**: How activation functions and layer depth impact model performance

### 2. linear-classifier-02: Basic Linear Classification
- **Key Concepts**: Binary classification, decision boundary visualization, SGD optimization
- **Visualizations**: ![decision boundary](linear-classifier-02/decision_boundary.png)
- **Core Components**: `nn.Linear`, `nn.Sigmoid`, `nn.BCELoss`

### 3. mnist_classifier-03: MNIST Digit Classification
- **Key Concepts**: Multi-class classification, fully connected networks, ReLU activation
- **Architecture**: 784→128→10 layer configuration with ReLU non-linearity
- **Performance**: ~97% accuracy on test set

### 4. mnist_cnn-04: Convolutional Neural Network
- **Key Concepts**: 2D convolutions, max pooling, feature extraction
- **Architecture**: Two convolutional layers (20→50 filters) followed by fully connected layers
- **What You'll Learn**: How CNNs capture spatial features in images

### 5. mnist_distillation-05: Knowledge Distillation
- **Key Concepts**: Teacher-student models, knowledge transfer, KLDivLoss
- **Visualizations**: ![loss comparison](mnist_distillation-05/loss_comparison.png)
- **Models**: Teacher (complex CNN) and Student (simplified network) implementation

### 6. mnist_gan-06: Generative Adversarial Network
- **Key Concepts**: GAN architecture, adversarial training, image generation
- **Visualizations**: [generated images](mnist_gan-06/generated_images/) (sample outputs across 50 epochs)
- **Components**: Deep convolutional generator and discriminator with BatchNorm

### 7. transformer_gpt2_07: Transformer Text Generation
- **Key Concepts**: Self-attention, causal masking, text generation
- **Architecture**: Mini-GPT implementation with multi-head attention
- **Applications**: Character-level text generation

### 8. transformer_finetune-08: Transformer Fine-tuning
- **Key Concepts**: Transfer learning, emotion classification, model fine-tuning
- **Visualizations**: [accuracy plot](transformer_finetune-08/accuracy-plot.pdf)
  - Loss curves: [loss plot](transformer_finetune-08/loss-plot.pdf)
- **Performance**: 89% accuracy on emotion classification task

## Getting StNavl directory (e.g., `cd linear_nn_fit-01`)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the example: `python main.py`

## Requirements
- Python 3.12.0
- PyTorch 2.0.0+ and torchvision 0.20.0+ (see requirements.txt for details)
- NumPy, Matplotlib

## Contributing
We're continuously expanding this collection with new mini-implementations of cutting-edge AI models. Star this repository to stay updated on new additions and implementations!
Feel free to add implementations of other AI models with clear explanations.