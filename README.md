# MatrixForge: A Linear Algebra and Machine Learning Library in C

This project implements a simple library for linear algebra and machine learning in C. The main program implements classifying handwritten digits from the MNIST dataset. The implementation includes support for matrix operations, forward propagation, backpropagation, and training using stochastic gradient descent with Leaky ReLU activations.

## Features
- **Matrix and Vector Operations**: Efficient linear algebra functions using multithreading for neural network computations.
- **Feedforward Neural Network**: Supports multi-layer architectures with customizable layer sizes.
- **Backpropagation**: Implements gradient descent to minimize the cost function.
- **Softmax and Cross-Entropy Loss**: For classification tasks.
- **Leaky ReLU Activation**: To mitigate vanishing gradient problems.
- **MNIST Dataset Support**: Reads and processes the MNIST dataset for training and testing.

## Prerequisites
- **C Compiler**: GCC or any modern C compiler.
- **OpenMP**: For parallelized matrix operations.

## Build Instructions

1. Clone the repository:
   ```bash
   git clone <url>
   cd <directory>
   ```

2. Build the project using the provided `Makefile`:
   ```bash
   make
   ```

3. The resulting executable will be located in the `bin` directory:
   ```bash
   ./bin/neural_net
   ```

## Usage

### Training the Neural Network
1. Place the MNIST dataset files in a `dataset` directory:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`

2. Run the training script:
   ```bash
   ./bin/matrix_test
   ```
   The program will:
   - Load the MNIST dataset.
   - Initialize the neural network parameters.
   - Train the network over the specified number of epochs.
   - Evaluate the test accuracy.

### Modifying Network Configuration
The neural network architecture can be customized by changing the `layer_dims` array in `main.c`:
```c
size_t layer_dims[] = {784, 128, 10}; // Example: 784 input, 128 hidden, 10 output
```

## Known Issues
- High learning rates may cause divergence, causing cost to become `-nan`. Finetune the learning rate for better results.
- Training time may be slow for large datasets or architectures due to the single-threaded implementation of some components.

## Future Improvements
- Add support for GPU acceleration.

## Acknowledgments
- **MNIST Dataset**: [Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- **Matrix Operations**: Custom implementation with OpenMP parallelism.
