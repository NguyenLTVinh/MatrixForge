#include "neuralnet.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

double relu(double z) {
    return (z > 0) ? z : 0;
}

double relu_derivative(double z) {
    return (z > 0) ? 1 : 0;
}

double leaky_relu(double z) {
    return (z > 0) ? z : 0.01 * z;
}

double leaky_relu_derivative(double z) {
    return (z > 0) ? 1.0 : 0.01;
}

void softmax(Matrix* Z, Matrix* A) {
    for (size_t i = 0; i < Z->cols; i++) {
        double sum_exp = 0.0;

        // Sum of exponentials for the column
        for (size_t j = 0; j < Z->rows; j++) {
            A->data[j * A->cols + i] = exp(Z->data[j * Z->cols + i]);
            sum_exp += A->data[j * A->cols + i];
        }

        // Normalize
        for (size_t j = 0; j < Z->rows; j++) {
            A->data[j * A->cols + i] /= sum_exp;
        }
    }
}

void init_params(Matrix** weights, Matrix** biases, size_t* layer_dims, size_t L) {
    for (size_t l = 1; l < L; l++) {
        weights[l] = create_matrix(layer_dims[l], layer_dims[l - 1]);
        if (!weights[l]) {
            fprintf(stderr, "Failed to allocate weights for layer %zu.\n", l);
            for (size_t k = 1; k < l; k++) {
                free_matrix(weights[k]);
                free_matrix(biases[k]);
            }
            return;
        }

        biases[l] = create_constant_matrix(layer_dims[l], 1, 0.0);
        if (!biases[l]) {
            fprintf(stderr, "Failed to allocate biases for layer %zu.\n", l);
            free_matrix(weights[l]);
            for (size_t k = 1; k < l; k++) {
                free_matrix(weights[k]);
                free_matrix(biases[k]);
            }
            return;
        }

        double init_range = sqrt(2.0 / layer_dims[l - 1]);
        for (size_t i = 0; i < weights[l]->rows * weights[l]->cols; i++) {
            weights[l]->data[i] = ((double)rand() / RAND_MAX) * 2 * init_range - init_range;
        }
    }
}

void forward_prop(Matrix* X, Matrix** weights, Matrix** biases, Matrix** activations, Matrix** Zs, size_t L) {
    activations[0] = X;

    for (size_t l = 1; l < L; l++) {
        Matrix* temp = matrix_mult(weights[l], activations[l - 1]);
        Zs[l] = matrix_add(temp, biases[l]);
        free_matrix(temp);

        activations[l] = create_matrix(Zs[l]->rows, Zs[l]->cols);

        if (l == L - 1) {
            // Softmax in the output layer
            softmax(Zs[l], activations[l]);
        } else {
            // ReLU for hidden layers
            #pragma omp parallel for
            for (size_t i = 0; i < Zs[l]->rows * Zs[l]->cols; i++) {
                activations[l]->data[i] = leaky_relu(Zs[l]->data[i]);
            }
        }
    }
}

double cost_function(Matrix* A, Matrix* Y) {
    size_t m = Y->cols;
    double cost = 0.0;
    double epsilon = 1e-10;

    for (size_t i = 0; i < Y->rows * Y->cols; i++) {
        double a = A->data[i];
        double y = Y->data[i];

        // Clamp to prevent log(0)
        if (a < epsilon) a = epsilon;
        if (a > 1.0 - epsilon) a = 1.0 - epsilon;

        cost += -y * log(a) - (1.0 - y) * log(1.0 - a);
    }

    return cost / m;
}

void backprop(Matrix* Y, Matrix** weights, Matrix** activations, Matrix** Zs, Matrix** dWs, Matrix** dbs, size_t L) {
    Matrix* dA = matrix_sub(activations[L - 1], Y);

    for (size_t l = L - 1; l > 0; l--) {
        Matrix* dZ = create_matrix(Zs[l]->rows, Zs[l]->cols);

        if (l == L - 1) {
            // For softmax, dZ = dA
            for (size_t i = 0; i < dZ->rows * dZ->cols; i++) {
                dZ->data[i] = dA->data[i];
            }
        } else {
            // For hidden layers, use derivative of ReLU
            #pragma omp parallel for
            for (size_t i = 0; i < dZ->rows * dZ->cols; i++) {
                dZ->data[i] = dA->data[i] * leaky_relu_derivative(Zs[l]->data[i]);
            }
        }

        Matrix* transposed = matrix_transpose(activations[l - 1]);
        dWs[l] = matrix_mult(dZ, transposed);
        free_matrix(transposed);

        dbs[l] = create_constant_matrix(dZ->rows, 1, 0.0);

        #pragma omp parallel for
        for (size_t i = 0; i < dZ->rows; i++) {
            for (size_t j = 0; j < dZ->cols; j++) {
                dbs[l]->data[i] += dZ->data[i * dZ->cols + j];
            }
        }

        if (l > 1) {
            Matrix* W_T = matrix_transpose(weights[l]);
            Matrix* dA_temp = matrix_mult(W_T, dZ);
            free_matrix(dA);
            dA = dA_temp;
            free_matrix(W_T);
        }

        free_matrix(dZ);
    }

    free_matrix(dA);
}

void update_parameters(Matrix** weights, Matrix** biases, Matrix** dWs, Matrix** dbs, size_t L, double learning_rate) {
    for (size_t l = 1; l < L; l++) {

        #pragma omp parallel for
        for (size_t i = 0; i < weights[l]->rows * weights[l]->cols; i++) {
            weights[l]->data[i] -= learning_rate * dWs[l]->data[i];
        }

        #pragma omp parallel for
        for (size_t i = 0; i < biases[l]->rows; i++) {
            biases[l]->data[i] -= learning_rate * dbs[l]->data[i];
        }

        free_matrix(dWs[l]);
        dWs[l] = NULL;

        free_matrix(dbs[l]);
        dbs[l] = NULL;
    }
}

void train(Matrix* X, Matrix* Y, size_t* layer_dims, size_t L, size_t epochs, double learning_rate) {
    Matrix* weights[L];
    Matrix* biases[L];
    Matrix* activations[L];
    Matrix* Zs[L];
    Matrix* dWs[L];
    Matrix* dbs[L];

    // Init
    for (size_t i = 0; i < L; i++) {
        weights[i] = NULL;
        biases[i] = NULL;
        activations[i] = NULL;
        Zs[i] = NULL;
        dWs[i] = NULL;
        dbs[i] = NULL;
    }

    init_params(weights, biases, layer_dims, L);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        forward_prop(X, weights, biases, activations, Zs, L);

        double cost = cost_function(activations[L - 1], Y);
        printf("Epoch %lu, Cost: %f\n", epoch, cost);

        // Debug: Print activation statistics
        double max_activation = 0.0, min_activation = 1.0;
        for (size_t i = 0; i < activations[L - 1]->rows * activations[L - 1]->cols; i++) {
            if (activations[L - 1]->data[i] > max_activation) max_activation = activations[L - 1]->data[i];
            if (activations[L - 1]->data[i] < min_activation) min_activation = activations[L - 1]->data[i];
        }
        printf("Debug: Output Activations - Min: %f, Max: %f\n", min_activation, max_activation);

        backprop(Y, weights, activations, Zs, dWs, dbs, L);
        update_parameters(weights, biases, dWs, dbs, L, learning_rate);

        for (size_t l = 1; l < L; l++) {
            free_matrix(Zs[l]);
            Zs[l] = NULL;

            free_matrix(activations[l]);
            activations[l] = NULL;
        }
    }

    for (size_t l = 1; l < L; l++) {
        free_matrix(weights[l]);
        free_matrix(biases[l]);
    }
}
