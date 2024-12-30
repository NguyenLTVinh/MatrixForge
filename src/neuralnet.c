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
        biases[l] = create_constant_matrix(layer_dims[l], 1, 0.0);

        double init_range = sqrt(2.0 / layer_dims[l - 1]);
        for (size_t i = 0; i < weights[l]->rows * weights[l]->cols; i++) {
            weights[l]->data[i] = ((double)rand() / RAND_MAX) * 2 * init_range - init_range;
        }
    }
}

void forward_prop(Matrix* X, Matrix** weights, Matrix** biases, Matrix** activations, Matrix** Zs, size_t L) {
    activations[0] = X;

    for (size_t l = 1; l < L; l++) {
        Zs[l] = matrix_add(matrix_mult(weights[l], activations[l - 1]), biases[l]);
        activations[l] = create_matrix(Zs[l]->rows, Zs[l]->cols);

        #pragma omp parallel for
        for (size_t i = 0; i < Zs[l]->rows * Zs[l]->cols; i++) {
            activations[l]->data[i] = relu(Zs[l]->data[i]);
        }
    }
}

double cost_function(Matrix* A, Matrix* Y) {
    size_t m = Y->cols;
    double cost = 0.0;
    double epsilon = 1e-10;

    for (size_t i = 0; i < Y->rows * Y->cols; i++) {
        cost += -Y->data[i] * log(A->data[i] + epsilon) - (1.0 - Y->data[i]) * log(1.0 - A->data[i] + epsilon);
    }

    return cost / m;
}

void backprop(Matrix* Y, Matrix** weights, Matrix** activations, Matrix** Zs, Matrix** dWs, Matrix** dbs, size_t L) {
    Matrix* dA = matrix_sub(activations[L - 1], Y);

    for (size_t l = L - 1; l > 0; l--) {
        Matrix* dZ = create_matrix(Zs[l]->rows, Zs[l]->cols);

        #pragma omp parallel for
        for (size_t i = 0; i < dZ->rows * dZ->cols; i++) {
            dZ->data[i] = dA->data[i] * relu_derivative(Zs[l]->data[i]);
        }

        dWs[l] = matrix_mult(dZ, matrix_transpose(activations[l - 1]));
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
    }
}

void train(Matrix* X, Matrix* Y, size_t* layer_dims, size_t L, size_t epochs, double learning_rate) {
    Matrix* weights[L];
    Matrix* biases[L];
    Matrix* activations[L];
    Matrix* Zs[L];
    Matrix* dWs[L];
    Matrix* dbs[L];

    init_params(weights, biases, layer_dims, L);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        forward_prop(X, weights, biases, activations, Zs, L);

        double cost = cost_function(activations[L - 1], Y);
        printf("Epoch %lu, Cost: %f\n", epoch, cost);

        backprop(Y, weights, activations, Zs, dWs, dbs, L);

        update_parameters(weights, biases, dWs, dbs, L, learning_rate);
    }

    for (size_t l = 1; l < L; l++) {
        free_matrix(weights[l]);
        free_matrix(biases[l]);
        free_matrix(dWs[l]);
        free_matrix(dbs[l]);
    }
}
