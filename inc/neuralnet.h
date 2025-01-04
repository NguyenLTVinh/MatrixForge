#ifndef NEURALNET_H
#define NEURALNET_H

#include "matrix.h"
#include "vector.h"

// Prototypes
void init_params(Matrix** weights, Matrix** biases, size_t* layer_dims, size_t L);
void forward_prop(Matrix* X, Matrix** weights, Matrix** biases, Matrix** activations, Matrix** Zs, size_t L);
double cost_function(Matrix* A, Matrix* Y);
void backprop(Matrix* Y, Matrix** weights, Matrix** activations, Matrix** Zs, Matrix** dWs, Matrix** dbs, size_t L);
void update_parameters(Matrix** weights, Matrix** biases, Matrix** dWs, Matrix** dbs, size_t L, double learning_rate, double clip_value);
void train(Matrix* X, Matrix* Y, size_t* layer_dims, size_t L, size_t epochs, double initial_learning_rate, double decay_factor, size_t patience);

#endif // NEURALNET_H
