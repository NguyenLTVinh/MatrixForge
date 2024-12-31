#include "matrix.h"
#include "vector.h"
#include "neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Util functions to load MNIST dataset
uint32_t flip_endian(uint32_t num) {
    return ((num & 0xFF) << 24) |
           ((num & 0xFF00) << 8) |
           ((num & 0xFF0000) >> 8) |
           ((num & 0xFF000000) >> 24);
}

Matrix* load_mnist_images(const char* filename, size_t* num_images, size_t* image_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t magic_number, num_imgs, rows, cols;

    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_imgs, sizeof(uint32_t), 1, file) != 1 ||
        fread(&rows, sizeof(uint32_t), 1, file) != 1 ||
        fread(&cols, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read MNIST image file header.\n");
        fclose(file);
        return NULL;
    }

    // Flip endian
    magic_number = flip_endian(magic_number);
    num_imgs = flip_endian(num_imgs);
    rows = flip_endian(rows);
    cols = flip_endian(cols);

    if (magic_number != 2051) {
        fprintf(stderr, "Error: Invalid magic number in MNIST image file.\n");
        fclose(file);
        return NULL;
    }

    *num_images = num_imgs;
    *image_size = rows * cols;

    Matrix* images = create_matrix(*image_size, *num_images);

    for (size_t i = 0; i < *num_images; i++) {
        for (size_t j = 0; j < *image_size; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel data from MNIST image file.\n");
                free_matrix(images);
                fclose(file);
                return NULL;
            }
            set_element(images, j, i, pixel / 255.0); // Normalize to [0, 1]
        }
    }

    fclose(file);
    return images;
}

Matrix* load_mnist_labels(const char* filename, size_t num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t magic_number, num_lbls;

    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_lbls, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read MNIST label file header.\n");
        fclose(file);
        return NULL;
    }

    // Flip endian
    magic_number = flip_endian(magic_number);
    num_lbls = flip_endian(num_lbls);

    if (magic_number != 2049 || num_lbls != num_labels) {
        fprintf(stderr, "Error: Invalid label file or mismatched label count.\n");
        fclose(file);
        return NULL;
    }

    Matrix* labels = create_matrix(10, num_labels); // One-hot encoding for 10 classes

    for (size_t i = 0; i < num_labels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label data from MNIST label file.\n");
            free_matrix(labels);
            fclose(file);
            return NULL;
        }

        // One-hot encode the label
        for (size_t j = 0; j < 10; j++) {
            set_element(labels, j, i, (j == label) ? 1.0 : 0.0);
        }
    }

    fclose(file);
    return labels;
}

double nnaccuracy(Matrix* predictions, Matrix* labels) {
    size_t correct = 0;
    size_t total = predictions->cols;

    for (size_t i = 0; i < total; i++) {
        double max_pred = 0.0;
        size_t pred_index = 0;

        double max_label = 0.0;
        size_t label_index = 0;

        for (size_t j = 0; j < predictions->rows; j++) {
            double pred_val = get_element(predictions, j, i);
            double label_val = get_element(labels, j, i);

            if (pred_val > max_pred) {
                max_pred = pred_val;
                pred_index = j;
            }
            if (label_val > max_label) {
                max_label = label_val;
                label_index = j;
            }
        }

        if (pred_index == label_index) {
            correct++;
        }
    }

    return (double)correct / total;
}

int main() {
    const char* train_images_file = "dataset/train-images-idx3-ubyte";
    const char* train_labels_file = "dataset/train-labels-idx1-ubyte";
    const char* test_images_file = "dataset/t10k-images-idx3-ubyte";
    const char* test_labels_file = "dataset/t10k-labels-idx1-ubyte";

    // Load datasets
    size_t num_train_images, train_image_size, num_test_images, test_image_size;
    Matrix* train_images = load_mnist_images(train_images_file, &num_train_images, &train_image_size);
    Matrix* train_labels = load_mnist_labels(train_labels_file, num_train_images);
    Matrix* test_images = load_mnist_images(test_images_file, &num_test_images, &test_image_size);
    Matrix* test_labels = load_mnist_labels(test_labels_file, num_test_images);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "Failed to load the MNIST dataset.\n");
        return -1;
    }

    fprintf(stdout, "MNIST dataset loaded.\n");
    // Neural network configuration
    size_t layer_dims[] = {train_image_size, 256, 256, 10}; // 784 input, 256 hidden, 256 hidden, 10 output
    size_t num_layers = sizeof(layer_dims) / sizeof(layer_dims[0]);
    size_t epochs = 50;
    double learning_rate = 0.001;

    // Initialize weights and biases
    Matrix* weights[num_layers];
    Matrix* biases[num_layers];
    for (size_t i = 0; i < num_layers; i++) {
        weights[i] = NULL;
        biases[i] = NULL;
    }
    init_params(weights, biases, layer_dims, num_layers);
    fprintf(stdout, "NN params configured.\n");

    // Train
    fprintf(stdout, "Training...\n");
    train(train_images, train_labels, layer_dims, num_layers, epochs, learning_rate);

    // Test
    Matrix* test_activations[num_layers];
    Matrix* test_Zs[num_layers];
    for (size_t i = 0; i < num_layers; i++) {
        test_activations[i] = NULL;
        test_Zs[i] = NULL;
    }
    forward_prop(test_images, weights, biases, test_activations, test_Zs, num_layers);

    double accuracy = nnaccuracy(test_activations[num_layers - 1], test_labels);
    printf("Test Accuracy: %.2f%%\n", accuracy * 100);

    // Clean ups
    free_matrix(train_images);
    free_matrix(train_labels);
    free_matrix(test_images);
    free_matrix(test_labels);

    for (size_t i = 1; i < num_layers; i++) {
        if (test_activations[i]) free_matrix(test_activations[i]);
        if (test_Zs[i]) free_matrix(test_Zs[i]);
        if (weights[i]) free_matrix(weights[i]);
        if (biases[i]) free_matrix(biases[i]);
    }

    return 0;
}
