#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
    size_t rows;
    size_t cols;
    double* data;
} Matrix;

// prototypes
Matrix* create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix* mat);

Matrix* matrix_add(const Matrix* A, const Matrix* B);
Matrix* matrix_sub(const Matrix* A, const Matrix* B);
Matrix* matrix_mult(const Matrix* A, const Matrix* B);
Matrix* matrix_transpose(const Matrix* A);
Matrix* matrix_inverse(const Matrix* A);

static inline double get_element(const Matrix* mat, size_t i, size_t j) {
    return mat->data[i * mat->cols + j];
}

static inline void set_element(Matrix* mat, size_t i, size_t j, double value) {
    mat->data[i * mat->cols + j] = value;
}

#endif // MATRIX_H
