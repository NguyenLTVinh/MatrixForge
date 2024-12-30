#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include "vector.h"

typedef struct {
    size_t rows;
    size_t cols;
    double* data;
} Matrix;

// prototypes
Matrix* create_matrix(size_t rows, size_t cols);
Matrix* create_identity_matrix(size_t n);
Matrix* create_constant_matrix(size_t rows, size_t cols, double value);
void free_matrix(Matrix* mat);

Matrix* matrix_add(const Matrix* A, const Matrix* B);
Matrix* matrix_sub(const Matrix* A, const Matrix* B);
Matrix* matrix_mult(const Matrix* A, const Matrix* B);
Matrix* matrix_transpose(const Matrix* A);
Matrix* matrix_inverse(const Matrix* A);
int gaussian_elimination(Matrix* mat);
int gauss_jordan_elimination(Matrix* mat);
int lu_decomposition(const Matrix* A, Matrix* L, Matrix* U);
int qr_decomposition(const Matrix* A, Matrix* Q, Matrix* R);
double determinant(const Matrix* A);
double trace(const Matrix* A);

Vector* get_row_vector(const Matrix* mat, size_t row);
Vector* get_column_vector(const Matrix* mat, size_t col);
Vector* matrix_vector_mult(const Matrix* A, const Vector* v);

static inline double get_element(const Matrix* mat, size_t i, size_t j) {
    return mat->data[i * mat->cols + j];
}

static inline void set_element(Matrix* mat, size_t i, size_t j, double value) {
    mat->data[i * mat->cols + j] = value;
}

#endif // MATRIX_H
