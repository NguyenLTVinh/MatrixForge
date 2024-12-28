#include "matrix.h"
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define MAX_THREADS 4

/**
 * @brief Create a matrix with specified dimensions.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Pointer to the newly created Matrix structure.
 */
Matrix* create_matrix(size_t rows, size_t cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc(rows * cols, sizeof(double));
    return mat;
}

/**
 * @brief Free the memory allocated for a matrix.
 *
 * @param mat Pointer to the Matrix structure to free.
 */
void free_matrix(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

/**
 * @brief Add two matrices.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix after addition, or NULL if dimensions do not match.
 */
Matrix* matrix_add(const Matrix* A, const Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Matrix dimensions do not match for addition.\n");
        return NULL;
    }

    Matrix* result = create_matrix(A->rows, A->cols);
    size_t total_elements = A->rows * A->cols;

    #pragma omp parallel for
    for (size_t i = 0; i < total_elements; i++) {
        result->data[i] = A->data[i] + B->data[i];
    }

    return result;
}

/**
 * @brief Subtract one matrix from another.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix after subtraction, or NULL if dimensions do not match.
 */
Matrix* matrix_sub(const Matrix* A, const Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Matrix dimensions do not match for subtraction.\n");
        return NULL;
    }

    Matrix* result = create_matrix(A->rows, A->cols);
    size_t total_elements = A->rows * A->cols;

    #pragma omp parallel for
    for (size_t i = 0; i < total_elements; i++) {
        result->data[i] = A->data[i] - B->data[i];
    }

    return result;
}

/**
 * @brief Multiply two matrices.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @return Pointer to the resulting matrix after multiplication, or NULL if dimensions do not match.
 */
Matrix* matrix_mult(const Matrix* A, const Matrix* B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return NULL;
    }

    Matrix* result = create_matrix(A->rows, B->cols);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                sum += get_element(A, i, k) * get_element(B, k, j);
            }
            set_element(result, i, j, sum);
        }
    }

    return result;
}

/**
 * @brief Transpose a matrix.
 *
 * @param A Pointer to the matrix to transpose.
 * @return Pointer to the resulting transposed matrix.
 */
Matrix* matrix_transpose(const Matrix* A) {
    Matrix* result = create_matrix(A->cols, A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            set_element(result, j, i, get_element(A, i, j));
        }
    }

    return result;
}

/**
 * @brief Compute the inverse of a square matrix using Gaussian elimination.
 *
 * @param A Pointer to the square matrix to invert.
 * @return Pointer to the resulting inverted matrix, or NULL if the matrix is singular or not square.
 */
Matrix* matrix_inverse(const Matrix* A) {
    if (A->rows != A->cols) {
        fprintf(stderr, "Matrix must be square for inversion.\n");
        return NULL;
    }

    size_t n = A->rows;
    Matrix* augmented = create_matrix(n, 2 * n);
    Matrix* result = create_matrix(n, n);

    // Augmented matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            set_element(augmented, i, j, get_element(A, i, j));
            set_element(augmented, i, j + n, (i == j) ? 1.0 : 0.0);
        }
    }

    // Gaussian elimination
    for (size_t i = 0; i < n; i++) {
        double diag = get_element(augmented, i, i);
        if (fabs(diag) < 1e-10) {
            fprintf(stderr, "Matrix is singular and cannot be inverted.\n");
            free_matrix(augmented);
            return NULL;
        }

        // Normalize row
        for (size_t j = 0; j < 2 * n; j++) {
            set_element(augmented, i, j, get_element(augmented, i, j) / diag);
        }

        // Eliminate column
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = get_element(augmented, k, i);
                for (size_t j = 0; j < 2 * n; j++) {
                    set_element(augmented, k, j, get_element(augmented, k, j) - factor * get_element(augmented, i, j));
                }
            }
        }
    }

    // Extract inverse
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            set_element(result, i, j, get_element(augmented, i, j + n));
        }
    }

    free_matrix(augmented);
    return result;
}
