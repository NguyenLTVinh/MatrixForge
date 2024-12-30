#include "matrix.h"
#include "vector.h"
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
 * @brief Create an identity matrix.
 *
 * @param n Size of the identity matrix (n x n).
 * @return Pointer to the newly created identity matrix.
 */
Matrix* create_identity_matrix(size_t n) {
    Matrix* mat = create_matrix(n, n);
    for (size_t i = 0; i < n; i++) {
        set_element(mat, i, i, 1.0);
    }
    return mat;
}

/**
 * @brief Create a matrix filled with a constant value.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param value The constant value to fill the matrix.
 * @return Pointer to the newly created matrix.
 */
Matrix* create_constant_matrix(size_t rows, size_t cols, double value) {
    Matrix* mat = create_matrix(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        mat->data[i] = value;
    }
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


/**
 * @brief Perform LU decomposition of a matrix.
 *
 * @param A Pointer to the matrix to decompose.
 * @param L Pointer to the lower triangular matrix (output).
 * @param U Pointer to the upper triangular matrix (output).
 * @return 0 if successful, -1 if dimensions do not match.
 */
int lu_decomposition(const Matrix* A, Matrix* L, Matrix* U) {
    if (A->rows != A->cols) {
        fprintf(stderr, "LU Decomposition requires a square matrix.\n");
        return -1;
    }

    size_t n = A->rows;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (j < i)
                set_element(L, j, i, 0);
            else {
                double sum = 0;
                for (size_t k = 0; k < i; k++)
                    sum += get_element(L, i, k) * get_element(U, k, j);
                set_element(U, i, j, get_element(A, i, j) - sum);
            }
        }
        for (size_t j = 0; j < n; j++) {
            if (j < i)
                set_element(U, j, i, 0);
            else if (j == i)
                set_element(L, j, i, 1);
            else {
                double sum = 0;
                for (size_t k = 0; k < i; k++)
                    sum += get_element(L, j, k) * get_element(U, k, i);
                set_element(L, j, i, (get_element(A, j, i) - sum) / get_element(U, i, i));
            }
        }
    }
    return 0;
}

/**
 * @brief Compute the determinant of a matrix.
 *
 * @param A Pointer to the square matrix.
 * @return Determinant value, or 0.0 if the matrix is not square.
 */
double determinant(const Matrix* A) {
    if (A->rows != A->cols) {
        fprintf(stderr, "Determinant requires a square matrix.\n");
        return 0.0;
    }

    Matrix* L = create_matrix(A->rows, A->cols);
    Matrix* U = create_matrix(A->rows, A->cols);

    lu_decomposition(A, L, U);

    double det = 1.0;
    for (size_t i = 0; i < A->rows; i++) {
        det *= get_element(U, i, i);
    }

    free_matrix(L);
    free_matrix(U);
    return det;
}

/**
 * @brief Compute the trace of a matrix.
 *
 * @param A Pointer to the matrix.
 * @return Trace value.
 */
double trace(const Matrix* A) {
    double tr = 0.0;
    for (size_t i = 0; i < A->rows && i < A->cols; i++) {
        tr += get_element(A, i, i);
    }
    return tr;
}

/**
 * @brief Return a row from the matrix as a vector.
 *
 * @param mat Pointer to the matrix.
 * @param row Index of the row to extract.
 * @return Pointer to the newly created vector containing the row, or NULL if the row index is out of bounds.
 */
Vector* get_row_vector(const Matrix* mat, size_t row) {
    if (row >= mat->rows) {
        fprintf(stderr, "Row index out of bounds.\n");
        return NULL;
    }

    Vector* row_vector = create_vector(mat->cols);
    for (size_t j = 0; j < mat->cols; j++) {
        set_vector_element(row_vector, j, get_element(mat, row, j));
    }

    return row_vector;
}

/**
 * @brief Return a column from the matrix as a vector.
 *
 * @param mat Pointer to the matrix.
 * @param col Index of the column to extract.
 * @return Pointer to the newly created vector containing the column, or NULL if the column index is out of bounds.
 */
Vector* get_column_vector(const Matrix* mat, size_t col) {
    if (col >= mat->cols) {
        fprintf(stderr, "Column index out of bounds.\n");
        return NULL;
    }

    Vector* col_vector = create_vector(mat->rows);
    for (size_t i = 0; i < mat->rows; i++) {
        set_vector_element(col_vector, i, get_element(mat, i, col));
    }

    return col_vector;
}
