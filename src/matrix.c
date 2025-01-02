#include "matrix.h"
#include "vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

#define MAX_THREADS 4
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-10
#define BLOCK_SIZE 64

/**
 * @brief Create a matrix with specified dimensions.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Pointer to the newly created Matrix structure.
 */
Matrix* create_matrix(size_t rows, size_t cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Failed to allocate memory for matrix structure.\n");
        return NULL;
    }
    int ret = posix_memalign((void**)&mat->data, 64, sizeof(double) * rows * cols);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for matrix data (error code: %d).\n", ret);
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
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
    if (!mat) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        mat->data[i * n + i] = 1.0;
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
    if (!mat) {
        return NULL;
    }
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
    if ((A->rows != B->rows && B->rows != 1) || (A->cols != B->cols && B->cols != 1)) {
        fprintf(stderr, "Matrix dimensions do not match for addition with broadcasting.\n");
        return NULL;
    }

    Matrix* result = create_matrix(A->rows, A->cols);
    if (!result) return NULL;

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            size_t b_row = (B->rows == 1) ? 0 : i;
            size_t b_col = (B->cols == 1) ? 0 : j;
            result->data[i * A->cols + j] = A->data[i * A->cols + j] + B->data[b_row * B->cols + b_col];
        }
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
    if ((A->rows != B->rows && B->rows != 1) || (A->cols != B->cols && B->cols != 1)) {
        fprintf(stderr, "Matrix dimensions do not match for subtraction with broadcasting.\n");
        return NULL;
    }

    Matrix* result = create_matrix(A->rows, A->cols);
    if (!result) return NULL;

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            size_t b_row = (B->rows == 1) ? 0 : i;
            size_t b_col = (B->cols == 1) ? 0 : j;
            result->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[b_row * B->cols + b_col];
        }
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
    if (!result) return NULL;

    size_t n = A->rows, m = A->cols, p = B->cols;

    #pragma omp parallel for
    for (size_t ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < p; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < m; kk += BLOCK_SIZE) {
                for (size_t i = ii; i < fmin(ii + BLOCK_SIZE, n); i++) {
                    for (size_t j = jj; j < fmin(jj + BLOCK_SIZE, p); j++) {
                        double sum = 0.0;
                        for (size_t k = kk; k < fmin(kk + BLOCK_SIZE, m); k++) {
                            sum += A->data[i * m + k] * B->data[k * p + j];
                        }
                        result->data[i * p + j] += sum;
                    }
                }
            }
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
    if (!result) return NULL;

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            result->data[j * A->rows + i] = A->data[i * A->cols + j];
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
    if (!augmented || !result) {
        free_matrix(augmented);
        free_matrix(result);
        return NULL;
    }

    // Augmented matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented->data[i * (2 * n) + j] = A->data[i * n + j];
            augmented->data[i * (2 * n) + j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gaussian elimination
    for (size_t i = 0; i < n; i++) {
        double diag = augmented->data[i * (2 * n) + i];
        if (fabs(diag) < TOLERANCE) {
            fprintf(stderr, "Matrix is singular and cannot be inverted.\n");
            free_matrix(augmented);
            free_matrix(result);
            return NULL;
        }

        // Normalize row
        for (size_t j = 0; j < 2 * n; j++) {
            augmented->data[i * (2 * n) + j] /= diag;
        }

        // Eliminate column
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented->data[k * (2 * n) + i];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented->data[k * (2 * n) + j] -= factor * augmented->data[i * (2 * n) + j];
                }
            }
        }
    }

    // Extract inverse
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result->data[i * n + j] = augmented->data[i * (2 * n) + j + n];
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
            if (j < i) {
                L->data[j * n + i] = 0;
            } else {
                double sum = 0;
                for (size_t k = 0; k < i; k++) {
                    sum += L->data[i * n + k] * U->data[k * n + j];
                }
                U->data[i * n + j] = A->data[i * n + j] - sum;
            }
        }

        for (size_t j = 0; j < n; j++) {
            if (j < i) {
                U->data[j * n + i] = 0;
            } else if (j == i) {
                L->data[j * n + i] = 1;
            } else {
                double sum = 0;
                for (size_t k = 0; k < i; k++) {
                    sum += L->data[j * n + k] * U->data[k * n + i];
                }
                L->data[j * n + i] = (A->data[j * n + i] - sum) / U->data[i * n + i];
            }
        }
    }
    return 0;
}

/**
 * @brief Perform QR decomposition on a matrix
 *
 * @param A Input matrix.
 * @param Q Output orthogonal matrix.
 * @param R Output upper triangular matrix.
 * @return 0 on success, -1 on failure.
 */
int qr_decomposition(const Matrix* A, Matrix* Q, Matrix* R) {
    if (A->rows != A->cols) {
        fprintf(stderr, "QR Decomposition requires a square matrix.\n");
        return -1;
    }

    size_t n = A->rows;

    if (Q->rows != n || Q->cols != n || R->rows != n || R->cols != n) {
        fprintf(stderr, "Q and R must match the dimensions of A.\n");
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        Vector* ai = get_column_vector(A, i);
        Vector* vi = create_vector(n);

        // Copy ai to vi
        for (size_t j = 0; j < n; j++) {
            set_vector_element(vi, j, get_vector_element(ai, j));
        }

        // Orthogonalize against previous q vectors
        for (size_t j = 0; j < i; j++) {
            Vector* qj = get_column_vector(Q, j);
            double dot = dot_product(ai, qj);

            for (size_t k = 0; k < n; k++) {
                double updated = get_vector_element(vi, k) - dot * get_vector_element(qj, k);
                set_vector_element(vi, k, updated);
            }
            free_vector(qj);
        }

        // Normalize vi to become qi
        if (normalize(vi) != 0) {
            fprintf(stderr, "Failed to normalize vector during QR decomposition.\n");
            free_vector(ai);
            free_vector(vi);
            return -1;
        }

        for (size_t j = 0; j < n; j++) {
            set_element(Q, j, i, get_vector_element(vi, j));
        }

        free_vector(ai);
        free_vector(vi);
    }

    // Compute R
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            Vector* qi = get_column_vector(Q, i);
            Vector* aj = get_column_vector(A, j);
            double r = dot_product(qi, aj);
            set_element(R, i, j, r);
            free_vector(qi);
            free_vector(aj);
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

    if (!L || !U) {
        free_matrix(L);
        free_matrix(U);
        return 0.0;
    }

    lu_decomposition(A, L, U);

    double det = 1.0;
    for (size_t i = 0; i < A->rows; i++) {
        det *= U->data[i * A->cols + i];
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
        tr += A->data[i * A->cols + i];
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
    if (!row_vector) return NULL;

    for (size_t j = 0; j < mat->cols; j++) {
        set_vector_element(row_vector, j, mat->data[row * mat->cols + j]);
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
    if (!col_vector) return NULL;

    for (size_t i = 0; i < mat->rows; i++) {
        set_vector_element(col_vector, i, mat->data[i * mat->cols + col]);
    }

    return col_vector;
}

/**
 * @brief Multiply a matrix by a vector.
 *
 * @param A Pointer to the matrix (m x n).
 * @param v Pointer to the vector (size n).
 * @return Pointer to the resulting vector (size m), or NULL on failure.
 */
Vector* matrix_vector_mult(const Matrix* A, const Vector* v) {
    if (A->cols != v->size) {
        fprintf(stderr, "Matrix and vector dimensions do not match for multiplication.\n");
        return NULL;
    }

    Vector* result = create_vector(A->rows);
    if (!result) return NULL;

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; j++) {
            sum += A->data[i * A->cols + j] * get_vector_element(v, j);
        }
        set_vector_element(result, i, sum);
    }

    return result;
}

/**
 * @brief Gaussian elimination on a matrix to reduce it to row echelon form.
 *
 * @param mat Pointer to the matrix to perform Gaussian elimination on.
 * @return 0 on success, -1 on failure (e.g., matrix is singular or non-invertible).
 */
int gaussian_elimination(Matrix* mat) {
    size_t n = mat->rows;
    size_t m = mat->cols;

    for (size_t i = 0; i < n; i++) {
        // Find the pivot element in the current column
        size_t pivot_row = i;
        double max_pivot = fabs(mat->data[i * m + i]);

        for (size_t k = i + 1; k < n; k++) {
            double current_pivot = fabs(mat->data[k * m + i]);
            if (current_pivot > max_pivot) {
                max_pivot = current_pivot;
                pivot_row = k;
            }
        }

        // If no valid pivot is found, the matrix is singular
        if (fabs(max_pivot) < TOLERANCE) {
            return -1;
        }

        // Swap rows
        if (pivot_row != i) {
            for (size_t k = 0; k < m; k++) {
                double temp = mat->data[i * m + k];
                mat->data[i * m + k] = mat->data[pivot_row * m + k];
                mat->data[pivot_row * m + k] = temp;
            }
        }

        // Normalize the pivot row
        double pivot = mat->data[i * m + i];
        for (size_t j = i; j < m; j++) {
            mat->data[i * m + j] /= pivot;
        }

        // Eliminate the entries below the pivot
        for (size_t j = i + 1; j < n; j++) {
            double factor = mat->data[j * m + i];
            for (size_t k = i; k < m; k++) {
                mat->data[j * m + k] -= factor * mat->data[i * m + k];
            }
        }
    }
    return 0;
}

/**
 * @brief Gauss-Jordan elimination to reduce a matrix to reduced row echelon form (RREF).
 *
 * @param mat Pointer to the matrix to perform Gauss-Jordan elimination on.
 * @return 0 on success, -1 on failure (e.g., matrix is singular or non-invertible).
 */
int gauss_jordan_elimination(Matrix* mat) {
    // Gaussian elimination to reach row echelon form
    if (gaussian_elimination(mat) != 0) {
        return -1;
    }

    size_t n = mat->rows;
    size_t m = mat->cols;

    // Backward elimination to RREF
    for (int i = n - 1; i >= 0; i--) {
        // Find the pivot in the current row
        int pivot_col = -1;
        for (size_t j = 0; j < m; j++) {
            if (fabs(mat->data[i * m + j]) > TOLERANCE) {
                pivot_col = j;
                break;
            }
        }

        // If no pivot found, skip the row
        if (pivot_col == -1) {
            continue;
        }

        // Eliminate entries above the pivot
        for (int j = i - 1; j >= 0; j--) {
            double factor = mat->data[j * m + pivot_col];
            for (size_t k = 0; k < m; k++) {
                mat->data[j * m + k] -= factor * mat->data[i * m + k];
            }
        }
    }

    return 0;
}
