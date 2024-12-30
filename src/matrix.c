#include "matrix.h"
#include "vector.h"
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>

#define MAX_THREADS 4
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-10

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
        if (fabs(diag) < TOLERANCE) {
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

    for (size_t i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; j++) {
            sum += get_element(A, i, j) * get_vector_element(v, j);
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
        double max_pivot = fabs(get_element(mat, i, i));

        for (size_t k = i + 1; k < n; k++) {
            double current_pivot = fabs(get_element(mat, k, i));
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
                double temp = get_element(mat, i, k);
                set_element(mat, i, k, get_element(mat, pivot_row, k));
                set_element(mat, pivot_row, k, temp);
            }
        }

        // Normalize the pivot row
        double pivot = get_element(mat, i, i);
        for (size_t j = i; j < m; j++) {
            double normalized = get_element(mat, i, j) / pivot;
            set_element(mat, i, j, normalized);
        }

        // Eliminate the entries below the pivot
        for (size_t j = i + 1; j < n; j++) {
            double factor = get_element(mat, j, i);
            for (size_t k = i; k < m; k++) {
                double updated = get_element(mat, j, k) - factor * get_element(mat, i, k);
                set_element(mat, j, k, updated);
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
            if (fabs(get_element(mat, i, j)) > TOLERANCE) {
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
            double factor = get_element(mat, j, pivot_col);
            for (size_t k = 0; k < m; k++) {
                double updated = get_element(mat, j, k) - factor * get_element(mat, i, k);
                set_element(mat, j, k, updated);
            }
        }
    }

    return 0;
}
