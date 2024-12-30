#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void print_matrix(const Matrix* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%.2f ", get_element(mat, i, j));
        }
        printf("\n");
    }
}

void print_vector(const Vector* vec) {
    for (size_t i = 0; i < vec->size; i++) {
        printf("%.2f ", vec->data[i]);
    }
    printf("\n");
}

int main() {
    printf("Testing matrix library...\n");

    Matrix* A = create_matrix(2, 2);
    set_element(A, 0, 0, 1.0);
    set_element(A, 0, 1, 2.0);
    set_element(A, 1, 0, 3.0);
    set_element(A, 1, 1, 4.0);

    printf("Matrix A:\n");
    print_matrix(A);

    Matrix* B = create_matrix(2, 2);
    set_element(B, 0, 0, 5.0);
    set_element(B, 0, 1, 6.0);
    set_element(B, 1, 0, 7.0);
    set_element(B, 1, 1, 8.0);

    printf("Matrix B:\n");
    print_matrix(B);

    Matrix* I = create_identity_matrix(2);
    printf("Matrix I:\n");
    print_matrix(I);

    double a_trace = trace(A);
    double a_det = determinant(A);
    printf("A trace: %.2f\n", a_trace);
    printf("A determinant: %.2f\n", a_det);
 
    Matrix* C = matrix_add(A, B);
    if (C) {
        printf("A + B:\n");
        print_matrix(C);
        free_matrix(C);
    }

    Matrix* D = matrix_sub(A, B);
    if (D) {
        printf("A - B:\n");
        print_matrix(D);
        free_matrix(D);
    }

    Matrix* E = matrix_mult(A, B);
    if (E) {
        printf("A * B:\n");
        print_matrix(E);
        free_matrix(E);
    }

    Matrix* F = matrix_transpose(A);
    if (F) {
        printf("Transpose of A:\n");
        print_matrix(F);
        free_matrix(F);
    }

    Matrix* G = matrix_inverse(A);
    if (G) {
        printf("Inverse of A:\n");
        print_matrix(G);
        free_matrix(G);
    }

    Matrix* H = matrix_mult(A, B);
    if (H) {
        printf("A x B:\n");
        print_matrix(H);
        free_matrix(H);
    }

    // Test: get_row_vector
    Vector* row_vec = get_row_vector(A, 1);
    if (row_vec) {
        printf("Row 1 of A:\n");
        print_vector(row_vec);
        free_vector(row_vec);
    }

    // Test: get_column_vector
    Vector* col_vec = get_column_vector(A, 0);
    if (col_vec) {
        printf("Column 0 of A:\n");
        print_vector(col_vec);
        free_vector(col_vec);
    }

    // Test: QR decomposition
    Matrix* Q = create_matrix(A->rows, A->cols);
    Matrix* R = create_matrix(A->cols, A->cols);
    if (qr_decomposition(A, Q, R) == 0) {
        printf("Matrix Q (QR decomposition):\n");
        print_matrix(Q);
        printf("Matrix R (QR decomposition):\n");
        print_matrix(R);
    } else {
        printf("QR decomposition failed.\n");
    }
    free_matrix(Q);
    free_matrix(R);

    free_matrix(A);
    free_matrix(B);

    printf("All tests complete.\n");
    return 0;
}
