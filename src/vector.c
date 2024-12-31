#include "vector.h"
#include <stdio.h>

/**
 * @brief Create a vector with the specified size.
 *
 * @param size Number of elements in the vector.
 * @return Pointer to the newly created Vector structure.
 */
Vector* create_vector(size_t size) {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    if (!vec) {
        fprintf(stderr, "Failed to allocate memory for vector structure.\n");
        return NULL;
    }

    vec->data = (double*)calloc(size, sizeof(double));
    if (!vec->data) {
        fprintf(stderr, "Failed to allocate memory for vector data.\n");
        free(vec);
        return NULL;
    }

    vec->size = size;
    return vec;
}

/**
 * @brief Free the memory allocated for a vector.
 *
 * @param vec Pointer to the Vector structure to free.
 */
void free_vector(Vector* vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

/**
 * @brief Compute the dot product of two vectors.
 *
 * @param v1 Pointer to the first vector.
 * @param v2 Pointer to the second vector.
 * @return The dot product, or 0.0 if sizes do not match.
 */
double dot_product(const Vector* v1, const Vector* v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Vectors must be of the same size for dot product.\n");
        return 0.0;
    }

    double result = 0.0;
    for (size_t i = 0; i < v1->size; i++) {
        result += v1->data[i] * v2->data[i];
    }
    return result;
}

/**
 * @brief Compute the cross product of two 3D vectors.
 *
 * @param v1 Pointer to the first vector (size 3).
 * @param v2 Pointer to the second vector (size 3).
 * @return Pointer to the resulting cross product vector, or NULL if sizes are not 3.
 */
Vector* cross_product(const Vector* v1, const Vector* v2) {
    if (v1->size != 3 || v2->size != 3) {
        fprintf(stderr, "Cross product is only defined for 3D vectors.\n");
        return NULL;
    }

    Vector* result = create_vector(3);
    set_vector_element(result, 0, v1->data[1] * v2->data[2] - v1->data[2] * v2->data[1]);
    set_vector_element(result, 1, v1->data[2] * v2->data[0] - v1->data[0] * v2->data[2]);
    set_vector_element(result, 2, v1->data[0] * v2->data[1] - v1->data[1] * v2->data[0]);

    return result;
}

/**
 * @brief Normalize a vector to have a magnitude of 1.
 *
 * @param vec Pointer to the vector to normalize.
 */
int normalize(Vector* vec) {
    double magnitude = 0.0;
    for (size_t i = 0; i < vec->size; i++) {
        magnitude += vec->data[i] * vec->data[i];
    }
    magnitude = sqrt(magnitude);

    if (magnitude > 1e-10) {
        for (size_t i = 0; i < vec->size; i++) {
            vec->data[i] /= magnitude;
        }
    } else {
        fprintf(stderr, "Cannot normalize a zero vector.\n");
        return -1;
    }
    return 0;
}
