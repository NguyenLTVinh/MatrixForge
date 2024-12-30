#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>
#include <math.h>

typedef struct {
    size_t size;
    double* data;
} Vector;

// Prototypes
Vector* create_vector(size_t size);
void free_vector(Vector* vec);
double dot_product(const Vector* v1, const Vector* v2);
Vector* cross_product(const Vector* v1, const Vector* v2);
int normalize(Vector* vec);
static inline double get_vector_element(const Vector* vec, size_t i);
static inline void set_vector_element(Vector* vec, size_t i, double value);

#endif // VECTOR_H
