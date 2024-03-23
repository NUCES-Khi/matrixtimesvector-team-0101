/*
 * Programmer: Hasnain
 * Dated: 19/03/2024
 * Description: Parallel version of matrix vector multiplication using OPEN MP library. 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double **create_matrix(int rows, int cols) {
    double **matrix = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
    }
    return matrix;
}

void destroy_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void fill_random(double **matrix, double *vector, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        vector[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void parallel_multiply(double **matrix, double *vector, double *res, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res[i] = 0.0;
        for (int j = 0; j < n; j++) {
            res[i] += matrix[i][j] * vector[j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <Matrix Size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    double **matrix = create_matrix(n, n);
    double *vector = malloc(n * sizeof(double));
    double *res = malloc(n * sizeof(double));

    fill_random(matrix, vector, n);

    double start_time = omp_get_wtime();
    parallel_multiply(matrix, vector, res, n);
    double end_time = omp_get_wtime();

    printf("Elapsed time: %f seconds\n", end_time - start_time);

    destroy_matrix(matrix, n);
    free(vector);
    free(res);

    return 0;
}

