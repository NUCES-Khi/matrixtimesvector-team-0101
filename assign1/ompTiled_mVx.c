/*
 * Programmer(s) : Hasnain, Aliza, Ahmed Raza
 * Date: 19/03/2024
 * Description: Parallel version of matrix vector multiplication using the OPEN MP library with tiling. 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define BLOCK_DIM 16

void tiled_matvec_product(double *mat, double *vec, double *res, int dim) {
    #pragma omp parallel for schedule(dynamic)
    for (int block_row = 0; block_row < dim; block_row += BLOCK_DIM) {
        for (int block_col = 0; block_col < dim; block_col += BLOCK_DIM) {
            for (int row = block_row; row < block_row + BLOCK_DIM && row < dim; ++row) {
                double sum = 0.0;
                for (int col = block_col; col < block_col + BLOCK_DIM && col < dim; ++col) {
                    sum += mat[row * dim + col] * vec[col];
                }
                #pragma omp critical
                res[row] += sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <Matrix Size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    double *mat = (double *)malloc(n * n * sizeof(double));
    double *vec = (double *)malloc(n * sizeof(double));
    double *res = (double *)calloc(n, sizeof(double));

    srand((unsigned int)time(NULL));

    for (int i = 0; i < n * n; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        vec[i] = (double)rand() / RAND_MAX;
    }

    double start_time = omp_get_wtime();
    tiled_matvec_product(mat, vec, res, n);
    double end_time = omp_get_wtime();

    printf("Elapsed time: %f seconds\n", end_time - start_time);

    free(mat);
    free(vec);
    free(res);

    return 0;
}

