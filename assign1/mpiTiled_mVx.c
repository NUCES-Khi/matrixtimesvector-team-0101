/*
 * Programmer(s) : Aliza
 * Dated: 19/03/2024
 * Description: Parallel version of matrix vector multiplication using the MPI library with tiling.  
 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define BLOCK_SIZE 32

void populate_matrices(double *mat, double *vec, int n) {
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) mat[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n; i++) vec[i] = (double)rand() / RAND_MAX;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <Matrix Size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    double *mat = malloc(n * n * sizeof(double));
    double *vec = malloc(n * sizeof(double));
    double *res = malloc(n * sizeof(double));

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) populate_matrices(mat, vec, n);

    MPI_Bcast(vec, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *tile = malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    double start = MPI_Wtime();
    for (int tr = 0; tr < tiles; tr++) {
        for (int tc = 0; tc < tiles; tc++) {
            int sr = tr * BLOCK_SIZE, sc = tc * BLOCK_SIZE;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    int gr = sr + i, gc = sc + j;
                    tile[i * BLOCK_SIZE + j] = (gr < n && gc < n) ? mat[gr * n + gc] : 0.0;
                }
            }
            double partial = 0.0;
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    partial += tile[i * BLOCK_SIZE + j] * vec[sc + j];
                }
            }
            double total;
            MPI_Allreduce(&partial, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if (rank == 0) {
                int gr = sr;
                res[gr] = total;
            }
        }
    }
    double end = MPI_Wtime();

    if (rank == 0) printf("Elapsed time: %f seconds\n", end - start);

    free(mat);
    free(vec);
    free(res);
    free(tile);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

