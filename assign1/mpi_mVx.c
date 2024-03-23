/*
 * Programmer(s) : Hasnain, Aliza, & Ahmed
 * Date: 19/03/2024
 * Desc: Parallel version of matrix vector multiplication using MPI library.
 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Populates a matrix and a vector with random double precision floating point numbers
void fill_with_random_doubles(double *matrix, double *vector, int dimension) {
    srand(time(NULL)); 

    for (int idx = 0; idx < dimension * dimension; ++idx) {
        matrix[idx] = (double)rand() / RAND_MAX;
    }

    for (int idx = 0; idx < dimension; ++idx) {
        vector[idx] = (double)rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: program <Matrix Dimension>\n");
        exit(EXIT_FAILURE);
    }

    int dimension = atoi(argv[1]);
    double *matrix = malloc(dimension * dimension * sizeof(double));
    double *vector = malloc(dimension * sizeof(double));
    double *partialResult = malloc(dimension * sizeof(double));

    MPI_Init(&argc, &argv);
    int processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (processId == 0) {
        fill_with_random_doubles(matrix, vector, dimension);
    }

    MPI_Bcast(vector, dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *localMatrix = malloc(dimension * (dimension / numProcesses) * sizeof(double));
    MPI_Scatter(matrix, dimension * (dimension / numProcesses), MPI_DOUBLE, localMatrix, dimension * (dimension / numProcesses), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();
    for (int i = 0; i < dimension / numProcesses; ++i) {
        partialResult[i] = 0;
        for (int j = 0; j < dimension; ++j) {
            partialResult[i] += localMatrix[i * dimension + j] * vector[j];
        }
    }
    double endTime = MPI_Wtime();

    double *gatheredResult = NULL;
    if (processId == 0) {
        gatheredResult = malloc(dimension * sizeof(double));
    }
    MPI_Gather(partialResult, dimension / numProcesses, MPI_DOUBLE, gatheredResult, dimension / numProcesses, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (processId == 0) {
        printf("Computation time: %f seconds\n", endTime - startTime);
        free(gatheredResult);
    }

    free(matrix);
    free(vector);
    free(partialResult);
    free(localMatrix);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

