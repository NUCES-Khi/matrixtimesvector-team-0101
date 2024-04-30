#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Mat image;
    if (rank == 0) {
        if (argc != 2) {
            printf("Usage: %s <image_path>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        image = imread(argv[1], IMREAD_COLOR);
        if (image.empty()) {
            printf("Error opening image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the image size
    int dimensions[2] = {image.rows, image.cols};
    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        // Allocate memory for image on other processes
        image.create(dimensions[0], dimensions[1], CV_8UC3);
    }

    // Broadcast the image data
    MPI_Bcast(image.data, image.total() * image.elemSize(), MPI_BYTE, 0, MPI_COMM_WORLD);

    Mat upscaledImage;
    double startTime = MPI_Wtime();

    // Resize the image on all processes (demonstrative purposes)
    resize(image, upscaledImage, Size(1280, 1024), 0, 0, INTER_LINEAR);

    double endTime = MPI_Wtime();

    if (rank == 0) {
        printf("Processing time with MPI: %.4fs\n", endTime - startTime);
        imwrite("upscaled_mpi.jpg", upscaledImage);
    }

    MPI_Finalize();
    return 0;
}
