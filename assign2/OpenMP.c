#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    // Load the image
    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        printf("Error opening image\n");
        return -1;
    }

    Mat upscaledImage;
    
    double startTime = omp_get_wtime();

    // Image upscaling (Using OpenCV's resize function with OpenMP acceleration)
    resize(image, upscaledImage, Size(1280, 1024), 0, 0, INTER_LINEAR);

    double endTime = omp_get_wtime();
    printf("Processing time with OpenMP: %.4fs\n", endTime - startTime);

    // Save the upscaled image
    imwrite("upscaled_openmp.jpg", upscaledImage);

    return 0;
}
