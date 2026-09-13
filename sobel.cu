#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

__global__ void sobelKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        output[y * width + x] = 0;
        return;
    }

    int gx =
        -input[(y - 1) * width + (x - 1)]
        + input[(y - 1) * width + (x + 1)]
        - 2 * input[y * width + (x - 1)]
        + 2 * input[y * width + (x + 1)]
        - input[(y + 1) * width + (x - 1)]
        + input[(y + 1) * width + (x + 1)];

    int gy =
        -input[(y - 1) * width + (x - 1)]
        - 2 * input[(y - 1) * width + x]
        - input[(y - 1) * width + (x + 1)]
        + input[(y + 1) * width + (x - 1)]
        + 2 * input[(y + 1) * width + x]
        + input[(y + 1) * width + (x + 1)];

    int magnitude = abs(gx) + abs(gy);

    if (magnitude > 255)
        magnitude = 255;

    output[y * width + x] = (unsigned char)magnitude;
}

int main()
{
    string inputDir = "input/";
    string outputDir = "output/";

    int processed = 0;

    for (int i = 1; i <= 120; i++)
    {
        char filename[100];
        sprintf(filename, "image_%03d.jpg", i);

        string inputPath = inputDir + filename;
        string outputPath = outputDir + "sobel_" + filename;

        Mat image = imread(inputPath, IMREAD_GRAYSCALE);

        if (image.empty())
        {
            cout << "Could not read: " << inputPath << endl;
            continue;
        }

        int width = image.cols;
        int height = image.rows;

        Mat output(height, width, CV_8UC1);

        unsigned char* d_input;
        unsigned char* d_output;

        size_t size = width * height * sizeof(unsigned char);

        cudaMalloc((void**)&d_input, size);
        cudaMalloc((void**)&d_output, size);

        cudaMemcpy(
            d_input,
            image.data,
            size,
            cudaMemcpyHostToDevice
        );

        dim3 block(16, 16);

        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y
        );

        sobelKernel<<<grid, block>>>(
            d_input,
            d_output,
            width,
            height
        );

        cudaDeviceSynchronize();

        cudaMemcpy(
            output.data,
            d_output,
            size,
            cudaMemcpyDeviceToHost
        );

        imwrite(outputPath, output);

        cudaFree(d_input);
        cudaFree(d_output);

        processed++;

        cout << "Processed " << filename << endl;
    }

    cout << endl;
    cout << "Total images processed: " << processed << endl;

    return 0;
}
