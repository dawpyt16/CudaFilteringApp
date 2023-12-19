#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#define BLOCK_SIZE      16 //MAX 1024
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

__global__ void sobelFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        float Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        float Ky[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

        float Gx = 0;
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++)
        {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++)
            {
                float fl = srcImage[(y + ky) * width + (x + kx)];
                Gx += fl * Kx[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
            }
        }
        float Gx_abs = fabs(Gx);

        float Gy = 0;
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++)
        {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++)
            {
                float fl = srcImage[(y + ky) * width + (x + kx)];
                Gy += fl * Ky[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
            }
        }
        float Gy_abs = fabs(Gy);

        dstImage[y * width + x] = static_cast<unsigned char>(Gx_abs + Gy_abs);
    }
}


extern "C" void sobelFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int inputSize = input.cols * input.rows;
    const int outputSize = output.cols * output.rows;
    unsigned char* d_input, * d_output;

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    cudaEventRecord(start);

    sobelFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows);

    cudaEventRecord(stop);

    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nSobel processing time on GPU (ms): " << milliseconds << "\n";
}

__global__ void sharpeningFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        for (int c = 0; c < channel; c++)
        {
            float sum = 0;
            for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
                for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                    float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
                    sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
                }
            }
            sum = fmin(1.0f, fmax(0.0f, sum / 255.0f));
            dstImage[(y * width + x) * channel + c] = static_cast<unsigned char>(sum * 255);

        }
    }
}


extern "C" void sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input.step / input.cols;

    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    unsigned char* d_input, * d_output;

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    cudaEventRecord(start);

    sharpeningFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

    cudaEventRecord(stop);

    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nSharpen processing time on GPU (ms): " << milliseconds << "\n";
}

__device__ void sort_bubble(unsigned char* filterVector)
{
    for (int i = 0; i < FILTER_WIDTH * FILTER_HEIGHT; i++) {
        for (int j = i + 1; j < FILTER_WIDTH * FILTER_HEIGHT; j++) {
            if (filterVector[i] > filterVector[j]) {
                unsigned char tmp = filterVector[i];
                filterVector[i] = filterVector[j];
                filterVector[j] = tmp;
            }
        }
    }
}

__device__ void sort_select(unsigned char* filterVector)
{
    for (int i = 0; i < FILTER_WIDTH * FILTER_HEIGHT - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < FILTER_WIDTH * FILTER_HEIGHT; j++) {
            if (filterVector[j] < filterVector[minIndex]) {
                minIndex = j;
            }
        }
        unsigned char tmp = filterVector[i];
        filterVector[i] = filterVector[minIndex];
        filterVector[minIndex] = tmp;
    }
}


__global__ void medianFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        for (int c = 0; c < channel; c++)
        {
            unsigned char filterVector[FILTER_WIDTH * FILTER_HEIGHT];
            for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
                for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                    filterVector[ky * FILTER_WIDTH + kx] = srcImage[((y + ky) * width + (x + kx)) * channel + c];
                }
            }
            sort_select(filterVector);
            dstImage[(y * width + x) * channel + c] = filterVector[(FILTER_WIDTH * FILTER_HEIGHT) / 2];
        }
    }
}


extern "C" void medianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input.step / input.cols;

    const int inputSize = input.cols * input.rows * channel;
    const int outputSize = output.cols * output.rows * channel;
    unsigned char* d_input, * d_output;

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    cudaEventRecord(start);

    medianFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

    cudaEventRecord(stop);

    cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}