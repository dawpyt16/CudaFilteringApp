#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <windows.h>
#include <opencv2/core/cuda.hpp>

int MAX_KERNEL_LENGTH = 9;

using namespace std;
using namespace cv;

LARGE_INTEGER t1, t2, frequency;

extern "C" void sobelFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    int rows = input.rows;
    int cols = input.cols;

    output = Mat::zeros(rows, cols, CV_32F);

    float kernelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    float kernelY[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

    for (int i = 1; i < rows - 1; ++i)
    {
        for (int j = 1; j < cols - 1; ++j)
        {
            float sumX = 0, sumY = 0;

            for (int m = -1; m <= 1; ++m)
            {
                for (int n = -1; n <= 1; ++n)
                {
                    sumX += input.at<uchar>(i + m, j + n) * kernelX[m + 1][n + 1];
                    sumY += input.at<uchar>(i + m, j + n) * kernelY[m + 1][n + 1];
                }
            }

            output.at<float>(i, j) = sqrt(sumX * sumX + sumY * sumY);
        }
    }

    output.convertTo(output, CV_32F, 1.0 / 255, 0);
    output *= 255;

    QueryPerformanceCounter(&t2);

    double secs = static_cast<double>(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    cout << "\nSobel processing time on CPU (ms): " << secs * 1000 << "\n";
}

extern "C" void sharpeningFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int kernel_size = 3;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    Mat kernel = (Mat_<double>(kernel_size, kernel_size) << -1, -1, -1, -1, 9, -1, -1, -1, -1);

    kernel /= sum(kernel)[0];

    filter2D(input, output, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

    QueryPerformanceCounter(&t2);

    double secs = static_cast<double>(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    cout << "\nSharpen processing time on CPU (ms): " << secs * 1000 << "\n";
}


extern "C" void medianFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    cv::Mat input_gray;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    medianBlur(input, output, kernel_size);

    QueryPerformanceCounter(&t2);

    double secs = static_cast<double>(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    cout << "\nMedian processing time on CPU (ms): " << secs * 1000 << "\n";
}
