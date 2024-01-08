#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

using namespace std;

extern "C" bool sobelFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sobelFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" bool sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sharpeningFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" bool medianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool medianFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" int blocksize_test_wrapper();
int BLOCK_SIZE = 16;

void printMenu(const string& menuTitle) {
    cout << "\n" << menuTitle << ":" << endl;
}

cv::Mat loadImage(const std::string& image_name) {
    std::string input_file = image_name + ".jpg";
    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);

    if (srcImage.empty()) {
        std::cout << "Image Not Found: " << input_file << std::endl;
    }

    std::cout << "\ninput image size (cols:rows:channels): " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    if (srcImage.channels() == 3) {
        cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);
    }

    return srcImage;
}
int performMedian() {
    string image_name = "sample";

    string input_file = image_name + ".jpg";
    string output_file_cpu = image_name + "_median_cpu.jpg";
    string output_file_gpu = image_name + "_median_gpu.jpg";

    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
    if (srcImage.empty())
    {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return 0;
    }
    cout << "\ninput image size (cols:rows:channels): " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    cv::Mat dstImage(srcImage.size(), srcImage.type());

    medianFilter_GPU_wrapper(srcImage, dstImage);
    imwrite(output_file_gpu, dstImage);

    medianFilter_CPU(srcImage, dstImage);
    imwrite(output_file_cpu, dstImage);
    return 1;
}
int performSobel() {
    string image_name = "sample";

    string input_file = image_name + ".jpg";
    string output_file_cpu = image_name + "_sobel_cpu.jpg";
    string output_file_gpu = image_name + "_sobel_gpu.jpg";

    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
    if (srcImage.empty()) {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return 0;
    }

    cout << "\ninput image size (cols:rows:channels): " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);

    cv::Mat dstImage(srcImage.size(), srcImage.type());

    sobelFilter_GPU_wrapper(srcImage, dstImage);

    dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
    dstImage *= 255;

    imwrite(output_file_gpu, dstImage);

    sobelFilter_CPU(srcImage, dstImage);

    imwrite(output_file_cpu, dstImage);
    return 1;
}

int performSharp() {
    string image_name = "sample";

    string input_file = image_name + ".jpg";
    string output_file_cpu = image_name + "_sharp_cpu.jpg";
    string output_file_gpu = image_name + "_sharp_gpu.jpg";

    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
    if (srcImage.empty())
    {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return 0;
    }
    cout << "\ninput image size (cols:rows:channels): " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    cv::Mat dstImage(srcImage.size(), srcImage.type());

    sharpeningFilter_GPU_wrapper(srcImage, dstImage);
    imwrite(output_file_gpu, dstImage);

    sharpeningFilter_CPU(srcImage, dstImage);
    imwrite(output_file_cpu, dstImage);
    return 1;
}

void printMenu() {
    cout << "\nSelect a filter to apply:" << endl;
    cout << "1. Median Filter" << endl;
    cout << "2. Sobel Filter" << endl;
    cout << "3. Sharpening Filter" << endl;
    cout << "4. Block size test" << endl;
    cout << "0. Exit" << endl;
}

void applyFilter(int choice, const cv::Mat& input, cv::Mat& output, bool useGPU) {
    switch (choice) {
    case 1:
        if (useGPU) {
            medianFilter_GPU_wrapper(input, output);
        }
        else {
            medianFilter_CPU(input, output);
        }
        break;
    case 2:
        if (useGPU) {
            sobelFilter_GPU_wrapper(input, output);
        }
        else {
            sobelFilter_CPU(input, output);
        }
        break;
    case 3:
        if (useGPU) {
            sharpeningFilter_GPU_wrapper(input, output);
        }
        else {
            sharpeningFilter_CPU(input, output);
        }
        break;
    default:
        cout << "Invalid choice." << endl;
    }
}

int main(int argc, char** argv) {
    int choice;
    int gpuChoice;
    do {
        printMenu();
        cout << "Enter your choice (0 to exit): ";
        cin >> choice;

        if (choice >= 1 && choice <= 3) {
            cout << "Do you want to apply the filter on GPU or CPU? (1: GPU, 2: CPU, 3: Both): ";
            cin >> gpuChoice;

            bool useGPU = (gpuChoice == 1 || gpuChoice == 3);
            bool useCPU = (gpuChoice == 2 || gpuChoice == 3);

            string image_name = "sample";
            string input_file = image_name + ".jpg";
            string output_file = image_name + "_output";

            cv::Mat srcImage = loadImage(image_name);
            cv::Mat dstImage(srcImage.size(), srcImage.type());

            applyFilter(choice, srcImage, dstImage, useGPU);

            if (useCPU) {
                cv::Mat cpuOutput(srcImage.size(), srcImage.type());
                applyFilter(choice, srcImage, cpuOutput, false);
                imwrite(output_file + "_cpu.jpg", cpuOutput);
                cout << "Filter applied successfully on CPU. Output saved to " << output_file + "_cpu.jpg" << endl;
            }

            if (useGPU) {
                imwrite(output_file + "_gpu.jpg", dstImage);
                cout << "Filter applied successfully on GPU. Output saved to " << output_file + "_gpu.jpg" << endl;
            }
        }
        else if (choice == 4) {
            BLOCK_SIZE = blocksize_test_wrapper();
            cout << "Block size value applied" << endl;
        }
    } while (choice != 0);

    return 0;
}



