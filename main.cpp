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
extern "C" bool blocksize_test_wrapper();

void printMenu() {
    std::cout << "Menu:" << std::endl;
    std::cout << "1. Test Block Size" << std::endl;
    std::cout << "2. Load Image" << std::endl;
    std::cout << "3. Run Sobel Filter" << std::endl;
    std::cout << "4. Run Sharpening Filter" << std::endl;
    std::cout << "5. Run Median Filter" << std::endl;
    std::cout << "6. Exit" << std::endl;
}

int main(int argc, char** argv) {
    string image_name = "sample";

    string input_file = image_name + ".jpg";
    string sobel_output_file_cpu = image_name + "_sobel_cpu.jpg";
    string sobel_output_file_gpu = image_name + "_sobel_gpu.jpg";
    string sharpen_output_file_cpu = image_name + "_sharpen_cpu.jpg";
    string sharpen_output_file_gpu = image_name + "_sharpen_gpu.jpg";
    string median_output_file_cpu = image_name + "_median_cpu.jpg";
    string median_output_file_gpu = image_name + "_median_gpu.jpg";

    bool imageLoaded = false;

    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
    if (srcImage.empty()) {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return -1;
    }

    //blocksize_test_wrapper();

    cout << "\ninput image size (cols:rows:channels): " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);

    cv::Mat dstImage(srcImage.size(), srcImage.type());

    cv::Mat sobelResult;
    cv::Mat sharpenResult;
    cv::Mat medianResult;
    int choice = 0;
    try {
        do {
            printMenu();
            std::cout << "Enter your choice: ";
            std::cin >> choice;

            switch (choice) {
            case 1:
                blocksize_test_wrapper();
                break;
            case 2:
                // Add code to load an image if needed
                break;
            case 3:
                sobelFilter_GPU_wrapper(srcImage, dstImage);
                cudaDeviceSynchronize();
                imwrite(sobel_output_file_gpu, dstImage);
                sobelFilter_CPU(srcImage, dstImage);
                imwrite(sobel_output_file_cpu, dstImage);

                break;
            case 4:
                // Sharpening Filter
                sharpeningFilter_CPU(srcImage, sharpenResult);
                cv::imwrite(sharpen_output_file_cpu, sharpenResult);
                sharpeningFilter_GPU_wrapper(srcImage, sharpenResult);
                cv::imwrite(sharpen_output_file_gpu, sharpenResult);
                break;
            case 5:
                // Median Filter
                medianFilter_CPU(srcImage, medianResult);
                cv::imwrite(median_output_file_cpu, medianResult);
                medianFilter_GPU_wrapper(srcImage, medianResult);
                cv::imwrite(median_output_file_gpu, medianResult);
                break;
            case 6:
                // Exit
                break;
            default:
                std::cout << "Invalid choice. Please enter a valid option." << std::endl;
            }
        } while (choice != 6);
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        // Handle the exception or log the error
    }
    catch (const std::exception& e) {
        std::cerr << "Standard C++ Exception: " << e.what() << std::endl;
        // Handle the exception or log the error
    }
    catch (...) {
        std::cerr << "Unknown Exception" << std::endl;
        // Handle the exception or log the error
    }
    

    //sharpeningFilter_GPU_wrapper(srcImage, dstImage);
    //cudaDeviceSynchronize();
    //imwrite(sharpen_output_file_gpu, dstImage);

    //sharpeningFilter_CPU(srcImage, dstImage);
    //imwrite(sharpen_output_file_cpu, dstImage);

    //medianFilter_GPU_wrapper(srcImage, dstImage);
    //cudaDeviceSynchronize();
    //imwrite(median_output_file_gpu, dstImage);
    //
    //medianFilter_CPU(srcImage, dstImage);
    //imwrite(median_output_file_cpu, dstImage);

    //sobelFilter_GPU_wrapper(srcImage, dstImage);
    //cudaDeviceSynchronize();
    //imwrite(sobel_output_file_gpu, dstImage);

    //sobelFilter_CPU(srcImage, dstImage);
    //imwrite(sobel_output_file_cpu, dstImage);


    return 0;
}
