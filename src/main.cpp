#include "yolop.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    std::string param_path = "/home/hit/Project/YOLOP/weights/yolop-640-640-opt.param";
    std::string bin_path = "/home/hit/Project/YOLOP/weights/yolop-640-640-opt.bin";
    
    // Check command line arguments to ensure an image or video path is provided
    if (argc < 2) {
        std::cerr << "Error: No image or video path provided. Please provide the image or video path as a command line argument." << std::endl;
        return -1;
    }

    std::string input_path = argv[1];  // First argument as the input (image/video path)
    
    std::string save_det_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_det_ncnn.jpg";
    std::string save_da_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_da_ncnn.jpg";
    std::string save_ll_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_ll_ncnn.jpg";
    std::string save_merge_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_merge_ncnn.jpg";

    std::unique_ptr<yoloP> YOLOP(new yoloP(param_path, bin_path, 8));

    SegmentContent da_seg_content;
    SegmentContent ll_seg_content;
    std::vector<Boxf> detected_boxes;

    // Check if the input is a video or an image
    cv::Mat img_bgr;
    cv::VideoCapture cap;

    // Try opening as a video stream first
    if (input_path.find(".mp4") != std::string::npos || input_path.find(".avi") != std::string::npos) {
        cap.open(input_path); // Open video file
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video file " << input_path << std::endl;
            return -1;
        }

        // Process the video frame by frame
        while (true) {
            cap >> img_bgr;  // Get the next frame
            if (img_bgr.empty()) {
                break;  // End of video stream
            }

            YOLOP->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

            if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag) {
                // Process detection and save results for each frame
                cv::Mat img_det = img_bgr.clone();
                YOLOP->draw_boxes_inplace(img_det, detected_boxes);
                
                // Optionally save detection results (if needed)
                // cv::imwrite(save_det_path, img_det);

                // Merge segmentation results
                cv::Mat img_merge = img_bgr.clone();
                cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;
                cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
                YOLOP->draw_boxes_inplace(img_merge, detected_boxes);
                
                // Optionally save merged result (if needed)
                // cv::imwrite(save_merge_path, img_merge);

                // Display the merged frame
                cv::imshow("Processed Frame", img_merge);
                if (cv::waitKey(1) == 27) { // Press ESC to exit
                    break;
                }
            }
        }
        cap.release();
    }
    else {
        // Otherwise, process as an image
        img_bgr = cv::imread(input_path);
        if (img_bgr.empty()) {
            std::cerr << "Error: Unable to load image at " << input_path << std::endl;
            return -1;
        }

        // Perform detection and segmentation on the image
        YOLOP->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

        if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag) {
            // Save the detection results
            cv::Mat img_det = img_bgr.clone();
            YOLOP->draw_boxes_inplace(img_det, detected_boxes);
            cv::imwrite(save_det_path, img_det);
            std::cout << "Saved " << save_det_path << " done!" << "\n";
            
            // Save segmentation results
            cv::imwrite(save_da_path, da_seg_content.class_mat);
            cv::imwrite(save_ll_path, ll_seg_content.class_mat);
            std::cout << "Saved " << save_da_path << " done!" << "\n";
            std::cout << "Saved " << save_ll_path << " done!" << "\n";
            
            // Merge segmentation results
            cv::Mat img_merge = img_bgr.clone();
            cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;
            cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
            YOLOP->draw_boxes_inplace(img_merge, detected_boxes);
            cv::imwrite(save_merge_path, img_merge);
            std::cout << "Saved " << save_merge_path << " done!" << "\n";

            // Display the merged image
            cv::imshow("Merged Image", img_merge);
            cv::waitKey(0); // Wait for a key press
        } else {
            std::cerr << "Detection failed or no valid segments detected." << std::endl;
        }
    }

    return 0;
}
