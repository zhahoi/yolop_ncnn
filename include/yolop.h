#include "net.h"
#include "layer.h"
#include <benchmark.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <unordered_map>

typedef struct
{
    int grid0;
    int grid1;
    int stride;
    float width;
    float height;
} YOLOPAnchor;


typedef struct
{
    float r;
    int dw;
    int dh;
    int new_unpad_w;
    int new_unpad_h;
    bool flag;
} YOLOPScaleParams;


typedef struct
{
    cv::Mat class_mat;
    cv::Mat color_mat;
    std::unordered_map<int, std::string> names_map;
    bool flag;
}SegmentContent;


struct Boxf {
    float x1;
    float y1;
    float x2;
    float y2;
    unsigned int label;
    float score;
    const char* label_text;
    bool flag;

    // Calculate the area of the current box
    float area() const {
        float width = x2 - x1;
        float height = y2 - y1;
        // Prevent negative values and ensure the area is positive
        return (width > 0 && height > 0) ? width * height : 0.0f;
    }

    // Calculate the IoU (Intersection over Union) between the current box and another box
    float iou_of(const Boxf& other) const {
        // Calculate the coordinates of the intersection
        float inner_x1 = (x1 > other.x1) ? x1 : other.x1;
        float inner_y1 = (y1 > other.y1) ? y1 : other.y1;
        float inner_x2 = (x2 < other.x2) ? x2 : other.x2;
        float inner_y2 = (y2 < other.y2) ? y2 : other.y2;

        // Calculate the width and height of the intersection
        float inner_w = inner_x2 - inner_x1;
        float inner_h = inner_y2 - inner_y1;

        // If the width or height of the intersection is less than or equal to 0, there is no intersection
        if (inner_h <= 0.0f || inner_w <= 0.0f) {
            return 0.0f;  // No intersection, return 0
        }

        // Calculate the area of the intersection
        float inner_area = inner_w * inner_h;

        // Calculate and return the IoU value
        return inner_area / (area() + other.area() - inner_area);
    }

    // Return a cv::Rect representing the current bounding box
    cv::Rect rect() const {
        return cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
    }

    // Return the top-left corner point of the rectangle (x1, y1)
    cv::Point tl() const {
        return cv::Point(static_cast<int>(x1), static_cast<int>(y1));
    }
};


class yoloP {
public:
	explicit yoloP(const std::string& _param_path,
                    const std::string& _bin_path,
                    unsigned int _num_threads = 1,
                    int _input_height = 640,
                    int _input_width = 640);
    ~yoloP();

private:
    const unsigned int num_threads; // initialize at runtime.
    // target image size after resize
    const int input_height; // 640/320/1280
    const int input_width; // 640/320/1280

    const float mean_vals[3] = { 255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f }; // RGB
    const float norm_vals[3] = { 1.f / (255.f * 0.229f), 1.f / (255.f * 0.224f), 1.f / (255.f * 0.225f) };

    enum NMS
    {
        HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

    std::vector<unsigned int> strides = { 8, 16, 32 };
    std::unordered_map<unsigned int, std::vector<YOLOPAnchor>> center_anchors;
    bool center_anchors_is_update = false;

private:
    void print_debug_string();

    void transform(const cv::Mat& mat_rs, ncnn::Mat& in);

    void resize_unscale(const cv::Mat& mat,
        cv::Mat& mat_rs,
        int target_height,
        int target_width,
        YOLOPScaleParams& scale_params);

    // only generate once
    void generate_anchors(unsigned int target_height, unsigned int target_width);

    void generate_bboxes_single_stride(const YOLOPScaleParams& scale_params,
        ncnn::Mat& det_pred,
        unsigned int stride,
        float score_threshold,
        float img_height,
        float img_width,
        std::vector<Boxf>& bbox_collection);

    void generate_bboxes_da_ll(const YOLOPScaleParams& scale_params,
        ncnn::Extractor& extractor,
        std::vector<Boxf>& bbox_collection,
        SegmentContent& da_seg_content,
        SegmentContent& ll_seg_content,
        float score_threshold, float img_height,
        float img_width); // det,da_seg,ll_seg

    void nms(std::vector<Boxf>& input, std::vector<Boxf>& output,
        float iou_threshold, unsigned int topk, unsigned int nms_type);
    void blending_nms(std::vector<Boxf>& input, std::vector<Boxf>& output, float iou_threshold, unsigned int topk);
    void offset_nms(std::vector<Boxf>& input, std::vector<Boxf>& output, float iou_threshold, unsigned int topk);
    void hard_nms(std::vector<Boxf>& input, std::vector<Boxf>& output, float iou_threshold, unsigned int topk);


public:
    void detect(const cv::Mat& mat,
        std::vector<Boxf>& detected_boxes,
        SegmentContent& da_seg_content,
        SegmentContent& ll_seg_content,
        float score_threshold = 0.50f, float iou_threshold = 0.30f,
        unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

    void draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<Boxf>& boxes);

private:
    ncnn::Net* net = nullptr;
    const char* log_id = nullptr;
    const char* param_path = nullptr;
    const char* bin_path = nullptr;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
};
