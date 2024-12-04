#include "yolop.h"

int main(int argc, char* argv[]) {
    std::string param_path = "/home/hit/Project/YOLOP/weights/yolop-640-640-opt.param";
    std::string bin_path = "/home/hit/Project/YOLOP/weights/yolop-640-640-opt.bin";
    std::string test_img_path = "/home/hit/Project/YOLOP/test_imgs/adb4871d-4d063244.jpg";
    std::string save_det_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_det_ncnn.jpg";
    std::string save_da_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_da_ncnn.jpg";
    std::string save_ll_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_ll_ncnn.jpg";
    std::string save_merge_path = "/home/hit/Project/YOLOP/results/test_lite_yolop_merge_ncnn.jpg";

    std::unique_ptr<yoloP> YOLOP(new yoloP(param_path, bin_path, 8));

    SegmentContent da_seg_content;
    SegmentContent ll_seg_content;
    std::vector<Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    YOLOP->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

    if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
    {
        // boxes.
        cv::Mat img_det = img_bgr.clone();
        YOLOP->draw_boxes_inplace(img_det, detected_boxes);
        cv::imwrite(save_det_path, img_det);
        std::cout << "Saved " << save_det_path << " done!" << "\n";
        // da && ll seg
        cv::imwrite(save_da_path, da_seg_content.class_mat);
        cv::imwrite(save_ll_path, ll_seg_content.class_mat);
        std::cout << "Saved " << save_da_path << " done!" << "\n";
        std::cout << "Saved " << save_ll_path << " done!" << "\n";
        // merge
        cv::Mat img_merge = img_bgr.clone();
        cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

        cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
        YOLOP->draw_boxes_inplace(img_merge, detected_boxes);
        cv::imwrite(save_merge_path, img_merge);
        std::cout << "Saved " << save_merge_path << " done!" << "\n";

        // label
        if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
        {

            for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
            {
                std::cout << "NCNN Version Detected Label: "
                    << it->first << " Name: " << it->second << std::endl;
            }

            for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
            {
                std::cout << "NCNN Version Detected Label: "
                    << it->first << " Name: " << it->second << std::endl;
            }
        }
    }
}
