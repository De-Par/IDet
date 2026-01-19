#pragma once

#include "detector.h"
#include "geometry.h"
#include "ort_headers.h"
#include "tdet.h"

#include <vector>

namespace cv {
class Mat;
}

class FaceDetector : public IDetector {
  public:
    explicit FaceDetector(tdet::FaceDetectorConfig cfg);

    // Runs SCRFD ONNX model; returns rectangles as Detection quads.
    std::vector<Detection> detect(const cv::Mat& img_bgr, double* ms_out = nullptr) override;
    bool supports_binding() const override {
        return true;
    }
    bool prepare_binding(int target_w, int target_h, int contexts) override;
    int binding_thread_limit() const override {
        return 1;
    }
    std::vector<Detection> detect_bound(const cv::Mat& img_bgr, int ctx_idx, double* ms_out = nullptr) override;
    // Prepare and reuse I/O binding for fixed WxH input (not thread-safe).
    bool prepare_binding(int target_w, int target_h);
    std::vector<Detection> detect_bound(const cv::Mat& img_bgr, double* ms_out = nullptr);

  private:
    void preprocess(const cv::Mat& img_bgr, cv::Mat& resized, cv::Mat& blob, float& sx, float& sy, int force_w = 0,
                    int force_h = 0) const;
    std::vector<Detection> postprocess(const float* boxes, const float* scores, size_t count, const cv::Size& orig,
                                       float sx, float sy) const;

    tdet::FaceDetectorConfig cfg_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "scrfd"};
    Ort::Session session_{nullptr};
    Ort::SessionOptions so_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::string in_name_;
    std::string boxes_name_;
    std::string scores_name_;

    std::vector<std::string> out_names_ = {"score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32"};
    std::vector<std::vector<int64_t>> out_shapes_;

    // Binding cache (single-thread use).
    int bound_w_ = 0, bound_h_ = 0;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::vector<float> input_buf_;
    std::vector<std::vector<float>> out_bufs_;
    std::vector<Ort::Value> bound_tensors_;
};
