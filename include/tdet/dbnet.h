#pragma once

#include "detector.h"
#include "tdet.h"

#include <memory>
#include <string>
#include <vector>

struct Detection;

namespace cv {
class Mat;
} // namespace cv

struct ImageSize {
    int width = 0;
    int height = 0;
};

class DBNet : public IDetector {
  public:
    DBNet(const std::string& model_path, tdet::TextDetectorConfig cfg, bool verbose = true);
    ~DBNet();

    // IDetector
    std::vector<Detection> detect(const cv::Mat& img_bgr, double* ms_out = nullptr) override;
    bool supports_binding() const override {
        return true;
    }
    bool prepare_binding(int w, int h, int contexts) override;
    int binding_thread_limit() const override {
        return pool_size_;
    }
    std::vector<Detection> detect_bound(const cv::Mat& img_bgr, int ctx_idx, double* ms_out = nullptr) override;

    // Legacy API (останутся для совместимости)
    std::vector<Detection> infer_unbound(const cv::Mat& img_bgr, double* ms_out = nullptr);
    std::vector<Detection> infer_bound(const cv::Mat& img_bgr, int ctx_idx, double* ms_out = nullptr);
    void ensure_pool_size(int n);

    std::vector<Detection> postprocess(const cv::Mat& prob_map, const ImageSize& orig_size) const;

  private:
    void preprocess_dynamic(const cv::Mat& img_bgr, cv::Mat& resized, cv::Mat& blob) const;

    void preprocess_fixed_into(float* dst_chw, const cv::Mat& img_bgr, const int W, const int H) const;

    void prepare_ctx_binding(int ctx_idx, const int W, const int H);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    tdet::TextDetectorConfig cfg_;
    int pool_size_ = 1;
};
