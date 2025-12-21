#pragma once
#include "geometry.h"
#include "timer.h"

#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#else
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>
#endif

class DBNet
{
public:
    float bin_thresh = 0.3f;
    float box_thresh = 0.3f;
    int limit_side_len = 960; // px
    float unclip_ratio = 1.0f;
    int min_text_size = 3; // px
    bool apply_sigmoid = false;

    int fixed_W = 0;
    int fixed_H = 0;

    // ORT objects
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dbnet"};
    Ort::Session session{nullptr};
    Ort::SessionOptions so;

    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name;
    std::string out_name;

    struct BindingCtx
    {
        Ort::IoBinding io;
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

        std::vector<float> in_buf;
        std::vector<float> out_buf;
        std::vector<int64_t> in_shape;
        std::vector<int64_t> out_shape;

        int curW = 0;
        int curH = 0;
        int curOW = 0;
        int curOH = 0;
        bool bound = false;

        Ort::Value in_tensor{nullptr};
        Ort::Value out_tensor{nullptr};

        explicit BindingCtx(Ort::Session &s) : io(s) {}
    };

    std::vector<std::unique_ptr<BindingCtx>> pool;

    DBNet(const std::string &model_path, const int intra_threads = 0, const int inter_threads = 1, bool verbose = true);

    std::vector<Detection> infer_unbound(const cv::Mat &img_bgr, double *ms_out = nullptr);

    std::vector<Detection> infer_bound(const cv::Mat &img_bgr, int ctx_idx, double *ms_out = nullptr);

    void ensure_pool_size(int n);

    std::vector<Detection> postprocess(const cv::Mat &prob_map, const cv::Size &orig_size) const;

private:
    void preprocess_dynamic(const cv::Mat &img_bgr, cv::Mat &resized, cv::Mat &blob) const;

    void preprocess_fixed_into(float *dst_chw, const cv::Mat &img_bgr, const int W, const int H) const;

    void prepare_binding(BindingCtx &ctx, const int W, const int H);
};