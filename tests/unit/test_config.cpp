#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include <idet.h>

namespace {

idet::DetectorConfig valid_text_config() {
    auto cfg = idet::DetectorConfig::setup(idet::Task::Text, "model.onnx");
    cfg.verbose = false;
    return cfg;
}

} // namespace

TEST(ResultContract, ErrWithOkStatusIsNotSuccessful) {
    auto r = idet::Result<int>::Err(idet::Status::Ok());

    EXPECT_FALSE(r.ok());
    EXPECT_FALSE(r.status().ok());
    EXPECT_NE(r.status().message.find("Result::Err"), std::string::npos);
}

TEST(DetectorConfig, ValidSetupPasses) {
    const auto cfg = valid_text_config();
    EXPECT_TRUE(cfg.validate().ok());
}

TEST(DetectorConfig, BindIoRequiresFixedInputDim) {
    auto cfg = valid_text_config();
    cfg.infer.bind_io = true;
    cfg.infer.fixed_input_dim = {0, 0};

    auto st = cfg.validate();
    EXPECT_FALSE(st.ok());
    EXPECT_NE(st.message.find("bind_io"), std::string::npos);
}

TEST(DetectorConfig, FixedInputDimMustBeUnsetOrPositivePair) {
    auto cfg = valid_text_config();
    cfg.infer.fixed_input_dim = {640, 0};

    auto st = cfg.validate();
    EXPECT_FALSE(st.ok());
    EXPECT_NE(st.message.find("fixed_input_dim"), std::string::npos);
}

TEST(DetectorConfig, RejectsInvalidRuntimeBudget) {
    auto cfg = valid_text_config();
    cfg.runtime.tile_omp_threads = 0;

    auto st = cfg.validate();
    EXPECT_FALSE(st.ok());
    EXPECT_NE(st.message.find("thread"), std::string::npos);
}

TEST(DetectorConfig, RejectsInvalidNmsThreshold) {
    auto cfg = valid_text_config();
    cfg.infer.nms_iou = 1.1f;

    auto st = cfg.validate();
    EXPECT_FALSE(st.ok());
    EXPECT_NE(st.message.find("nms_iou"), std::string::npos);
}
