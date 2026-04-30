#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include <idet.h>

TEST(DetectorWorker, EmptyWorkerIsIdleAndRejectsOperations) {
    idet::DetectorWorker worker;

    EXPECT_FALSE(static_cast<bool>(worker));
    EXPECT_EQ(worker.state(), idet::DetectorWorkerState::Idle);
    EXPECT_FALSE(worker.submit(idet::Image{}).ok());
    EXPECT_FALSE(worker.take_result().ok());
}

TEST(DetectorWorker, InvalidBindingContextsRejectedBeforeDetectorCreation) {
    idet::DetectorConfig cfg{};
    idet::DetectorWorkerOptions opt{};
    opt.binding_contexts = 0;

    auto r = idet::DetectorWorker::create(cfg, opt);
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.status().message.find("binding_contexts"), std::string::npos);
}

TEST(DetectorWorker, InvalidBindingContextIndexRejectedBeforeDetectorCreation) {
    idet::DetectorConfig cfg{};
    idet::DetectorWorkerOptions opt{};
    opt.binding_contexts = 2;
    opt.binding_context_index = 2;

    auto r = idet::DetectorWorker::create(cfg, opt);
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.status().message.find("binding_context_index"), std::string::npos);
}

TEST(DetectorWorker, BoundModeRequiresPositiveBindingShape) {
    idet::DetectorConfig cfg{};
    idet::DetectorWorkerOptions opt{};
    opt.use_bound = true;

    auto r = idet::DetectorWorker::create(cfg, opt);
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.status().message.find("binding_width"), std::string::npos);
}
