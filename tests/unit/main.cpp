#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include <cstdio>
#include <cstdlib>

#if defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define IDET_TEST_ADDRESS_SANITIZER 1
    #endif
#endif

#if defined(__SANITIZE_ADDRESS__)
    #define IDET_TEST_ADDRESS_SANITIZER 1
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#if defined(__APPLE__) && defined(IDET_TEST_ADDRESS_SANITIZER)
    // Some Homebrew-linked runtime dependencies can crash during process-exit destructors on
    // macOS ASAN builds after all tests have passed. Exit directly in this narrow configuration
    // so sanitizer runs still report real test-time findings.
    std::fflush(nullptr);
    std::_Exit(result);
#else
    return result;
#endif
}
