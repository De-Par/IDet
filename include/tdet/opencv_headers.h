#pragma once

// opencv.hpp
#if defined(__has_include) && __has_include(<opencv4/opencv2/opencv.hpp>)
    #include <opencv4/opencv2/opencv.hpp>
#elif defined(__has_include) && __has_include(<opencv2/opencv.hpp>)
    #include <opencv2/opencv.hpp>
#else
    #error "[ERROR] OpenCV 'opencv.hpp' header not found"
#endif

// core.hpp
#if defined(__has_include) && __has_include(<opencv4/opencv2/core.hpp>)
    #include <opencv4/opencv2/core.hpp>
#elif defined(__has_include) && __has_include(<opencv2/core.hpp>)
    #include <opencv2/core.hpp>
#else
    #error "[ERROR] OpenCV 'core.hpp' header not found"
#endif

// imgcodecs.hpp
#if defined(__has_include) && __has_include(<opencv4/opencv2/imgcodecs.hpp>)
    #include <opencv4/opencv2/imgcodecs.hpp>
#elif defined(__has_include) && __has_include(<opencv2/imgcodecs.hpp>)
    #include <opencv2/imgcodecs.hpp>
#else
    #error "[ERROR] OpenCV 'imgcodecs.hpp' header not found"
#endif

// imgproc.hpp
#if defined(__has_include) && __has_include(<opencv4/opencv2/imgproc.hpp>)
    #include <opencv4/opencv2/imgproc.hpp>
#elif defined(__has_include) && __has_include(<opencv2/imgproc.hpp>)
    #include <opencv2/imgproc.hpp>
#else
    #error "[ERROR] OpenCV 'imgproc.hpp' header not found"
#endif