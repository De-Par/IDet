# ACL Execution Provider

[Back to README](../README.md) | [Documentation index](index.md)

## Setup

1. Install dependencies:
    ```bash
    sudo apt update
    sudo apt install -y \
        build-essential git cmake ninja-build \
        python3 python3-dev python3-pip python3-venv \
        scons wget unzip \
        libssl-dev zlib1g-dev
    ```

2. Download ACL library:
    ```bash
    git clone https://github.com/ARM-software/ComputeLibrary.git
    cd ComputeLibrary
    git fetch --all --tags
    # If you want - change version to v.24.07 (stable)
    git checkout c5dd7753d0475ffec0f192f3181fe67a1d761680
    ```

3. Build ACL library with correct parameters for your system:
    ```bash
    # Example for Kunpeng-920a

    scons -j$(nproc) \
        Werror=1 \
        neon=1 \
        opencl=0 \
        os=linux \
        arch=arm64-v8a \
        examples=0 \
        build=native \
        extra_cxx_flags="-O3 -ffast-math -fPIC"
    ```

4. Download ONNX Runtime:
    ```bash
    git clone --recursive https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    ```

5. Build ONNX Runtime with ACL (replace acl libs/home with correct paths):
    ```bash
    ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel \
        --update --build \
        --skip_tests \
        --enable_lto \
        --use_acl \
        --acl_home=/path/to/ComputeLibrary \
        --acl_libs=/path/to/ComputeLibrary/build \
        --cmake_extra_defines \
            onnxruntime_BUILD_FOR_NATIVE_MACHINE=ON \
            CMAKE_C_FLAGS_RELEASE="-O3 -fPIC" \
            CMAKE_CXX_FLAGS_RELEASE="-O3 -fPIC"
    ```

6. After build finishes, copy headers+libs to /usr/local (adjust paths if needed):
    ```bash
    sudo cp -r include/onnxruntime /usr/local/include/
    sudo cp -d build/Linux/Release/libonnxruntime.so* /usr/local/lib/
    sudo cp -d build/Linux/Release/libonnxruntime_providers_shared.so /usr/local/lib/
    sudo ldconfig
    