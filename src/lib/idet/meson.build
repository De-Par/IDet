# Get sources from nested dirs
subdir('algo')
subdir('engine')
subdir('platform')

# Forming final sources list
idet_lib_source = [
    files('idet.cpp', 'image.cpp'),
    idet_lib_algo_source,
    idet_lib_engine_source,
    idet_lib_platform_source,
]

# Additional defines
idet_build_defs = []

# Embed onnx models into final binary
if embed_models
    models_root = join_paths(meson.project_source_root(), 'assets', 'models')

    # Path to script, which convert binary to `.cpp` file
    script_path = join_paths(meson.project_source_root(), 'tools', 'bin_to_cpp.py')
    if not fs.exists(script_path)
        error('Script for binary convertion not found at ' + script_path)
    endif

    # Python3 interpreter
    python = import('python').find_installation('python3', required: false)
    if not python.found()
        error('python3 not found in system for generating embedded model sources')
    endif

    # Text detection model
    text_model_path = join_paths(models_root, 'paddleocr', 'ch_ppocr_v2_det.onnx')
    if not fs.exists(text_model_path)
        warning('DBNet model (text) not found at ' + text_model_path)
    else
        text_blob = custom_target(
            'dbnet_model_gen',
            input  : text_model_path,
            output : 'dbnet_model_gen.cpp',
            command: [
                python, script_path,
                '--name', 'dbnet_model', 
                '--input', '@INPUT@',
                '--output', '@OUTPUT@',
            ],
        )
        idet_lib_source += [text_blob]
        idet_build_defs += ['-DIDET_HAVE_DBNET_EMBED']
    endif

    # Face detection model
    face_model_path = join_paths(models_root, 'scrfd', 'scrfd_500m_bnkps.onnx')
    if not fs.exists(face_model_path)
        warning('SCRFD model (face) not found at ' + face_model_path)
    else
        face_blob = custom_target(
            'scrfd_model_gen',
            input  : face_model_path,
            output : 'scrfd_model_gen.cpp',
            command: [
                python, script_path,
                '--name', 'scrfd_model', 
                '--input', '@INPUT@',
                '--output', '@OUTPUT@',
            ],
        )
        idet_lib_source += [face_blob]
        idet_build_defs += ['-DIDET_HAVE_SCRFD_EMBED']
    endif
endif

if idet_libtype in ['shared', 'both']

    idet_lib_shared = shared_library(
        'idet',
        idet_lib_source,
        include_directories : idet_lib_inc,           # public + private
        cpp_args            : cxx_args + ['-DIDET_BUILD_SHARED'] + idet_build_defs,
        link_args           : ld_args,
        dependencies        : idet_lib_deps,          # public + private
        link_depends        : ort_link_depends,
        build_rpath         : build_rpath,
        install_rpath       : install_rpath,
        install             : true,
    )

    idet_dep_shared = declare_dependency(
        link_with           : idet_lib_shared,
        include_directories : [idet_public_inc],      # public 
        dependencies        : [],                     # public 
        compile_args        : ['-DIDET_USE_SHARED'],
    )

endif

if idet_libtype in ['static', 'both']

    idet_lib_static = static_library(
        'idet',
        idet_lib_source,
        include_directories : idet_lib_inc,           # public + private
        cpp_args            : cxx_args + ['-DIDET_BUILD_STATIC'] + idet_build_defs,
        link_args           : ld_args,  
        dependencies        : idet_lib_deps,          # public + private
        link_depends        : ort_link_depends,
        install             : true,
    )

    idet_dep_static = declare_dependency(
        link_with           : idet_lib_static,
        include_directories : [idet_public_inc],      # public 
        dependencies        : [],                     # public 
    )

endif

idet_dep = disabler()

if idet_libtype == 'both'
    if idet_exe_link == 'static'
        idet_dep = idet_dep_static
    else
        idet_dep = idet_dep_shared
    endif
elif idet_libtype == 'static'
    idet_dep = idet_dep_static
else  # idet_libtype == 'shared'
    idet_dep = idet_dep_shared
endif

if not idet_dep.found()
    error('IDet public dependency not found')
endif

# -------------------------------------------------------------------------
# Internal dependency for white-box tests
#  - exports private include dirs (src/lib/idet)
#  - links statically when needed (so internal symbols are resolvable)
# -------------------------------------------------------------------------

idet_internal_dep = disabler()

if build_tests 
    # Test targets need external compile/include flags from deps, but linking is already
    # provided via `link_with` target and its dependency closure. Keep compile-only deps
    # to avoid duplicate link items like `-lonnxruntime`
    idet_internal_compile_deps = [
        opencv_dep.partial_dependency(
            compile_args: true,
            includes: true,
        ),
        ort_dep.partial_dependency(
            compile_args: true,
            includes: true,
        ),
    ]

    # If we already have a static idet library, use it for tests
    if idet_libtype in ['static', 'both']
        idet_internal_dep = declare_dependency(
            link_with           : idet_lib_static,
            include_directories : idet_lib_inc,   # public + private (tests need internal headers)
            dependencies        : idet_internal_compile_deps,
            sources             : ort_link_depends,
            compile_args        : ['-DIDET_TESTING_INTERNAL'],
        )
    else
        # Shared-only build: create a test-only static lib 
        idet_test_static = static_library(
            'idet_test_static',
            idet_lib_source,
            include_directories : idet_lib_inc,
            cpp_args            : cxx_args + ['-DIDET_BUILD_STATIC'] + idet_build_defs,
            link_args           : ld_args,
            dependencies        : idet_lib_deps,
            link_depends        : ort_link_depends,
            install             : false,
        )

        idet_internal_dep = declare_dependency(
            link_with           : idet_test_static,
            include_directories : idet_lib_inc,
            dependencies        : idet_internal_compile_deps,
            sources             : ort_link_depends,
            compile_args        : ['-DIDET_TESTING_INTERNAL'],
        )
    endif

    if not idet_internal_dep.found()
        error('IDet internal dependency not found (for tests)')
    endif
endif
