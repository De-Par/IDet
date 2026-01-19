tdet_source = files(
    'tdet.cpp',
    'cross_topology.cpp',
    'dbnet.cpp',
    'face_detector.cpp',
    'drawing.cpp',
    'geometry.cpp',
    'nms.cpp',
    'omp_config.cpp',
    'run.cpp',
    'run_common.cpp',
    'tiling.cpp',
    'cli.cpp',
)

build_defines = []
if get_option('default_library') != 'static'
    build_defines += ['-DTDET_BUILD_SHARED']
endif

if use_blob_models
    fs = import('fs')
    src_root = meson.project_source_root()
    models_root = join_paths(src_root, 'assets', 'models')

    face_model_path = join_paths(models_root, 'scrfd', 'scrfd_500m_bnkps.onnx')
    if not fs.exists(face_model_path)
        error('SCRFD model not found at ' + face_model_path)
    endif

    face_blob = custom_target(
        'face_model_blob_gen',
        input  : face_model_path,
        output : 'face_model_blob_gen.cpp',
        command: ['xxd', '-n', 'scrfd_model', '-i', '@INPUT@'],
        capture: true,
    )
    tdet_source += [face_blob]
    build_defines += ['-DTDET_HAVE_FACE_BLOB']
endif

tdet_lib = library(
    'tdet_lib',
    tdet_source,
    include_directories : tdet_inc,
    cpp_args            : cxx_args + build_defines,
    link_args           : ld_args,
    dependencies        : tdet_deps,
    build_rpath         : build_rpath_str,
    install_rpath       : install_rpath_str,
    install             : true,
)

tdet_lib_dep = declare_dependency(
    link_with           : tdet_lib,
    include_directories : tdet_inc,
    dependencies        : tdet_deps,
)
