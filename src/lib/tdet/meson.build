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
