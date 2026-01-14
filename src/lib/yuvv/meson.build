yuvv_source = files(
    'yuv_viewer.cpp',
)

build_defines = []
if get_option('default_library') != 'static'
    build_defines += ['-DYUVV_BUILD_SHARED']
endif

yuvv_lib = library(
    'yuvv_lib',
    yuvv_source,
    include_directories : yuvv_inc,
    cpp_args            : cxx_args + build_defines,
    link_args           : ld_args,
    dependencies        : yuvv_deps,
    install             : true,
)

yuvv_lib_dep = declare_dependency(
    link_with           : yuvv_lib,
    include_directories : yuvv_inc,
    dependencies        : yuvv_deps,
)