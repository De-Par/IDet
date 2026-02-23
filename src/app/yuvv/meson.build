yuvv_app_source = files(
    'main.cpp',
    'cli.cpp',
)

yuvv_app_deps = [
    yuvv_dep,
]

yuvv_app = executable(
    'yuv_viewer',
    yuvv_app_source,
    cpp_args     : cxx_args,
    link_args    : ld_args,
    dependencies : yuvv_app_deps,
    install      : true,
)