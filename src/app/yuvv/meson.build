# --- YUV Viewer Application ---
yuvv_app_deps = [
    yuvv_lib_dep,
]

yuvv_app = executable(
    'yuv_viewer',
    ['main.cpp'],
    cpp_args     : cxx_args,
    link_args    : ld_args,
    dependencies : yuvv_app_deps,
    install      : true,
)