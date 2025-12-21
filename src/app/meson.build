# --- Text Detector Application ---
tdet_app_deps = [
    tdet_lib_dep,
    yuvv_lib_dep,
]

tdet_app = executable(
    'text_det',
    ['tdet_app.cpp'],
    include_directories : tdet_app_inc,
    cpp_args            : cxx_args,
    link_args           : ld_args,
    dependencies        : tdet_app_deps,
    install             : true,
)

# --- YUV Viewer Application ---
yuvv_app_deps = [
    yuvv_lib_dep,
]

yuvv_app = executable(
    'yuv_viewer',
    ['yuvv_app.cpp'],
    cpp_args     : cxx_args,
    link_args    : ld_args,
    dependencies : yuvv_app_deps,
    install      : true,
)