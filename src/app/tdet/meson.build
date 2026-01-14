# --- Text Detector Application ---
tdet_app_deps = [
    tdet_lib_dep,
    yuvv_lib_dep,
]

tdet_app = executable(
    'text_det',
    ['main.cpp'],
    include_directories : [],
    cpp_args            : cxx_args,
    link_args           : ld_args,
    dependencies        : tdet_app_deps,
    install             : true,
)