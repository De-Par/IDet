idet_app_source = files(
    'main.cpp',
    'cli.cpp',
    'io.cpp',
)

idet_app_deps = [
    idet_dep,
    opencv_dep,
    indicators_dep,
]

idet_app_build_rpath   = ''
idet_app_install_rpath = ''

# In static mode, ONNX Runtime becomes a direct runtime dependency of the application,
# so the RPATH must be set on the executable, not only on the shared library
if idet_libtype == 'static' or (idet_libtype == 'both' and idet_exe_link == 'static')
    idet_app_build_rpath   = build_rpath
    idet_app_install_rpath = install_rpath
endif

executable(
    'idet_app',
    idet_app_source,
    include_directories : [],
    cpp_args            : cxx_args,
    link_args           : ld_args,
    dependencies        : idet_app_deps,
    build_rpath         : idet_app_build_rpath,
    install_rpath       : idet_app_install_rpath,
    install             : true,
)
