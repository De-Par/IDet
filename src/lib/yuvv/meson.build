yuvv_lib_source = files(
    'yuvv.cpp',
)

yuvv_libtype = get_option('default_library')

if yuvv_libtype in ['shared', 'both']

    yuvv_lib_shared = shared_library(
        'yuvv',
        yuvv_lib_source,
        include_directories : yuvv_lib_inc,
        cpp_args            : cxx_args + ['-DYUVV_BUILD_SHARED'],
        link_args           : ld_args,
        dependencies        : yuvv_lib_deps,  # private dependencies
        install             : true,
    )

    yuvv_dep_shared = declare_dependency(
        link_with           : yuvv_lib_shared,
        include_directories : yuvv_lib_inc,
        dependencies        : [opencv_dep],  # public dependencies
        compile_args        : ['-DYUVV_USE_SHARED'],
    )

endif

if yuvv_libtype in ['static', 'both']

    yuvv_lib_static = static_library(
        'yuvv',
        yuvv_lib_source,
        include_directories : yuvv_lib_inc,
        cpp_args            : cxx_args + ['-DYUVV_BUILD_STATIC'],
        link_args           : ld_args,
        dependencies        : yuvv_lib_deps,  # private dependencies
        install             : true,
    )

    yuvv_dep_static = declare_dependency(
        link_with           : yuvv_lib_static,
        include_directories : yuvv_lib_inc,
        dependencies        : [opencv_dep],  # public dependencies
    )

endif

yuvv_dep = disabler()

if yuvv_libtype == 'both'
    # force to use dynamic implementation
    yuvv_dep = yuvv_dep_shared  # or use 'yuvv_dep_static'
elif yuvv_libtype == 'static'
    yuvv_dep = yuvv_dep_static
else  # yuvv_libtype == 'shared'
    yuvv_dep = yuvv_dep_shared
endif

if not yuvv_dep.found()
    error('YUVV library not found')
endif

if yuvv_libtype in ['shared', 'both']
    yuvv_pkg_lib = yuvv_lib_shared
    yuvv_pkg_cflags = ['-DYUVV_USE_SHARED']
else
    yuvv_pkg_lib = yuvv_lib_static
    yuvv_pkg_cflags = []
endif

pkg.generate(
    yuvv_pkg_lib,
    name        : 'YUVV',
    filebase    : 'yuvv',
    version     : meson.project_version(),
    description : 'YUV viewer support library',
    subdirs     : ['yuvv'],
    extra_cflags: yuvv_pkg_cflags,
)
