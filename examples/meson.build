idet_example_sources = [
    'sync_detector.cpp',
    'hot_loop_worker.cpp',
]

idet_examples_build_rpath = ''
idet_examples_install_rpath = ''

if idet_libtype == 'static' or (idet_libtype == 'both' and idet_exe_link == 'static')
    idet_examples_build_rpath = build_rpath
    idet_examples_install_rpath = install_rpath
endif

foreach example_source : idet_example_sources
    executable(
        example_source.split('.')[0],
        files(example_source),
        dependencies  : [idet_dep],
        build_rpath   : idet_examples_build_rpath,
        install_rpath : idet_examples_install_rpath,
        install       : false,
    )
endforeach
