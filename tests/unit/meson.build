unit_tests_sources = files(
    'main.cpp',
    'test_ort_tensor.cpp',
    'test_geometry.cpp',
    'test_tiling.cpp',
    'test_nms.cpp',
)

idet_unit_tests_exe = executable(
    'idet_unit_tests',
    unit_tests_sources,
    dependencies  : tests_deps,
    build_rpath   : build_rpath,
    install_rpath : install_rpath,
    install       : false,
)

test(
    'idet_unit_tests', 
    idet_unit_tests_exe,
    suite: 'unit', 
    timeout: 120,
    args: [
        '--gtest_color=yes',
        '--gtest_print_time=1',
    ],
)
