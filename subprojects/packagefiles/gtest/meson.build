project('gtest', 'cpp',
    version: '1.17.0',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ],
)

threads = dependency('threads')

gtest_inc = include_directories('googletest', 'googletest/include', is_system: true)
gmock_inc = include_directories('googlemock', 'googlemock/include', is_system: true)

gtest_lib = static_library('gtest',
    'googletest/src/gtest-all.cc',
    include_directories: [gtest_inc],
    dependencies: [threads],
)

gtest_main_lib = static_library('gtest_main',
    'googletest/src/gtest_main.cc',
    include_directories: [gtest_inc],
    link_with: [gtest_lib],
    dependencies: [threads],
)

gmock_lib = static_library('gmock',
    'googlemock/src/gmock-all.cc',
    include_directories: [gmock_inc, gtest_inc],
    link_with: [gtest_lib],
    dependencies: [threads],
)

gmock_main_lib = static_library('gmock_main',
    'googlemock/src/gmock_main.cc',
    include_directories: [gmock_inc, gtest_inc],
    link_with: [gmock_lib, gtest_lib],
    dependencies: [threads],
)

gtest_dep = declare_dependency(
    link_with: [gtest_lib],
    include_directories: [gtest_inc],
    dependencies: [threads],
).as_system('system')

gtest_main_dep = declare_dependency(
    link_with: [gtest_main_lib, gtest_lib],
    include_directories: [gtest_inc],
    dependencies: [threads],
).as_system('system')

gmock_dep = declare_dependency(
    link_with: [gmock_lib, gtest_lib],
    include_directories: [gmock_inc, gtest_inc],
    dependencies: [threads],
).as_system('system')

gmock_main_dep = declare_dependency(
    link_with: [gmock_main_lib, gmock_lib, gtest_lib],
    include_directories: [gmock_inc, gtest_inc],
    dependencies: [threads],
).as_system('system')

meson.override_dependency('gtest', gtest_dep)
meson.override_dependency('gtest_main', gtest_main_dep)
meson.override_dependency('gmock', gmock_dep)
meson.override_dependency('gmock_main', gmock_main_dep)
