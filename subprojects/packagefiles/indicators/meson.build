project('indicators', 'cpp',
    version: '2.3',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ]
)

indicators_inc = include_directories('include', is_system: true)

indicators_dep = declare_dependency(
    include_directories: indicators_inc,
)