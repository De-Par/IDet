project('stb', 'c',
    version: 'master',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ],
)

stb_inc = include_directories('.', is_system: true)

stb_dep = declare_dependency(
    include_directories: stb_inc,
)

# so dependency('stb') works in the parent project
# meson.override_dependency('stb', stb_dep)
