project('clipper2', 'cpp', 
    version: '1.5.4',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ]
)

clipper2_inc = include_directories('CPP/Clipper2Lib/include', is_system: true)

# Depends on Clipper2 version, check paths before build!
clipper2_src = files(
    'CPP/Clipper2Lib/src/clipper.engine.cpp',
    'CPP/Clipper2Lib/src/clipper.offset.cpp',
    'CPP/Clipper2Lib/src/clipper.rectclip.cpp',
)

clipper2_lib = static_library('clipper2',
    clipper2_src,
    include_directories: clipper2_inc,
    pic: true,
    install: false,
)

clipper2_dep = declare_dependency(
    link_with: clipper2_lib,
    include_directories: clipper2_inc,
)