project('clipper2', 'cpp',
    version: '2.0.1',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ],
)

clipper2_inc = include_directories('CPP/Clipper2Lib/include', is_system: true)

# Depends on Clipper2 version, check paths before build!
clipper2_src = files(
    'CPP/Clipper2Lib/src/clipper.engine.cpp',
    'CPP/Clipper2Lib/src/clipper.offset.cpp',
    'CPP/Clipper2Lib/src/clipper.rectclip.cpp',
    'CPP/Clipper2Lib/src/clipper.triangulation.cpp',
)

clipper2_lib = static_library('clipper2',
    clipper2_src,
    include_directories: clipper2_inc,
    cpp_args: ['-Wno-deprecated-declarations'],
    pic: true,
    install: false,
)

clipper2_dep = declare_dependency(
    link_with: clipper2_lib,
    include_directories: clipper2_inc,
).as_system('system')
