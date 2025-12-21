project('onnxruntime_wrap', 'cpp',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ]
)

python = find_program('python3', 'python', required: true)
cmake  = find_program('cmake', required: true)

# ORT repo root (this subproject directory)
src_root = meson.current_source_dir()

# build/install dirs inside Meson build tree
bld_dir  = join_paths(meson.current_build_dir(), '__cmake_build')
inst_dir = join_paths(meson.current_build_dir(), '__install')

script = files('meson_ort_build.py')

cmd = [
    python, script,
    '--cmake', cmake,
    '--src', src_root,
    '--build', bld_dir,
    '--install', inst_dir,
    '--stamp', '@OUTPUT@',
    '--tests',   (get_option('ort_build_tests')    ? '1' : '0'),
    '--acl',     (get_option('ort_enable_acl')     ? '1' : '0'),
    '--xnnpack', (get_option('ort_enable_xnnpack') ? '1' : '0'),
    '--extra', get_option('ort_cmake_extra'),
]

ort_stamp = custom_target(
    'onnxruntime_build',
    output: 'onnxruntime_build.stamp',
    command: cmd,
    build_by_default: true,
    console: true,  # write logs without bufferization by Ninja
)

ort_inc = include_directories('include', is_system: true)
ort_lib_dir = join_paths(inst_dir, 'lib')

ort_link_args = [
    '-L' + ort_lib_dir, 
    '-lonnxruntime'
]

onnxruntime_dep = declare_dependency(
    include_directories: ort_inc,
    link_args: ort_link_args,
    sources: [ort_stamp],
)