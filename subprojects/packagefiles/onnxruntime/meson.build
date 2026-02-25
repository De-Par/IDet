project('onnxruntime_wrap', ['c', 'cpp'],
    version: '1.23.2',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'buildtype=release',
    ],
)

python = find_program('python3', 'python', required: true)
cmake = find_program('cmake', required: true)

# ORT repo root (this subproject directory)
ort_src_root = meson.current_source_dir()

# build/install dirs inside Meson build tree
build_dir = join_paths(meson.current_build_dir(), '__cmake_build')
install_dir = join_paths(meson.current_build_dir(), '__install')

script = files('meson_ort_build.py')

cmd = [
    python, script,
    '--cmake', cmake,
    '--src', ort_src_root,
    '--build', build_dir,
    '--install', install_dir,
    '--stamp', '@OUTPUT@',
    '--tests',   (get_option('ort_build_tests')    ? '1' : '0'),
    '--acl',     (get_option('ort_enable_acl')     ? '1' : '0'),
    '--xnnpack', (get_option('ort_enable_xnnpack') ? '1' : '0'),
    '--extra', get_option('ort_cmake_extra'),
    '--ort-ver', meson.project_version(),
    '--force',
]

# Export build-target separately
onnxruntime_stamp = custom_target(
    'onnxruntime_build',
    output: 'onnxruntime_build.stamp',
    command: cmd,
    depend_files: script,
    build_by_default: false,
    console: true, # write logs without buffering by Ninja
)

# NOTE:
# ORT headers are installed into: <install_dir>/include
ort_inc_dir = join_paths(install_dir, 'include')

ort_lib_dir = join_paths(install_dir, 'lib')

ort_link_args = [
    '-L' + ort_lib_dir,
    '-lonnxruntime',
]

onnxruntime_dep = declare_dependency(
    compile_args: ['-isystem', ort_inc_dir],
    link_args: ort_link_args,

    # Force the build/install step to happen before anything that uses this dep
    sources: onnxruntime_stamp,
).as_system('system')
