project('text_det', 'cpp',
    version: '1.0.0',
    default_options: [
        'cpp_std=c++17',
        'warning_level=2',
        'optimization=3',
        'buildtype=release',
        'b_ndebug=if-release',
        'default_library=shared',  # shared / static / both
    ]
)

# ------------------------------------------------------------------------
# Compiler / platform detection
# ------------------------------------------------------------------------

cpp  = meson.get_compiler('cpp')
host = host_machine
sys  = host.system()
cpu  = host.cpu_family()

is_darwin = (sys == 'darwin')
is_linux  = (sys == 'linux')
is_clang  = (cpp.get_id() == 'clang' or cpp.get_id() == 'apple-clang')
is_gcc    = (cpp.get_id() == 'gcc')

# ------------------------------------------------------------------------
# Project options (see meson_options.txt)
# ------------------------------------------------------------------------

is_system_ort = get_option('onnxruntime_system')
ort_inc_dir    = get_option('onnxruntime_inc')
ort_lib_dir    = get_option('onnxruntime_lib')
cpu_opt        = get_option('cpu_opt')
fast_math     = get_option('fast_math')
thinlto       = get_option('thinlto')
use_openmp    = get_option('use_openmp')
use_numa      = get_option('use_numa')
rpath_extra    = get_option('install_rpath_extra')
strict_warn   = get_option('strict_warnings')

# ------------------------------------------------------------------------
# Compile / link flags 
# ------------------------------------------------------------------------

cxx_args = ['-O3']
ld_args  = []

# Smaller/faster binaries via better dead-code elimination
foreach f : ['-ffunction-sections', '-fdata-sections']
    if cpp.has_argument(f)
        cxx_args += f
    endif
endforeach

default_lib = get_option('default_library')  # 'shared', 'static', 'both'
if default_lib != 'static'
    if cpp.has_argument('-fvisibility=hidden')
        cxx_args += '-fvisibility=hidden'
    endif
endif

# Fast-math: relaxed IEEE semantics, usually fine for inference
if fast_math
    foreach f : ['-ffast-math', '-fno-math-errno', '-fno-trapping-math', '-ffp-contract=fast']
        if cpp.has_argument(f)
            cxx_args += f
        endif
    endforeach
endif

# CPU tuning:
#   cpu_opt = generic → no extra -march/-mcpu
#   cpu_opt = native  → auto-tune for build host
#   cpu_opt = <string> → use as -march / -mcpu value
if cpu_opt != '' and cpu_opt != 'generic'
    if cpu == 'x86_64'
        optflag = (cpu_opt == 'native') ? '-march=native' : '-march=' + cpu_opt
        if cpp.has_argument(optflag)
            cxx_args += optflag
        endif
        if cpp.has_argument('-mtune=native')
            cxx_args += '-mtune=native'
        endif
    elif cpu == 'aarch64'
        optflag = (cpu_opt == 'native') ? '-mcpu=native' : '-mcpu=' + cpu_opt
        if cpp.has_argument(optflag)
            cxx_args += optflag
        endif
        if cpp.has_argument('-mtune=native')
            cxx_args += '-mtune=native'
        endif
    endif
endif

# ThinLTO hint for Clang when Meson LTO (b_lto) is enabled
# Usage: meson setup builddir -Db_lto=true -Dthinlto=true
if thinlto and is_clang and get_option('b_lto')
    if cpp.has_argument('-flto=thin')
        cxx_args += '-flto=thin'
        if cpp.has_link_argument('-flto=thin')
            ld_args += '-flto=thin'
        endif
    endif
elif thinlto and get_option('b_lto') and not is_clang
    warning('thinlto=true requested but compiler is not Clang; ignoring ThinLTO hint.')
endif

# Linker dead-stripping (remove unused sections from final binary)
if is_linux
    if cpp.has_link_argument('-Wl,--gc-sections')
        ld_args += '-Wl,--gc-sections'
    endif
elif is_darwin
    foreach f : ['-Wl,-dead_strip', '-Wl,-dead_strip_dylibs', '-Wl,-headerpad_max_install_names']
        if cpp.has_link_argument(f)
            ld_args += f
        endif
    endforeach
endif

# Optional strict warnings – good for CI/dev, can be disabled for noisy code
if strict_warn
    foreach f : ['-Wall', '-Wextra', '-Wpedantic', '-Wconversion', '-Wshadow']
        if cpp.has_argument(f)
            cxx_args += f
        endif
    endforeach
endif

# ------------------------------------------------------------------------
# Dependencies / subprojects
# ------------------------------------------------------------------------

# --- OpenCV (system) ---
opencv_dep = dependency('opencv4', required: true)

# --- OpenMP (system) ---
openmp_dep = disabler()
if use_openmp
    openmp_dep = dependency('openmp', required: true)
endif

# --- NUMA (system) ---
numa_dep = disabler()
if use_numa
    numa_dep = dependency('numa', required: false)
    if numa_dep.found()
        add_project_arguments('-DUSE_LIBNUMA', language: 'cpp')
    else 
        warning('NUMA not found in system and will be excluded from the build!')
    endif
endif

# --- Indicators (wrap) ---
indicators_dep = subproject('indicators').get_variable('indicators_dep')

# --- Clipper2 (wrap) ---
clipper2_dep = subproject('clipper2').get_variable('clipper2_dep')

# --- ONNX Runtime (system -> manual -> cmake subproject fallback) ---
ort_source = 'none'
ort_dep = disabler()
ort_inc = []

# 1) system first (auto + cmake)
if is_system_ort
    ort_dep = dependency('onnxruntime', required: false, method: 'pkg-config')
    if ort_dep.found()
        ort_source = 'system (pkg-config)'
    else
        ort_dep = dependency('onnxruntime', required: false, method: 'cmake')
        if ort_dep.found()
            ort_source = 'system (cmake package)'
        endif
    endif
endif

if not ort_dep.found()
    # 2) manual paths second
    if ort_inc_dir != '' or ort_lib_dir != ''
        if ort_inc_dir == '' or ort_lib_dir == ''
            error('Manual ONNX Runtime requires BOTH -Donnxruntime_inc=... and -Donnxruntime_lib=... (or leave both empty)')
        endif

        header = join_paths('onnxruntime', 'core', 'session', 'onnxruntime_cxx_api.h')
        fs = import('fs')

        if not fs.exists(join_paths(ort_inc_dir, header))
            error('Manual ONNX Runtime: cannot locate ' + header + ' under onnxruntime_inc=' + ort_inc_dir + '. Specify correct -Donnxruntime_inc=...')
        endif

        ort_inc = [
            include_directories(ort_inc_dir),
            # For older ORT headers that use core/session/onnxruntime_cxx_api.h, etc.
            include_directories(join_paths(ort_inc_dir, 'onnxruntime', 'core', 'session')),
        ]
        ort_lib = cpp.find_library('onnxruntime', dirs: [ort_lib_dir], required: true)
        
        ort_dep = declare_dependency(
            dependencies: [ort_lib],
            include_directories: ort_inc,
        )
        ort_source = 'manual -> ' + ort_lib_dir
    else
        # 3) fallback: build ORT from subprojects via CMake module
        warning('ONNX Runtime not found in system (or disabled) and manual paths are empty. Falling back to building from subprojects/onnxruntime (CMake)')
        
        ort_dep = subproject('onnxruntime').get_variable('onnxruntime_dep')
        ort_source = 'subproject (cmake)'

        if not ort_dep.found()
            error('ONNX Runtime not found. Install it, or set -Donnxruntime_inc/-Donnxruntime_lib')
        endif
        
        ort_lib_dir = join_paths(meson.project_build_root(), 'subprojects', 'onnxruntime', '__install', 'lib')
    endif
endif


# ------------------------------------------------------------------------
# RPATH (build + install)
# ------------------------------------------------------------------------

# Runtime paths list for binary
rpaths = [] 

# Typical Homebrew locations for libomp and onnxruntime
if is_darwin
    if openmp_dep.found()
        rpaths += ['/opt/homebrew/opt/libomp/lib'] 
    endif
    # if 
    #     '/opt/homebrew/opt/onnxruntime/lib'
    # ]
endif

# If ONNX Runtime is provided manually, make sure we can find it at runtime
if ort_lib_dir != ''
    rpaths += [ort_lib_dir]
endif

# Extra RPATH specified by user (colon-separated)
if rpath_extra != ''
    rpaths += rpath_extra.split(':')
endif

build_rpath_str   = ''
install_rpath_str = ''

if rpaths.length() > 0
    build_rpath_str   = ':'.join(rpaths)
    install_rpath_str = ':'.join(rpaths)
endif

# ------------------------------------------------------------------------
# Sources / headers / deps
# ------------------------------------------------------------------------

# --- Text Detector Library ---
tdet_deps = [
    opencv_dep,
    numa_dep,
    openmp_dep,
    indicators_dep,
    clipper2_dep,
    ort_dep,
]

tdet_inc = [ 
    include_directories(
        join_paths('include', 'tdet'),
    ),
]

# --- YUV Viewer Library ---
yuvv_deps = [
    opencv_dep,
]

yuvv_inc = [
    include_directories(
        join_paths('include', 'yuvv'),
    ),
]

# --- Text Detector Application ---
tdet_app_inc = [
    include_directories(
        join_paths('include', 'app'), 
    ),
]

# --- YUV Viewer Application ---
yuvv_app_inc = [
    include_directories(
        join_paths('include', 'app'), 
    ),
]

subdir('src')

# ------------------------------------------------------------------------
# Summary log
# ------------------------------------------------------------------------

summary(
    {
        'System'   : sys,
        'CPU'      : cpu,
        'Compiler' : cpp.get_id(),
    },
    section : 'Env',
    bool_yn : true,
)

summary(
    {
        'buildtype'       : get_option('buildtype'),
        'optimization'    : get_option('optimization'),
        'lib_mode'        : get_option('default_library'),
        'lto (b_lto)'     : get_option('b_lto'),
        'thinlto'         : thinlto,
        'extra cpp_args'  : cxx_args,
        'extra link_args' : ld_args,
    },
    list_sep: ' ',
    section : 'Build / Flags',
    bool_yn : true,
)

summary(
    {
        'OpenCV'          : opencv_dep.found(),
        'ONNXRuntime'     : ort_dep.found(),
        '└──> source'     : ort_source,
        'OpenMP'          : use_openmp and openmp_dep.found(),
        'NUMA'            : numa_dep.found(),
    },
    section : 'Deps',
    bool_yn : true,
)

summary(
    {
        'cpu_opt'     : cpu_opt,
        'fast_math'   : fast_math,
        'lto (b_lto)' : get_option('b_lto'),
        'thinlto'     : thinlto,
    },
    section : 'Build options',
    bool_yn : true,
)

if install_rpath_str != ''
    summary(
        { 
            'install_rpath' : install_rpath_str 
        },
        section : 'RPATH',
    )
endif