project('IDet', ['c', 'cpp'],
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

fs = import('fs')
cpp = meson.get_compiler('cpp')
cc  = meson.get_compiler('c')
sys      = host_machine.system()
cpu_fam  = host_machine.cpu_family()
endian   = host_machine.endian()

is_cross  = meson.is_cross_build()
is_darwin = (sys == 'darwin')
is_linux  = (sys == 'linux')
is_clang  = (cpp.get_id() == 'clang' or cpp.get_id() == 'apple-clang')
is_gcc    = (cpp.get_id() == 'gcc')

# ------------------------------------------------------------------------
# Project options (see meson_options.txt)
# ------------------------------------------------------------------------

idet_libtype   = get_option('idet_libtype')
idet_exe_link  = get_option('idet_exe_link')
is_system_ort = get_option('onnxruntime_system')
ort_inc_dir    = get_option('onnxruntime_inc')
ort_lib_dir    = get_option('onnxruntime_lib')
embed_models  = get_option('embed_onnx_models')
cpu_opt        = get_option('cpu_opt')
fast_math     = get_option('fast_math')
thinlto       = get_option('thinlto')
use_openmp    = get_option('use_openmp')
use_numa      = get_option('use_numa')
rpath_extra    = get_option('install_rpath_extra')
strict_warn   = get_option('strict_warnings')
build_tests   = get_option('build_tests')

# ------------------------------------------------------------------------
# Compile / link flags
# ------------------------------------------------------------------------

cxx_args = []
ld_args  = []

# Smaller/faster binaries via better dead-code elimination
foreach f : ['-ffunction-sections', '-fdata-sections']
    if cpp.has_argument(f)
        cxx_args += f
    endif
endforeach

default_lib = get_option('default_library')  # shared / static / both
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

    if is_clang and cpp.has_argument('-Wno-nan-infinity-disabled')
        cxx_args += '-Wno-nan-infinity-disabled'
    endif
endif

# CPU tuning:
#   cpu_opt = generic → no extra -march/-mcpu
#   cpu_opt = native  → auto-tune for build host
#   cpu_opt = <string> → use as -march / -mcpu value
if cpu_opt != '' and cpu_opt != 'generic'
    if cpu_fam == 'x86_64'
        optflag = (cpu_opt == 'native') ? '-march=native' : '-march=' + cpu_opt
        if cpp.has_argument(optflag)
            cxx_args += optflag
        endif
        if cpp.has_argument('-mtune=native')
            cxx_args += '-mtune=native'
        endif
    elif cpu_fam == 'aarch64'
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
message('Run-time dependencies...')

# We will treat deps headers as system to keep static analysis focused on project code!
# dependency(...) : add flag `include_type: 'system'`
# declare_dependency(...) : wrap in `.as_system('system')`

# --- OpenCV (system) ---
opencv_dep = dependency('opencv4', required: true, include_type: 'system')

# --- OpenMP (system) ---
openmp_dep = disabler()
if use_openmp
    openmp_base = dependency('openmp', required: false, include_type: 'system')
    if not openmp_base.found()
        error('OpenMP requested but not found. Install OpenMP dev/runtime or disable: -Duse_openmp=false')
    endif

    # macOS + clang: force a single libomp location to avoid duplicate runtime
    # loads (e.g. /opt/homebrew/opt/libomp vs /opt/homebrew/opt/llvm/lib)
    if is_darwin and is_clang
        libomp_candidates = [
            '/opt/homebrew/opt/libomp/lib/libomp.dylib', # Apple Silicon Homebrew
            '/usr/local/opt/libomp/lib/libomp.dylib',    # Intel Homebrew
        ]

        pinned_libomp = ''
        foreach p : libomp_candidates
            if fs.exists(p)
                pinned_libomp = p
                break
            endif
        endforeach

        if pinned_libomp != ''
            openmp_dep = declare_dependency(
                dependencies: [openmp_base.partial_dependency(compile_args: true)],
                link_args: [pinned_libomp],
            ).as_system('system')
            message('OpenMP runtime pinned to: ' + pinned_libomp)
        else
            openmp_dep = openmp_base
            warning('Homebrew libomp not found in known paths; using default OpenMP dependency resolution')
        endif
    else
        openmp_dep = openmp_base
    endif
endif

# --- NUMA (system) ---
numa_dep = disabler()
if use_numa
    numa_dep = dependency('numa', required: false, include_type: 'system')
    if numa_dep.found()
        add_project_arguments('-DUSE_LIBNUMA', language: 'cpp')
    else
        warning('NUMA not found in system and will be excluded from the build!')
    endif
endif

# --- Google Tests (system -> subproject fallback) ---
gtest_dep = disabler()
if build_tests
    gtest_dep = dependency('gtest', required: false, include_type: 'system')
    if not gtest_dep.found()
        warning('GTest not found in system. Falling back to building from subprojects/gtest')
        gtest_dep = subproject('gtest').get_variable('gtest_dep')
        if not gtest_dep.found()
            error('GTest is not available (system and wrap fallback both failed). Disable tests in meson options or install it manually')
        endif
    endif
endif

# --- Indicators (wrap) ---
indicators_dep = subproject('indicators').get_variable('indicators_dep')

# --- Clipper2 (wrap) ---
clipper2_dep = subproject('clipper2').get_variable('clipper2_dep')

# --- Stb (wrap) ---
stb_dep = subproject('stb').get_variable('stb_dep')

# --- ONNX Runtime (system -> manual -> subproject fallback) ---
ort_source = 'none'
ort_dep = disabler()
ort_link_depends = []
ort_inc = []

# 1) system first (auto + cmake)
if is_system_ort
    ort_dep = dependency('onnxruntime', required: false, method: 'pkg-config', include_type: 'system')
    if ort_dep.found()
        ort_source = 'system (pkg-config)'
    else
        ort_dep = dependency('onnxruntime', required: false, method: 'cmake', include_type: 'system')
        if ort_dep.found()
            ort_source = 'system (cmake package)'
        endif
    endif
endif

is_ort_installed_by_sys = false
if not ort_dep.found()
    # 2) manual paths second
    if ort_inc_dir != '' or ort_lib_dir != ''
        if ort_inc_dir == '' or ort_lib_dir == ''
            error('Manual ONNX Runtime requires BOTH -Donnxruntime_inc=... and -Donnxruntime_lib=... (or leave both empty)')
        endif

        header = join_paths('onnxruntime', 'core', 'session', 'onnxruntime_cxx_api.h')

        if not fs.exists(join_paths(ort_inc_dir, header))
            error('Manual ONNX Runtime: cannot locate ' + header + ' under onnxruntime_inc=' + ort_inc_dir + '. Specify correct -Donnxruntime_inc=...')
        endif

        ort_inc = [
            include_directories(ort_inc_dir, is_system: true),
            # For older ORT headers that use core/session/onnxruntime_cxx_api.h, etc.
            include_directories(join_paths(ort_inc_dir, 'onnxruntime', 'core', 'session'), is_system: true),
        ]
        ort_lib = cpp.find_library('onnxruntime', dirs: [ort_lib_dir], required: true)

        ort_dep = declare_dependency(
            dependencies: [ort_lib],
            include_directories: ort_inc,
        ).as_system('system')

        ort_source = 'manual -> ' + ort_lib_dir
    else
        # 3) fallback: build ORT from subprojects via CMake module
        warning('ONNX Runtime not found in system (or disabled) and manual paths are empty. Falling back to building from subprojects/onnxruntime (CMake)')

        ort_sp = subproject('onnxruntime')
        ort_dep = ort_sp.get_variable('onnxruntime_dep')
        ort_link_depends = [ort_sp.get_variable('onnxruntime_stamp')]
        ort_source = 'subproject (cmake)'

        if not ort_dep.found()
            error('ONNX Runtime not found. Install it, or set -Donnxruntime_inc/-Donnxruntime_lib')
        endif

        ort_lib_dir = join_paths(meson.project_build_root(), 'subprojects', 'onnxruntime', '__install', 'lib')
    endif
else
    is_ort_installed_by_sys = true
endif

# ------------------------------------------------------------------------
# RPATH (build + install)
# ------------------------------------------------------------------------

# Runtime paths list for binary
rpaths = []

# Typical Homebrew location for onnxruntime
if is_darwin
    if ort_dep.found() and is_ort_installed_by_sys
        rpaths += ['/opt/homebrew/opt/onnxruntime/lib']
    endif
endif

# If ONNX Runtime is provided manually, make sure we can find it at runtime
if ort_lib_dir != ''
    rpaths += [ort_lib_dir]
endif

# Extra RPATH specified by user (colon-separated)
if rpath_extra != ''
    rpaths += rpath_extra.split(':')
endif

build_rpath   = ''
install_rpath = ''

if rpaths.length() > 0
    build_rpath   = ':'.join(rpaths)
    install_rpath = ':'.join(rpaths)
endif

# ------------------------------------------------------------------------
# Headers and dependencies
# ------------------------------------------------------------------------

# --- IDet Library ---

idet_lib_deps = [
    ort_dep,
    opencv_dep,
    numa_dep,
    openmp_dep,
    stb_dep,
]

idet_public_inc = include_directories(
    join_paths('include', 'idet'),
)

idet_private_inc = include_directories(
    join_paths('src', 'lib', 'idet'),
)

idet_lib_inc = [idet_public_inc, idet_private_inc]

# --- YUV Viewer Library ---

yuvv_lib_deps = [
    opencv_dep,
]

yuvv_lib_inc = include_directories(
    join_paths('include', 'yuvv'),
)

# ------------------------------------------------------------------------
# Enters subdirectory and executes meson build configs
# ------------------------------------------------------------------------

subdir('src') # libs / apps

if build_tests
    subdir('tests') # unit tests
endif

# ------------------------------------------------------------------------
# Summary log
# ------------------------------------------------------------------------

summary(
    {
        'System'     : sys,
        'CPU family' : cpu_fam,
        'Endian'     : endian,

        # Host vs build machines
        'Build machine' : '@0@/@1@'.format(build_machine.system(), build_machine.cpu_family()),
        'Host machine'  : '@0@/@1@'.format(host_machine.system(), host_machine.cpu_family()),

        # Compilers: id/version/exe
        'C compiler'   : '@0@ @1@'.format(cc.get_id(),  cc.version()),
        'C++ compiler' : '@0@ @1@'.format(cpp.get_id(), cpp.version()),
    },
    section : 'Environment',
    bool_yn : true,
)

summary(
    {
        'build_type'   : get_option('buildtype'),
        'lib_mode'     : get_option('default_library'),
        'optimization' : get_option('optimization'),
        'lto (b_lto)'  : get_option('b_lto'),
        'cross_build'  : is_cross,
        'thinlto'      : thinlto,
        'strict_warn'  : strict_warn,
        'build_tests'  : build_tests,
        'fast_math'    : fast_math,
        'cpp_args'     : cxx_args,
        'link_args'    : ld_args,
        'cpu_opt'      : cpu_opt,
    },
    list_sep : ' ',
    section  : 'Build options',
    bool_yn  : true,
)

summary(
    {
        'build_rpath'   : (build_rpath != '')   ? build_rpath   : 'none',
        'install_rpath' : (install_rpath != '') ? install_rpath : 'none',
    },
    section : 'Runtime search paths',
    bool_yn : true,
)

summary(
    {
        'opencv4'     : opencv_dep.found(),
        'onnxruntime' : ort_dep.found(),
        '└──> source' : ort_source,
        'openmp'      : use_openmp and openmp_dep.found(),
        'numa'        : numa_dep.found(),
        'gtest'       : gtest_dep.found(),
    },
    section : 'Dependencies',
    bool_yn : true,
)
