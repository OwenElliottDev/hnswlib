import os
import sys
import platform

import numpy as np
import pybind11
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = '0.10.0'


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]

# compatibility when run in python_bindings
bindings_dir = 'python_bindings'
if bindings_dir in os.path.basename(os.getcwd()):
    source_files = ['./bindings.cpp']
    include_dirs.extend(['../hnswlib/'])
else:
    source_files = ['./python_bindings/bindings.cpp']
    include_dirs.extend(['./hnswlib/'])


libraries = []
extra_objects = []


ext_modules = [
    Extension(
        'hnswlib',
        source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        language='c++',
        extra_objects=extra_objects,
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def host_supports_arm_bf16():
    """Return True when the build host is an Apple Silicon Mac whose CPU
    supports the bf16 vector instructions (M2 and later)."""
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        return False
    try:
        import subprocess
        out = subprocess.run(['sysctl', '-n', 'hw.optional.arm.FEAT_BF16'],
                             capture_output=True, text=True)
        return out.stdout.strip() == '1'
    except Exception:
        return False


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    compiler_flag_native = '-march=native'
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', compiler_flag_native],  # , '-w'
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if os.environ.get("HNSWLIB_NO_NATIVE"):
        c_opts['unix'].remove(compiler_flag_native)

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append("-fopenmp")
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = BuildExt.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            if not os.environ.get("HNSWLIB_NO_NATIVE"):
                # check that native flag is available
                print('checking avalability of flag:', BuildExt.compiler_flag_native)
                if not has_flag(self.compiler, BuildExt.compiler_flag_native):
                    print('removing unsupported compiler flag:', BuildExt.compiler_flag_native)
                    opts.remove(BuildExt.compiler_flag_native)
                    # for macos add apple-m1 flag if it's available
                    if sys.platform == 'darwin':
                        candidate_flags = []
                        # apple-m1 is the universal-binary-safe baseline but
                        # predates bf16; add it when the host CPU has it so
                        # the native bf16 kernels are compiled in
                        if host_supports_arm_bf16():
                            candidate_flags.append('-mcpu=apple-m1+bf16')
                        candidate_flags.append('-mcpu=apple-m1')
                        for m1_flag in candidate_flags:
                            print('checking avalability of flag:', m1_flag)
                            if has_flag(self.compiler, m1_flag):
                                print('adding flag:', m1_flag)
                                opts.append(m1_flag)
                                break
                            print(f'flag: {m1_flag} is not available')
                else:
                    print(f'flag: {BuildExt.compiler_flag_native} is available')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(BuildExt.link_opts.get(ct, []))

        build_ext.build_extensions(self)


# static metadata lives in pyproject.toml; only the version (dynamic) and
# the extension build remain here
setup(
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
