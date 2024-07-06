"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='emd_ext',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda',
            sources=[
                'cuda/emd.cpp',
                'cuda/emd_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
# 报错：  File "/home/liweigang/miniconda3/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 1774, in _write_ninja_file_and_compile_objects
#     _run_ninja_build(
#   File "/home/liweigang/miniconda3/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 2116, in _run_ninja_build
#     raise RuntimeError(message) from e
# RuntimeError: Error compiling objects for extension