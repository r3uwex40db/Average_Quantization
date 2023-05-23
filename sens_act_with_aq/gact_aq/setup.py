from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='gact_aq',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'gact_aq.cpp_extension.calc_precision',
              ['gact_aq/cpp_extension/calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'gact_aq.cpp_extension.minimax',
              ['gact_aq/cpp_extension/minimax.cc', 'gact_aq/cpp_extension/minimax_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'gact_aq.cpp_extension.quantization',
              ['gact_aq/cpp_extension/quantization.cc',
                  'gact_aq/cpp_extension/quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
      )
