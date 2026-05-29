from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

cuda_sources = glob.glob("csrc/**/*.cu", recursive=True)
cpp_sources = glob.glob("binding/**/*.cpp", recursive=True)

setup(
    name="minicuda",
    packages=["minicuda"],
    ext_modules=[
        CUDAExtension(
            name="minicuda._C",
            sources=cpp_sources + cuda_sources,
            include_dirs=["include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
