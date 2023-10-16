from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="op",
    include_dirs=["include"],
    ext_modules=[
        CppExtension(
            "op",
            [
                "calculate_force.cpp", 
                "register_op.cpp",
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
