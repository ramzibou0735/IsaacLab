from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def build_extension():
    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch must be installed in the active environment before building "
            "rlPx4ControllerTorch. Install with the IsaacLab Python interpreter and "
            "use '--no-build-isolation' for editable installs."
        ) from exc

    extension = CppExtension(
        name="rlPx4ControllerTorch._parallel_control",
        sources=[
            str(ROOT / "csrc" / "bindings.cpp"),
            str(ROOT / "csrc" / "parallel_controllers.cpp"),
        ],
        include_dirs=[str(ROOT / "csrc")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )

    return BuildExtension, extension


BuildExtension, extension = build_extension()


setup(
    name="rlPx4ControllerTorch",
    version="0.1.0",
    description="Torch-native parallel PX4-like controllers for IsaacLab",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"rlPx4ControllerTorch": ["py.typed"]},
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    extras_require={
        "dev": ["pytest>=8.0", "sphinx>=7.0", "sphinx-rtd-theme>=2.0"],
    },
    python_requires=">=3.10",
    zip_safe=False,
)
