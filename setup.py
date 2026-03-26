import glob
import os
import os.path as osp
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from setuptools import find_packages, setup

__version__ = "0.1.0"

URL = "https://github.com/PaPieta/gs-voxelizer"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

GLM_VERSION = "1.0.3"
GLM_ARCHIVE_URL = (
    f"https://github.com/g-truc/glm/archive/refs/tags/{GLM_VERSION}.tar.gz"
)
GLM_LOCAL_CACHE = Path(__file__).parent / "gs_voxelizer" / "third_party" / "glm"


def _has_glm_headers(path: Path) -> bool:
    return (path / "glm" / "glm.hpp").exists()


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base_path = Path(path).resolve()
    for member in tar.getmembers():
        member_path = (base_path / member.name).resolve()
        common = os.path.commonpath([str(base_path), str(member_path)])
        if common != str(base_path):
            raise RuntimeError("GLM archive contains illegal paths.")
    tar.extractall(base_path)


def _download_glm(target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "glm.tar.gz"
        print(f"Downloading GLM {GLM_VERSION}...")
        try:
            with urllib.request.urlopen(GLM_ARCHIVE_URL) as response, open(
                archive_path, "wb"
            ) as fh:
                shutil.copyfileobj(response, fh)
        except OSError as exc:
            raise RuntimeError(
                "Unable to download GLM automatically. "
                "Set GLM_HOME to an existing checkout."
            ) from exc
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, tmpdir_path)
        extracted_dir = tmpdir_path / f"glm-{GLM_VERSION}"
        if not extracted_dir.exists():
            raise RuntimeError("Downloaded GLM archive has unexpected structure.")
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.move(str(extracted_dir), str(target_path))


def resolve_glm_include_dir() -> str:
    env_dir = os.getenv("GLM_HOME")
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if _has_glm_headers(candidate):
            return str(candidate)
        raise RuntimeError(f"GLM_HOME was set to {candidate} but no headers found.")

    if not _has_glm_headers(GLM_LOCAL_CACHE):
        _download_glm(GLM_LOCAL_CACHE)

    if not _has_glm_headers(GLM_LOCAL_CACHE):
        raise RuntimeError(
            "GLM headers are still missing after download. "
            f"Please place them in {GLM_LOCAL_CACHE}."
        )

    return str(GLM_LOCAL_CACHE)


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gs_voxelizer", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )

    # remove generated 'hip' files, in case of rebuilds
    sources = [path for path in sources if "hip" not in path]

    undef_macros = []
    define_macros = []
    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    extension = CUDAExtension(
        f"gs_voxelizer.cuda",
        sources,
        include_dirs=[resolve_glm_include_dir()],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


setup(
    name="gs_voxelizer",
    version=__version__,
    description=" Python package for differentiable voxelization of gaussians",
    keywords="gaussian, splatting, cuda, volumetric",
    url=URL,
    python_requires=">=3.7",
    install_requires=[
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
