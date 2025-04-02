import os, os.path as osp, glob
from setuptools import find_packages
from setuptools import setup

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension


torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

def make_cuda_ext(name: str, module: str, sources: "list", include_dirs=None, define_macros=None):
    """sources will be concated after module
    """
    if include_dirs is None:
        include_dirs = ["."]
    if define_macros is None:
        define_macros = []
    extra_compile_args = {"cxx": []}
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args['nvcc'] = [
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        extension_cls = CUDAExtension
    else:
        extension_cls = CppExtension

    # It's better if pytorch can do this by default ..
    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))
    
    if module:
        n_sources = []
        for p in sources:
            src_loc = osp.join(*module.split('.'), p) 
            assert osp.exists(src_loc), src_loc
            n_sources.append(src_loc)
        name = '{}.{}'.format(module, name)
    else:
        n_sources = sources
        # name = '{}.{}'.format(module, name)
    ext = extension_cls(
        name=name,
        sources=n_sources,
        define_macros=define_macros,
        extra_compile_args = extra_compile_args,
        include_dirs=include_dirs,
    )
    return ext

CURR_DIR = osp.dirname(osp.abspath(__file__))
def get_mmcv_extensions():
    ### mmcv 1.6.0
    define_macros = []
    define_macros += [('MMCV_WITH_CUDA', None)]
    csrc_dir = osp.join(CURR_DIR, "gdet", "ops", "csrc", )
    pytorch_dir = osp.join(csrc_dir, "pytorch" )
    op_files = glob.glob(osp.join(pytorch_dir, '*.cpp')) + \
                glob.glob(osp.join(pytorch_dir, 'cpu/*.cpp')) + \
                glob.glob(osp.join(pytorch_dir, 'cuda/*.cu')) + \
                glob.glob(osp.join(pytorch_dir, 'cuda/*.cpp'))
    include_dirs = []
    include_dirs.append(osp.join(csrc_dir, "common"))
    include_dirs.append(osp.join(csrc_dir, "common", "cuda"))
    mmcv_ext = make_cuda_ext("gdet._ext", module="", sources=op_files, define_macros=define_macros, include_dirs=include_dirs)
    return mmcv_ext

## 
def get_extensions():
    mmcv_ext = get_mmcv_extensions()
    nms_rotated_cuda = make_cuda_ext(
        name='nms_rotated_cuda',
        module='gdet.ops.nms_rotated',
        sources=['src/nms_rotated_cpu.cpp', 'src/nms_rotated_cuda.cu']
    )

    ext_modules = [
        mmcv_ext,
        nms_rotated_cuda
    ]

    return ext_modules


setup(name='gdet', version='0.5', packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
