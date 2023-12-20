- [1. ubuntu](#1-ubuntu)
- [2. win10](#2-win10)
  - [2.1. 成了](#21-成了)
- [3. 测试](#3-测试)
  - [3.1. 解决问题](#31-解决问题)
    - [3.1.1. cl 警告](#311-cl-警告)
    - [3.1.2. nvcc\_args](#312-nvcc_args)
    - [3.1.3. CUB](#313-cub)
    - [3.1.4. 梯子](#314-梯子)
    - [3.1.5. ninja](#315-ninja)
    - [3.1.6. DISTUTILS\_USE\_SDK](#316-distutils_use_sdk)
    - [3.1.7. torch的编译问题 lazy\_init\_num\_threads](#317-torch的编译问题-lazy_init_num_threads)
    - [3.1.8. link错误](#318-link错误)
    - [3.1.9. pip安装时typing\_extensions](#319-pip安装时typing_extensions)
- [4. 纯失败](#4-纯失败)
  - [4.1. conda 会冲突](#41-conda-会冲突)

---

测试环境：11.7

## 1. ubuntu

能装。python=3.10, cuda 11.7/11.8

1. 系统cuda
2. build-essential
    ```bash
    sudo apt install build-essential
    # gcc, g++
    sudo apt install gcc
    sudo apt install g++
    # 不要使用 conda install gcc。否则会出现，gcc用conda安装的，g++用apt安装的，两者版本不一致而导致编译失败。
    ```
3. 
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    ```
    ```bash
    conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
    ```
    PS： 不用 ninja, ubuntu上用了 ninja 则编译不成功。

3. 安装

    ```bash
    pip install git+https://github.com/facebookresearch/pytorch3d.git
    ```
    网络不好就下载
    ```bash
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install .
    ```

## 2. win10
### 2.1. 成了
1. cuda 12.1 成功了 (11.6, 11.7, 11.8 都未成功过)
    - 需要装系统 cuda (`nvcc -V`)。`iopath` 与nvcc有关。

    - 不需要装 conda cuda, 但 pytorch的cuda版本(`torch.version.cuda`)需要和系统cuda一致

    - cudnn：不需要装cudnn，但如果装了，cudnn又需要得是系统cuda。

    - torch
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
2. 依赖

    gcc/g++的mingw不用装，vs的cl,link装好就行。

    ```bash
    conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
    ```
3. 然后不需要什么也没做（没有修改 cl 警告、 nvcc_args、 CUB、DISTUTILS_USE_SDK和PYTORCH3D_NO_NINJA、lazy_init_num_threads。没遇到 link错误），甚至用了 ninja， 就成了。

    ```bash
    pip install ninja
    pip install .
    ```
## 3. 测试

```python
import pytorch3d
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
```
都不报错就成了。

```python
>>> from pytorch3d.utils import ico_sphere
ImportError: cannot import name '_C' from 'pytorch3d' (D:\git\pytorch3d\pytorch3d\__init__.py)
```
这是因为工程风格是 flat-layout，需要换个路径打开python就好了。<https://github.com/facebookresearch/pytorch3d/issues/1013>
### 3.1. 解决问题

#### 3.1.1. cl 警告

```
cl: 命令行 warning D9002 :忽略未知选项“-std=c++17”
```
```python
- extra_compile_args = {"cxx": ["-std=c++17"]}
+ extra_compile_args = {"cxx": []}
```
说是可以这样改，但只是warning, 似乎不用改。

#### 3.1.2. nvcc_args
```python
  nvcc_args = [
      "-DCUDA_HAS_FP16=1",
      "-D__CUDA_NO_HALF_OPERATORS__",
      "-D__CUDA_NO_HALF_CONVERSIONS__",
      "-D__CUDA_NO_HALF2_OPERATORS__",
+     "-DWIN32_LEAN_AND_MEAN"
  ]
```

#### 3.1.3. CUB
> For the CUB build time dependency, which you only need if you have CUDA **older than 11.7**, if you are using conda, you can continue with

意思是小于11.7的才需要安装 NVIDIA CUB，没有集中, 大于等于11.7的不需要，CUB已经集成在cuda里. 

CUDA 11.7 has CUB version 1.15 integrated., `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include\cub`。

但是有人说集成的cub太老，需要安装cub-1.17.1

两种思路。第一种是，新建文件夹，`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cub-1.17.1`, `set CUB_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cub-1.17.1`；第二种直接替换原来的 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cub`。但是两种结果都失败。

`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include\thrust/system/cuda/config.h`:
```c
+ #define THRUST_IGNORE_CUB_VERSION_CHECK
  #ifndef THRUST_IGNORE_CUB_VERSION_CHECK
```
#### 3.1.4. 梯子

https://github.com/sword4869/learn_git/blob/main/docs/config.md#122-wsl

#### 3.1.5. ninja

```bash
pip install ninja

'''
  warnings.warn(
running bdist_wheel
D:\Applications\miniconda\envs\ldm\lib\site-packages\torch\utils\cpp_extension.py:476: UserWarning: Attempted to use ninja as the BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
  warnings.warn(msg.format('we could not find ninja.'))
'''
意思是不用 ninja 编译的慢
```

```
Traceback (most recent call last):
  File "D:\Applications\miniconda\envs\gaussian_splatting\Lib\site-packages\torch\utils\cpp_extension.py", line 1814, in _run_ninja_build
    env=env)
  File "D:\Applications\miniconda\envs\gaussian_splatting\lib\subprocess.py", line 512, in run
    output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
```

修改文件 `"D:\Applications\miniconda\envs\py3d\Lib\site-packages\torch\utils\cpp_extension.py"`
```python
  def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
-     command = ['ninja', '-v']
+     command = ['ninja', '--version']
```

或者不用ninja
```
set PYTORCH3D_NO_NINJA=1
# 因为 setup.py 里相关实现 L135
# if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":
```
#### 3.1.6. DISTUTILS_USE_SDK

```
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "D:\git\pytorch3d\setup.py", line 146, in <module>
          setup(
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\__init__.py", line 107, in setup
          return distutils.core.setup(**attrs)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\core.py", line 185, in setup
          return run_commands(dist)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\core.py", line 201, in run_commands
          dist.run_commands()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\wheel\bdist_wheel.py", line 364, in run
          self.run_command("build")
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\command\build.py", line 131, in run
          self.run_command(cmd_name)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\command\build_ext.py", line 84, in run
          _build_ext.run(self)
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\setuptools\_distutils\command\build_ext.py", line 345, in run
          self.build_extensions()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\torch\utils\cpp_extension.py", line 485, in build_extensions
          compiler_name, compiler_version = self._check_abi()
        File "D:\Applications\miniconda\envs\py3d\lib\site-packages\torch\utils\cpp_extension.py", line 875, in _check_abi
          raise UserWarning(msg)
      UserWarning: It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.This may lead to multiple activations of the VC env.Please set `DISTUTILS_USE_SDK=1` and try again.
      [end of output]
```
```
set DISTUTILS_USE_SDK=1
```
#### 3.1.7. torch的编译问题 lazy_init_num_threads

```
Creating library E:\AI_hub\pytorch3d-0.7.2\build\temp.win-amd64-cpython-39\Release\AI_hub\pytorch3d-0.7.2\pytorch3d\csrc\ball_query_C.cp39-win_amd64.lib and Object E:\AI_hub\pytorch3d-0.7.2\build\temp.win-amd64-cpython-39\Release\AI_hub\pytorch3d-0.7.2\pytorch3d\csrc\ball_query_C.cp39-win_amd64.exp
ball_query_cpu.obj : error LNK2001: unresolved external symbol _imp___tls_offset?init@?1??lazy_init_num_threads@internal@at@@yaxxz@4_NA
marching_cubes.obj : error LNK2001: unresolved external symbol _imp___tls_offset?init@?1??lazy_init_num_threads@internal@at@@yaxxz@4_NA
ball_query_cpu.obj : error LNK2001: unresolved external symbol _imp___tls_index?init@?1??lazy_init_num_threads@internal@at@@yaxxz@4_NA
marching_cubes.obj : error LNK2001: unresolved external symbol _imp___tls_index?init@?1??lazy_init_num_threads@internal@at@@yaxxz@4_NA
build\lib.win-amd64-cpython-39\pytorch3d_C.cp39-win_amd64.pyd : fatal error LNK1120: 2 unresolved external commands
error: command 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\HostX64\x64\link.exe' failed with exit code 1120
```

`D:\Applications\miniconda\envs\py3d\Lib\site-packages\torch\include\ATen\Parallel.h`
```c
// Initialise num_threads lazily at first parallel call
- inline void lazy_init_num_threads() {
-   thread_local bool init = false;
-   if (C10_UNLIKELY(!init)) {
-     at::init_num_threads();
-     init = true;
-   }
- }
+ TORCH_API void lazy_init_num_threads();
```

#### 3.1.8. link错误

```
marching_cubes.obj : error LNK2001: 无法解析的外部符号 "public: long __cdecl at::Tensor::item<long>(void)const " (??$item@J@Tensor@at@@QEBAJXZ)
marching_cubes.obj : error LNK2001: 无法解析的外部符号 "public: long * __cdecl at::TensorBase::data_ptr<long>(void)const " (??$data_ptr@J@TensorBase@at@@QEBAPEAJXZ)
build\lib.win-amd64-cpython-310\pytorch3d\_C.cp310-win_amd64.pyd : fatal error LNK1120: 2 个无法解析的外部命令
error: command 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.37.32822\\bin\\HostX64\\x64\\link.exe' failed with exit code 1120
```

#### 3.1.9. pip安装时typing_extensions

分明和之前一样啊，但是为什么报错了。而且`typing_extensions`分明存在的

```
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [10 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/home/lab/project/pytorch3d/setup.py", line 15, in <module>
          import torch
        File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/torch/__init__.py", line 1122, in <module>
          from .serialization import save, load
        File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/torch/serialization.py", line 17, in <module>
          from typing_extensions import TypeAlias  # Python 3.10+
      ModuleNotFoundError: No module named 'typing_extensions'
      ....
      [end of output]
```

```
(ldm) lab@eleven:~/project/pytorch3d$ pip show typing_extensions
Name: typing_extensions
Version: 4.4.0
Summary: Backported and Experimental Type Hints for Python 3.7+
Home-page: 
Author: 
Author-email: "Guido van Rossum, Jukka Lehtosalo, Łukasz Langa, Michael Lee" <levkivskyi@gmail.com>
License: 
Location: /home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages
Requires: 
Required-by: altair, huggingface-hub, lightning-utilities, pytorch-lightning, qudida, streamlit, torch
```
原来是 torch 真出问题了。

```bash

(ldm) lab@eleven:~/project/DECA$ python
Python 3.10.12 (main, Jul  5 2023, 18:54:27) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/torch/__init__.py", line 1122, in <module>
    from .serialization import save, load
  File "/home/lab/miniconda3/envs/ldm/lib/python3.10/site-packages/torch/serialization.py", line 17, in <module>
    from typing_extensions import TypeAlias  # Python 3.10+
ModuleNotFoundError: No module named 'typing_extensions'
```

那么为什么torch出问题了，原因不是包的问题。而是因为安装了 `conda install -c bottler nvidiacub`.

## 4. 纯失败

### 4.1. conda 会冲突

```bash
sudo apt install gcc
sudo apt install g++
conda create -n pytorch3d python=3.11 -y
conda activate pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
```bash
conda install pytorch3d -c pytorch3d
```
哪怕新环境，没安pytorch，也有冲突。
```bash
The following specifications were found to be incompatible with your system:
  - feature:/linux-64::__glibc==2.35=0
  - feature:|@/linux-64::__glibc==2.35=0
  - pytorch3d -> torchvision[version='>=0.5'] -> __glibc[version='>=2.17,<3.0.a0']
Your installed version is: 2.35
```