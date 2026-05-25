# 环境配置指南

## 目录

- [系统要求](#系统要求)
- [C++ 开发环境](#c-开发环境)
- [Python 训练环境](#python-训练环境)
- [ONNX Runtime 配置](#onnx-runtime-配置)
- [GPU 加速配置](#gpu-加速配置)
- [验证安装](#验证安装)

---

## 系统要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| Windows | 10/11 | |
| GCC (MinGW-w64) | 13+ | C++17 |
| CMake | 3.22+ | |
| Python | 3.11+ | |
| GPU (可选) | NVIDIA RTX 20xx+ | 训练/推理加速 |

---

## C++ 开发环境

### 1. MinGW-w64 GCC

本项目使用 GCC 15.2，通过 Scoop 安装：

```powershell
scoop install mingw
```

验证：

```powershell
g++ --version
```

### 2. CMake

```powershell
scoop install cmake
# 或从 https://cmake.org/download/ 下载安装
```

### 3. SFML 2.6.2

项目已内置 `3rdparty/SFML_new/`，使用当前 GCC 编译的 SFML 2.6.2。

如需重新编译：

```powershell
cd 3rdparty
git clone https://github.com/SFML/SFML.git SFML_src
cd SFML_src
git checkout 2.6.2
cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build
# 将 build/lib/*.dll 和 build/lib/*.a 复制到 3rdparty/SFML_new/
```

### 4. ONNX Runtime C++ SDK

#### CPU 版 (仅 CPU 推理)

```powershell
cd 3rdparty
curl -L -o onnxruntime.zip "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-win-x64-1.26.0.zip"
unzip onnxruntime.zip
mv onnxruntime-win-x64-1.26.0 onnxruntime
```

#### GPU 版 (CUDA 推理, 推荐)

```powershell
cd 3rdparty
curl -L -o onnxruntime.zip "https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-win-x64-gpu-1.26.0.zip"
unzip onnxruntime.zip
mv onnxruntime-win-x64-gpu-1.26.0 onnxruntime
```

目录结构：

```
3rdparty/onnxruntime/
├── include/          # 头文件
│   └── onnxruntime_cxx_api.h
└── lib/              # 库和 DLL
    ├── onnxruntime.dll
    ├── onnxruntime.lib
    ├── onnxruntime_providers_cuda.dll    # GPU 版才有
    ├── onnxruntime_providers_shared.dll
    └── onnxruntime_providers_tensorrt.dll
```

---

## Python 训练环境

### 1. 创建虚拟环境

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. 安装依赖

```powershell
# PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio

# ONNX Runtime (GPU 版)
pip install onnxruntime-gpu

# 其他依赖
pip install numpy
```

### 3. 验证

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import onnxruntime; print('Providers:', onnxruntime.get_available_providers())"
```

预期输出：

```
CUDA: True
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## GPU 加速配置

### C++ 端

ONNX Runtime GPU SDK 自带 `onnxruntime_providers_cuda.dll`，但还依赖 CUDA 运行时库
(cublas, cudnn 等)。这些库随 PyTorch 安装，CMake 会自动从 PyTorch 目录拷贝到 build 目录。

**无需单独安装 CUDA Toolkit。**

CMakeLists.txt 中的自动拷贝逻辑：

```cmake
# 查找 PyTorch 安装目录下的 CUDA DLL
execute_process(
    COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib')"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    ...
)
# 拷贝 cublas, cudnn, cudart 等 DLL 到 build 目录
```

### 代码端

`az_onnx_evaluator.cpp` 中自动尝试 CUDA provider：

```cpp
try {
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
} catch (...) {
    // 失败则回退 CPU
}
```

### 手动拷贝 CUDA DLL (仅当自动拷贝失败时)

```powershell
# 查找 PyTorch CUDA 库位置
$TORCH_LIB = python -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')"

# 拷贝到 build 目录
copy "$TORCH_LIB\cublas*.dll" build_onnx\
copy "$TORCH_LIB\cudnn*.dll" build_onnx\
copy "$TORCH_LIB\cudart*.dll" build_onnx\
copy "$TORCH_LIB\cufft*.dll" build_onnx\
copy "$TORCH_LIB\curand*.dll" build_onnx\
copy "$TORCH_LIB\cusolver*.dll" build_onnx\
copy "$TORCH_LIB\cusparse*.dll" build_onnx\
```

---

## 编译

### CPU 版

```powershell
cmake -G "MinGW Makefiles" -B build_onnx -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX=ON
cmake --build build_onnx
```

### GPU 版 (需要先配置 GPU SDK)

编译命令相同，CMake 自动检测并拷贝 CUDA DLL。

### 构建目标

| 目标 | 说明 |
|---|---|
| `dots_and_boxes` | GUI 对弈程序 |
| `selfplay` | 前向自对弈 CLI |
| `evaluate_ai` | Arena 评估工具 |
| `backward_selfplay` | 反向自对弈 CLI |
| `az_tests` | 单元测试 (27 项) |

---

## 验证安装

### 1. 编译

```powershell
cmake --build build_onnx
```

### 2. 单元测试

```powershell
.\build_onnx\az_tests.exe
# 预期: === Results: 27/27 passed, 0 failed ===
```

### 3. GPU 推理测试

```powershell
.\build_onnx\evaluate_ai.exe 1 data/models/backward_model.onnx heuristic --sims 200 --temp 0
# 预期第一行输出: [ONNXEvaluator] CUDA provider enabled (GPU 0)
```

### 4. 训练测试

```powershell
cd train
python model.py resnet_s
# 预期: Total params: 330,619
```

### 5. 双模型 Smoke Test

```powershell
.\build_onnx\backward_selfplay.exe 1 100 0 data/test --black-model data/models/backward_model_black.onnx --white-model data/models/backward_model_white.onnx
# 预期: Evaluator: Dual (BLACK=..., WHITE=...)
```
