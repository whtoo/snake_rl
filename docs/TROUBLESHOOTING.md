# 故障排除指南

本文档描述了在运行Snake RL项目时可能遇到的常见错误及其解决方案。

## 常见错误及解决方案

### 1. OpenMP错误

**错误信息:**
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**原因:** 多个库（如PyTorch、NumPy、OpenCV等）都包含了OpenMP运行时，导致冲突。

**解决方案:**
- 项目已自动设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`
- 限制OpenMP线程数为1: `OMP_NUM_THREADS=1`

### 2. Qt线程存储错误

**错误信息:**
```
QThreadStorage: entry X destroyed before end of thread
```

**原因:** Qt库的线程管理问题，通常与GUI相关组件有关。

**解决方案:**
- 设置Qt环境变量禁用调试输出
- 在程序结束时正确清理资源

### 3. pybind11 GIL错误

**错误信息:**
```
pybind11::handle::dec_ref() is being called while the GIL is either not held or invalid
```

**原因:** Python全局解释器锁(GIL)管理问题，通常发生在多线程环境或程序退出时。

**解决方案:**
- 确保在访问Python对象时持有GIL
- 正确清理环境资源
- 添加异常处理避免程序崩溃

## 自动修复

项目包含了自动修复脚本：

### 1. 环境修复脚本
```bash
python fix_environment.py
```

这个脚本会：
- 设置必要的环境变量
- 检查依赖是否正确安装
- 提供使用指导

### 2. 修改后的训练脚本

`train.py` 已经包含了以下修复：
- 在导入前设置环境变量
- 添加异常处理
- 正确的资源清理

### 3. 修改后的启动脚本

`run_training.py` 会自动：
- 调用环境修复函数
- 检查依赖
- 设置适当的环境变量

## 使用建议

### 推荐的运行方式

1. **使用修复后的启动脚本（推荐）:**
   ```bash
   python run_training.py [model_type] [episodes]
   ```

2. **手动运行环境修复:**
   ```bash
   python fix_environment.py
   python src/train.py --model rainbow --episodes 1000
   ```

### 环境要求

- Python 3.10
- PyTorch (CPU或CUDA版本)
- Gymnasium[atari]
- 其他依赖见 `environment.yml`

### Windows特定注意事项

## 输入屏蔽功能

**新功能**: 训练期间自动屏蔽键盘和鼠标输入，只允许 Ctrl+C 中断

**特性**:
- 自动屏蔽标准输入，防止意外输入干扰训练
- 保留 Ctrl+C 信号处理，可安全终止训练
- 跨平台支持（Windows/Linux/Mac）
- 训练开始时自动启动，结束时自动停止

**测试方法**:
```bash
# 测试输入屏蔽功能
python test_input_shield.py
```

## 使用建议

- 确保使用PowerShell或命令提示符
- 项目已移除DirectML支持，建议使用CUDA GPU或CPU进行训练
- 某些杀毒软件可能会干扰训练过程
- 训练期间避免在终端中输入内容，使用 Ctrl+C 安全终止

## 如果问题仍然存在

1. **检查依赖版本:**
   ```bash
   conda list
   ```

2. **重新创建环境:**
   ```bash
   conda env remove -n snake_rl
   conda env create -f environment.yml
   ```

3. **查看详细错误信息:**
   - 运行时添加 `--verbose` 参数
   - 检查 `logs/` 目录中的日志文件

4. **联系支持:**
   - 提供完整的错误信息
   - 包含系统信息（操作系统、Python版本等）
   - 描述重现步骤

## 性能优化建议

### 线程数优化

**问题**: 设置 `OMP_NUM_THREADS=1` 后性能下降严重

**解决方案**: 使用智能线程数设置

```bash
# 使用优化的线程数（推荐）
python fix_environment.py

# 使用自定义线程数
python fix_environment.py --threads 8

# 使用单线程（最安全但性能最低）
python fix_environment.py --single-thread

# 运行性能基准测试
python fix_environment.py --benchmark
```

**线程数选择策略**:
- **4核心及以下**: 使用全部核心
- **5-8核心**: 保留1个核心给系统
- **8核心以上**: 保留2个核心，最大16线程

### 其他性能优化

1. **使用GPU加速**（如果可用）
2. **调整批处理大小**
3. **优化网络架构**
4. **使用混合精度训练**
5. **合理设置OpenMP线程数**
- 调整批量大小和缓冲区大小
- 监控系统资源使用情况
- 定期保存模型检查点