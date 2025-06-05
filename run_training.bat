@echo off
:: 强化学习训练启动脚本
:: 使用方法: run_training.bat [模型类型] [回合数]

setlocal

:: 默认参数
if "%1"=="" (set MODEL_TYPE=dqn) else (set MODEL_TYPE=%1)
if "%2"=="" (set EPISODES=1000) else (set EPISODES=%2)

echo ======================================
echo 强化学习 Atari 游戏训练
echo ======================================
echo 模型类型: %MODEL_TYPE%
echo 训练回合: %EPISODES%
echo 游戏环境: ALE/Assault-v5
echo ======================================

:: 检查 conda 环境
call conda info --envs | findstr /C:"snake_rl" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到 snake_rl conda 环境
    echo 请先运行: conda env create -f environment.yml
    exit /b 1
)

:: 激活环境
echo 激活 conda 环境...
call conda activate snake_rl

:: 创建必要的目录
echo 创建输出目录...
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "videos" mkdir videos

:: 开始训练
echo 开始训练...
python src\train.py ^
    --model %MODEL_TYPE% ^
    --episodes %EPISODES% ^
    --save_dir checkpoints ^
    --log_dir logs ^
    --save_interval 100 ^
    --eval_interval 50

echo ======================================
echo 训练完成！
echo 模型保存在: checkpoints/
echo 日志保存在: logs/
echo ======================================
echo 使用以下命令评估模型:
echo python src\evaluate.py --model_path checkpoints\best_model_%MODEL_TYPE%.pth
echo ======================================

endlocal