import os

# 设置环境变量解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置Qt相关环境变量
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import torch
import numpy as np
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import signal  # 新增导入
from model import DQN, DuelingDQN, RainbowDQN
from agent import DQNAgent, RainbowAgent
from utils import make_env, plot_rewards
from input_shield import input_shield_context, setup_signal_handlers


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DQN训练脚本")
    parser.add_argument("--env", type=str, default="ALE/Assault-v5", help="Gym环境名称")
    parser.add_argument(
        "--model",
        type=str,
        default="rainbow",
        choices=["dqn", "dueling", "rainbow"],
        help="模型类型",
    )
    parser.add_argument("--episodes", type=int, default=100, help="训练回合数")
    parser.add_argument(
        "--buffer_size", type=int, default=100000, help="经验回放缓冲区大小"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="训练批量大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon_final", type=float, default=0.01, help="最终探索率")
    parser.add_argument(
        "--epsilon_decay", type=int, default=100000, help="探索率衰减帧数"
    )
    parser.add_argument(
        "--target_update", type=int, default=1000, help="目标网络更新频率"
    )
    parser.add_argument(
        "--prioritized_replay", action="store_true", help="是否使用优先经验回放"
    )
    # Rainbow DQN 特有参数
    # parser.add_argument("--n_step", type=int, default=3, help="N步学习的步数") # Replaced by base_n_step
    parser.add_argument("--use_noisy", action="store_true", help="是否使用噪声网络")
    parser.add_argument(
        "--use_distributional", action="store_true", help="是否使用分布式Q学习"
    )
    parser.add_argument("--n_atoms", type=int, default=51, help="分布式Q学习的原子数量")
    parser.add_argument("--v_min", type=float, default=-10, help="值函数的最小值")
    parser.add_argument("--v_max", type=float, default=10, help="值函数的最大值")

    # Adaptive N-step parameters
    parser.add_argument("--base_n_step", type=int, default=3, help="基础N步学习的步数 (用于AdaptiveNStepBuffer)")
    parser.add_argument("--max_n_step", type=int, default=10, help="最大N步学习的步数 (用于AdaptiveNStepBuffer)")
    parser.add_argument("--adapt_n_step_freq", type=int, default=1000, help="自适应N步调整频率 (训练步数)")
    parser.add_argument("--td_error_threshold_low", type=float, default=0.1, help="用于降低N的TD误差阈值")
    parser.add_argument("--td_error_threshold_high", type=float, default=0.5, help="用于增加N的TD误差阈值")

    # Experience Augmentation parameters
    parser.add_argument("--use_state_augmentation", action="store_true", help="是否启用状态增强")
    parser.add_argument("--aug_noise_scale", type=float, default=5.0, help="高斯噪声增强的标准差 (基于0-255的像素值)") # Adjusted default based on uint8

    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="模型保存目录"
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="日志保存目录")
    parser.add_argument(
        "--save_interval", type=int, default=100, help="模型保存间隔（回合）"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=10, help="评估间隔（回合）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def train(args):
    """
    训练DQN智能体

    参数:
        args: 命令行参数
    """
    stop_training = False  # 控制训练循环的标志

    def stop_callback():
        nonlocal stop_training
        print("\n准备终止训练...")
        stop_training = True

    # 设置信号处理器，只允许Ctrl+C中断
    setup_signal_handlers(stop_callback)

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 创建环境
    try:
        env = make_env(args.env)
        eval_env = make_env(args.env)
    except Exception as e:
        print(f"创建环境时出错: {e}")
        print("尝试使用备用环境设置...")
        # 可以在这里添加备用环境创建逻辑
        raise

    # 设置设备
    device_str = "cpu"  # 默认值

    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_built():
        device_str = "mps"

    device = torch.device(device_str)
    print(f"Using device: {device}")
    print(f"Choose model: {args.model}")

    # 创建模型
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    if args.model == "dqn":
        model = DQN(input_shape, n_actions)
        target_model = DQN(input_shape, n_actions)
    elif args.model == "dueling":
        model = DuelingDQN(input_shape, n_actions)
        target_model = DuelingDQN(input_shape, n_actions)
    else:  # rainbow
        model = RainbowDQN(
            input_shape=input_shape,
            n_actions=n_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            use_noisy=args.use_noisy,
            use_distributional=args.use_distributional,
        )
        target_model = RainbowDQN(
            input_shape=input_shape,
            n_actions=n_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            use_noisy=args.use_noisy,
            use_distributional=args.use_distributional,
        )

    # 创建智能体
    if args.model == "rainbow":
        augmentation_config = None
        if args.use_state_augmentation:
            augmentation_config = {'add_noise': {'scale': args.aug_noise_scale}}
            print(f"Augmentation enabled with config: {augmentation_config}")

        agent = RainbowAgent(
            model=model,
            target_model=target_model,
            env=env,
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
            target_update=args.target_update,
            prioritized_replay=args.prioritized_replay,
            # n_step=args.n_step, # Replaced by base_n_step for RainbowAgent with AdaptiveNStepBuffer
            base_n_step=args.base_n_step,
            max_n_step=args.max_n_step,
            adapt_n_step_freq=args.adapt_n_step_freq,
            td_error_threshold_low=args.td_error_threshold_low,
            td_error_threshold_high=args.td_error_threshold_high,
            use_noisy=args.use_noisy,
            use_distributional=args.use_distributional,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            augmentation_config=augmentation_config, # Pass augmentation config
        )
    else:
        agent = DQNAgent(
            model=model,
            target_model=target_model,
            env=env,
            device=device,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            epsilon_decay=args.epsilon_decay,
            target_update=args.target_update,
            prioritized_replay=args.prioritized_replay,
        )

    # 创建TensorBoard日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)

    # 训练统计
    rewards = []
    avg_rewards = []
    best_avg_reward = -float("inf")

    # 训练循环
    total_steps = 0
    start_time = time.time()

    print("开始训练。按 Ctrl+C 终止训练。")

    # 使用输入屏蔽上下文，只允许Ctrl+C中断
    with input_shield_context():
        for episode in range(1, args.episodes + 1):
            if stop_training:
                print(f"在第 {episode} 回合开始前终止训练。")
                break

            state, _ = env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0

            done = False
            truncated = False

            # 单回合循环
            while not (done or truncated):
                if stop_training:
                    print(f"在第 {episode} 回合中途终止训练。")
                    break

                # 选择动作
                action = agent.select_action(state)

                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)

                # 存储经验
                if isinstance(agent, RainbowAgent):
                    agent.store_experience(state, action, reward, next_state, done)
                else:
                    agent.memory.push(state, action, reward, next_state, done)

                # 更新模型
                loss = agent.update_model()
                if loss is not None:
                    episode_loss += loss

                # 更新目标网络
                if total_steps % args.target_update == 0:
                    agent.update_target_model()

                # 更新状态和统计
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

            if stop_training and not (
                done or truncated
            ):  # 如果是因为Ctrl+C跳出内部循环
                break  # 也跳出外部回合循环

            # 记录回合统计
            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-100:])  # 最近100回合的平均奖励
            avg_rewards.append(avg_reward)

            # 记录到TensorBoard
            writer.add_scalar("Train/Reward", episode_reward, episode)
            writer.add_scalar("Train/AvgReward", avg_reward, episode)
            writer.add_scalar(
                "Train/Loss",
                episode_loss / episode_steps if episode_steps > 0 else 0,
                episode,
            )
            # 计算当前epsilon值（仅对DQN有效，Rainbow使用噪声网络）
            if hasattr(agent, "epsilon_start"):
                current_epsilon = agent.epsilon_final + (
                    agent.epsilon_start - agent.epsilon_final
                ) * np.exp(-1.0 * agent.steps_done / agent.epsilon_decay)
                writer.add_scalar("Train/Epsilon", current_epsilon, episode)
            else:
                # Rainbow使用噪声网络，epsilon固定为0
                writer.add_scalar("Train/Epsilon", 0.0, episode)

            # 打印进度
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                # 计算当前探索率显示
                if hasattr(agent, 'epsilon_start'):
                    current_epsilon = agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) * \
                                    np.exp(-1. * agent.steps_done / agent.epsilon_decay)
                    epsilon_str = f"探索率: {current_epsilon:.3f}, "
                else:
                    epsilon_str = "探索率: 0.000 (噪声网络), "
                
                print(
                    f"回合 {episode}/{args.episodes}, "
                    f"奖励: {episode_reward:.2f}, "
                    f"平均奖励: {avg_reward:.2f}, "
                    f"{epsilon_str}"
                    f"用时: {elapsed_time:.1f}s"
                )

            # 评估模型
            if episode % args.eval_interval == 0:
                eval_reward = evaluate(agent, eval_env, device)
                writer.add_scalar("Eval/Reward", eval_reward, episode)
                print(f"评估奖励: {eval_reward:.2f}")

                # 保存最佳模型
                if eval_reward > best_avg_reward:
                    best_avg_reward = eval_reward
                    agent.save_model(
                        os.path.join(args.save_dir, f"best_model_{args.model}.pth")
                    )
                    print(f"新的最佳模型已保存，评估奖励: {eval_reward:.2f}")

            # 定期保存模型
            if episode % args.save_interval == 0:
                agent.save_model(
                    os.path.join(args.save_dir, f"{args.model}_episode_{episode}.pth")
                )
                print(f"在第 {episode} 回合保存了模型。")

        if not stop_training:
            print("训练完成。")
            # 保存最终模型
            agent.save_model(
                os.path.join(args.save_dir, f"final_model_{args.model}.pth")
            )
            print("最终模型已保存。")
        else:
            print("训练被用户提前终止。")
            # 考虑是否在终止时也保存当前模型
            # agent.save_model(os.path.join(args.save_dir, f"interrupted_model_{args.model}_episode_{episode}.pth"))
            # print(f"已保存中断时的模型到 episode {episode}。")

        # 绘制奖励曲线 (即使中断也绘制已有的数据)
        if rewards:  # 确保有数据可绘制
            plot_rewards(
                rewards,
                avg_rewards,
                title=f"{args.model.upper()} Training on {args.env} (Interrupted: {stop_training})",
                save_path=os.path.join(
                    args.log_dir,
                    f"{args.model}_rewards_{'interrupted' if stop_training else 'final'}.png",
                ),
            )
            print("奖励曲线已绘制并保存。")

        # 关闭TensorBoard写入器
        writer.close()
        print("TensorBoard写入器已关闭。")

        # 清理环境资源，避免pybind11 GIL问题
        try:
            if hasattr(env, "close"):
                env.close()
            if hasattr(eval_env, "close"):
                eval_env.close()
            print("环境资源已清理。")
        except Exception as e:
            print(f"清理环境时出现警告: {e}")
            # 不抛出异常，因为这只是清理操作

    return rewards, avg_rewards


def evaluate(agent, env, device, n_episodes=5, max_steps=10000):
    """
    评估智能体性能

    参数:
        agent: DQN智能体
        env: 游戏环境
        device: 计算设备
        n_episodes: 评估回合数
        max_steps: 每回合最大步数

    返回:
        平均奖励
    """
    total_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # 选择动作 (评估模式，不使用探索)
            action = agent.select_action(state, evaluate=True)

            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)

            # 更新状态和奖励
            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        total_rewards.append(episode_reward)

    # 返回平均奖励
    return np.mean(total_rewards)


if __name__ == "__main__":
    args = parse_args()
    train(args)
