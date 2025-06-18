import os
import torch
import numpy as np
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import signal  # 新增导入
# import torch_directml as dml
from model import DQN, DuelingDQN, RainbowDQN
from agent import DQNAgent, RainbowAgent
from utils import make_env, plot_rewards


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DQN训练脚本")
    parser.add_argument("--env", type=str,
                        default="ALE/Assault-v5", help="Gym环境名称")
    parser.add_argument("--model", type=str, default="rainbow",
                        choices=["dqn", "dueling", "rainbow"], help="模型类型")
    parser.add_argument("--episodes", type=int, default=100, help="训练回合数")
    parser.add_argument("--buffer_size", type=int,
                        default=100000, help="经验回放缓冲区大小")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批量大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epsilon_start", type=float,
                        default=1.0, help="初始探索率")
    parser.add_argument("--epsilon_final", type=float,
                        default=0.01, help="最终探索率")
    parser.add_argument("--epsilon_decay", type=int,
                        default=100000, help="探索率衰减帧数")
    parser.add_argument("--target_update", type=int,
                        default=1000, help="目标网络更新频率")
    parser.add_argument("--prioritized_replay",
                        action="store_true", help="是否使用优先经验回放")
    # Rainbow DQN 特有参数
    parser.add_argument("--n_step", type=int, default=3, help="N步学习的步数")
    parser.add_argument("--use_noisy", action="store_true", help="是否使用噪声网络")
    parser.add_argument("--use_distributional",
                        action="store_true", help="是否使用分布式Q学习")
    parser.add_argument("--n_atoms", type=int, default=51, help="分布式Q学习的原子数量")
    parser.add_argument("--v_min", type=float, default=-10, help="值函数的最小值")
    parser.add_argument("--v_max", type=float, default=10, help="值函数的最大值")
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志保存目录")
    parser.add_argument("--save_interval", type=int,
                        default=100, help="模型保存间隔（回合）")
    parser.add_argument("--eval_interval", type=int,
                        default=10, help="评估间隔（回合）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def train(args):
    """
    训练DQN智能体

    参数:
        args: 命令行参数
    """
    stop_training = False  # 控制训练循环的标志

    def signal_handler(sig, frame):
        nonlocal stop_training
        print("\n捕获到 Ctrl+C 信号，准备终止训练...")
        stop_training = True

    signal.signal(signal.SIGINT, signal_handler)  # 注册信号处理器

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 创建环境
    env = make_env(args.env)
    eval_env = make_env(args.env)

    # 设置设备
    # 设置设备
    device_str = "cpu"  # 默认值
    device = torch.device(device_str)

    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_built():
        device_str = "mps"
    else:
        try:
            # 尝试将张量移动到 "dml" 设备以检查其可用性
            # 这需要 torch-directml 包已安装
           # Initialize the DirectML device
            # device = dml.device(dml.default_device())
            print(f"Using DirectML device: {device}")
            print("Attempting to use DirectML (dml) device.")
        except RuntimeError:
            # "dml" 不可用
            print(
                "DirectML (dml) device not available (RuntimeError). Falling back to CPU.")
            print(
                "Ensure 'torch-directml' is installed and a compatible GPU + driver are present.")
        except Exception as e:
            # 捕获其他可能的导入或属性错误
            print(
                f"An error occurred while trying to use DirectML (dml) device: {e}. Falling back to CPU.")
            pass  # 保持 "cpu"

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
            use_distributional=args.use_distributional
        )
        target_model = RainbowDQN(
            input_shape=input_shape,
            n_actions=n_actions,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            use_noisy=args.use_noisy,
            use_distributional=args.use_distributional
        )

    # 创建智能体
    if args.model == "rainbow":
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
            n_step=args.n_step,
            use_noisy=args.use_noisy,
            use_distributional=args.use_distributional,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max
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
            prioritized_replay=args.prioritized_replay
        )

    # 创建TensorBoard日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)

    # 训练统计
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')

    # 训练循环
    total_steps = 0
    start_time = time.time()

    print("开始训练。按 Ctrl+C 终止训练。")
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

        if stop_training and not (done or truncated):  # 如果是因为Ctrl+C跳出内部循环
            break  # 也跳出外部回合循环

        # 记录回合统计
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:])  # 最近100回合的平均奖励
        avg_rewards.append(avg_reward)

        # 记录到TensorBoard
        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar("Train/AvgReward", avg_reward, episode)
        writer.add_scalar("Train/Loss", episode_loss /
                          episode_steps if episode_steps > 0 else 0, episode)
        writer.add_scalar("Train/Epsilon", agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) *
                          np.exp(-1. * agent.steps_done / agent.epsilon_decay), episode)

        # 打印训练信息
        if episode % 10 == 0:
            print(f"Episode {episode}/{args.episodes} | Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f} | "
                  f"Loss: {episode_loss/episode_steps if episode_steps > 0 else 0:.6f} | "
                  f"Time: {(time.time() - start_time)/60:.2f} min")

        # 评估智能体
        if episode % args.eval_interval == 0 and not stop_training:
            eval_reward = evaluate(agent, eval_env, device)
            writer.add_scalar("Eval/Reward", eval_reward, episode)
            print(f"Evaluation at episode {episode}: {eval_reward:.2f}")

        # 保存最佳模型
        if avg_reward > best_avg_reward and not stop_training:
            best_avg_reward = avg_reward
            agent.save_model(os.path.join(
                args.save_dir, f"best_model_{args.model}.pth"))
            print(f"在第 {episode} 回合保存了新的最佳模型，平均奖励: {best_avg_reward:.2f}")

        # 定期保存模型
        if episode % args.save_interval == 0 and not stop_training:
            agent.save_model(os.path.join(
                args.save_dir, f"{args.model}_episode_{episode}.pth"))
            print(f"在第 {episode} 回合保存了模型。")

    if not stop_training:
        print("训练完成。")
        # 保存最终模型
        agent.save_model(os.path.join(
            args.save_dir, f"final_model_{args.model}.pth"))
        print("最终模型已保存。")
    else:
        print("训练被用户提前终止。")
        # 考虑是否在终止时也保存当前模型
        # agent.save_model(os.path.join(args.save_dir, f"interrupted_model_{args.model}_episode_{episode}.pth"))
        # print(f"已保存中断时的模型到 episode {episode}。")

    # 绘制奖励曲线 (即使中断也绘制已有的数据)
    if rewards:  # 确保有数据可绘制
        plot_rewards(rewards, avg_rewards, title=f"{args.model.upper()} Training on {args.env} (Interrupted: {stop_training})",
                     save_path=os.path.join(args.log_dir, f"{args.model}_rewards_{'interrupted' if stop_training else 'final'}.png"))
        print("奖励曲线已绘制并保存。")

    # 关闭TensorBoard写入器
    writer.close()
    print("TensorBoard写入器已关闭。")

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
