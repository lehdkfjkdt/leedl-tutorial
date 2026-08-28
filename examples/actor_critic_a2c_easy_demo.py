"""
Actor-Critic / A2C 极简教学 Demo
=================================

这个 demo 的目标不是追求性能，而是让你一眼看懂：

1. Actor 到底是什么
2. Critic 到底是什么
3. TD error / Advantage 是怎么同时更新两者的

环境非常简单：

    位置: 0 -> 1 -> 2 -> 3 -> 4(终点)

- 初始在 0
- 动作只有两个：left / right
- 到达终点奖励 +1
- 每走一步奖励 -0.02（鼓励尽快到达）

我们故意不用 PyTorch、NumPy，只用 Python 标准库，
让公式和更新过程尽量透明。
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    num_states: int = 5
    goal_state: int = 4
    gamma: float = 0.95
    actor_lr: float = 0.12
    critic_lr: float = 0.20
    step_penalty: float = -0.02
    goal_reward: float = 1.0
    max_steps_per_episode: int = 20
    train_episodes: int = 250
    seed: int = 7


class LineWorld:
    """一个最小的一维导航环境。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        action = 0 表示向左
        action = 1 表示向右
        """
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.config.goal_state, self.state + 1)

        done = self.state == self.config.goal_state
        reward = self.config.goal_reward if done else self.config.step_penalty
        return self.state, reward, done


class ActorCriticAgent:
    """
    教学版 Actor-Critic。

    - Actor: 每个状态存两个动作的 logits
    - Critic: 每个状态存一个 V(s)

    logits 经过 softmax 变成动作概率。
    """

    ACTIONS = ["left", "right"]

    def __init__(self, config: Config) -> None:
        self.config = config

        # Actor 参数：每个状态两维 logits，分别对应 left / right
        self.actor_logits: List[List[float]] = [
            [0.0, 0.0] for _ in range(config.num_states)
        ]

        # Critic 参数：每个状态一个价值估计 V(s)
        self.critic_values: List[float] = [0.0 for _ in range(config.num_states)]

    def _softmax(self, logits: List[float]) -> List[float]:
        max_logit = max(logits)
        exp_values = [math.exp(value - max_logit) for value in logits]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def get_action_probs(self, state: int) -> List[float]:
        return self._softmax(self.actor_logits[state])

    def sample_action(self, state: int) -> int:
        probs = self.get_action_probs(state)
        threshold = random.random()
        return 0 if threshold < probs[0] else 1

    def greedy_action(self, state: int) -> int:
        probs = self.get_action_probs(state)
        return 0 if probs[0] > probs[1] else 1

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        核心更新：

        1. 先算 TD error
        2. Critic 用 TD error 更新 V(s)
        3. Actor 用 TD error 更新 log π(a|s)

        这里的 TD error 也被直接当成 advantage 使用，
        所以这是一个最小可运行的 A2C 风格实现。
        """
        current_value = self.critic_values[state]
        next_value = 0.0 if done else self.critic_values[next_state]

        td_error = reward + self.config.gamma * next_value - current_value

        # ---------- Critic 更新 ----------
        self.critic_values[state] += self.config.critic_lr * td_error

        # ---------- Actor 更新 ----------
        probs = self.get_action_probs(state)

        # softmax + log policy 的梯度：
        # 对被选中动作: 1 - pi(a|s)
        # 对其他动作:   -pi(a'|s)
        for action_index in range(len(self.ACTIONS)):
            indicator = 1.0 if action_index == action else 0.0
            grad_log_pi = indicator - probs[action_index]
            self.actor_logits[state][action_index] += (
                self.config.actor_lr * td_error * grad_log_pi
            )

        return td_error


def rollout_episode(env: LineWorld, agent: ActorCriticAgent, greedy: bool) -> Tuple[float, List[Tuple[int, str, float]]]:
    state = env.reset()
    total_reward = 0.0
    trajectory: List[Tuple[int, str, float]] = []

    for _ in range(env.config.max_steps_per_episode):
        action = agent.greedy_action(state) if greedy else agent.sample_action(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, agent.ACTIONS[action], reward))
        total_reward += reward
        state = next_state
        if done:
            break

    return total_reward, trajectory


def print_policy_and_values(agent: ActorCriticAgent) -> None:
    print("\n当前策略和价值估计")
    print("状态 | P(left) | P(right) | V(s) | 贪心动作")
    print("-" * 52)
    for state in range(agent.config.num_states):
        probs = agent.get_action_probs(state)
        greedy_action = agent.ACTIONS[agent.greedy_action(state)]
        print(
            f"{state:>4} | {probs[0]:>7.3f} | {probs[1]:>8.3f} |"
            f" {agent.critic_values[state]:>4.3f} | {greedy_action}"
        )


def train_demo() -> None:
    config = Config()
    random.seed(config.seed)

    env = LineWorld(config)
    agent = ActorCriticAgent(config)

    print("=" * 72)
    print("Actor-Critic / A2C 极简教学 Demo")
    print("=" * 72)
    print("环境: 0 -> 1 -> 2 -> 3 -> 4(终点)")
    print("动作: left / right")
    print("奖励: 到终点 +1, 其他每步 -0.02")

    print("\n训练前，先看看策略：")
    print_policy_and_values(agent)

    print("\n训练开始...\n")

    for episode in range(1, config.train_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        last_td_error = 0.0

        for _ in range(config.max_steps_per_episode):
            action = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            last_td_error = agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if done:
                break

        if episode <= 5 or episode % 50 == 0:
            print(
                f"Episode {episode:>3}: reward={episode_reward:>6.3f},"
                f" last_td_error={last_td_error:>7.4f}"
            )

    print("\n训练结束。")
    print_policy_and_values(agent)

    print("\n用贪心策略跑 3 次，看看它学会了没有：")
    for test_id in range(1, 4):
        total_reward, trajectory = rollout_episode(env, agent, greedy=True)
        readable = " -> ".join(
            f"(s={state}, a={action}, r={reward:+.2f})"
            for state, action, reward in trajectory
        )
        print(f"测试 {test_id}: total_reward={total_reward:.3f}")
        print(f"  {readable}")

    print("\n你真正该观察的点：")
    print("1. Critic 的 V(s) 会在越靠近终点的状态越大。")
    print("2. Actor 在每个状态下的 P(right) 会越来越高。")
    print("3. TD error > 0 时，当前动作概率会被推高；TD error < 0 时会被压低。")
    print("4. 这就是 Actor-Critic 的核心：Critic 打分，Actor 改动作。")


if __name__ == "__main__":
    train_demo()