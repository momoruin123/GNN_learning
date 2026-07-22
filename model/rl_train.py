"""
RL + GNN + 脉冲选择 训练骨架(概念版,不依赖实际 GNN 库)
核心思想:
  1. 网络对每个 UAV→任务 打出"意愿分值"
  2. mask 掉不可行的(时间窗超、能量不够、已完成)
  3. Top-K 脉冲化(只保留 K 个最强信号)
  4. 环境反馈奖励 = -makespan
  5. 反复试炼,网络学会避开低奖励的路径
"""

import random
from math import hypot


# =====================================================================
# 环境:构造一个算例,接收动作,返回奖励
# =====================================================================
class Env:
    def __init__(self, inst):
        self.inst = inst
        self.reset()

    def reset(self):
        self.done_tasks = set()          # 已完成的任务 id
        self.routes = {u.id: [] for u in self.inst.uavs}
        self.uav_time = {u.id: 0.0 for u in self.inst.uavs}    # 每架 UAV 当前时间
        self.uav_energy = {u.id: u.e_full for u in self.inst.uavs}
        self.uav_pos = {u.id: (u.x, u.y) for u in self.inst.uavs}
        self.task_s = {}                  # 任务实际开始时间

    def step(self, uav_id, task_id):
        """一架 UAV 选择执行一个任务:返回 (reward, done, info)"""
        inst = self.inst
        t = [t for t in inst.tasks if t.id == task_id][0]
        u = [u for u in inst.uavs if u.id == uav_id][0]

        # 飞到任务的耗时
        px, py = self.uav_pos[uav_id]
        fly_t = hypot(t.x - px, t.y - py) / u.speed
        arr_t = self.uav_time[uav_id] + fly_t

        # 时间窗检查
        if arr_t + t.dur > t.l:
            return -100.0, True, {'fail': '时间窗超'}

        # 协同:所有参与者到齐的时间
        self.task_s.setdefault(task_id, arr_t)
        if arr_t > self.task_s[task_id]:
            self.task_s[task_id] = arr_t

        # 推进时间
        task_start = self.task_s[task_id]
        self.uav_time[uav_id] = task_start + t.dur
        self.uav_pos[uav_id] = (t.x, t.y)

        # 能量
        energy_used = fly_t + t.task_cost
        self.uav_energy[uav_id] -= energy_used
        if self.uav_energy[uav_id] < u.e_min:
            return -100.0, True, {'fail': '能量耗尽'}

        # 记录
        self.routes[uav_id].append(task_id)
        self.done_tasks.add(task_id)

        # 所有任务完成?
        if len(self.done_tasks) >= len(inst.tasks):
            makespan = max(self.uav_time.values())
            return -makespan, True, {'makespan': makespan}

        return 0.0, False, {}


# =====================================================================
# 动作 mask:过滤不可行选择
# =====================================================================
def get_action_mask(env, uav_id):
    """返回一个 bool 数组:哪些任务可以选择"""
    inst = env.inst
    u = [u for u in inst.uavs if u.id == uav_id][0]
    mask = []
    for t in inst.tasks:
        # 已完成 → 不可选
        if t.id in env.done_tasks:
            mask.append(0)
            continue
        # 时间窗 → 算到达后是否超 l
        px, py = env.uav_pos[uav_id]
        fly_t = hypot(t.x - px, t.y - py) / u.speed
        if env.uav_time[uav_id] + fly_t + t.dur > t.l:
            mask.append(0)
            continue
        # 能量 → 飞过去+干活后是否低于下限
        if env.uav_energy[uav_id] - fly_t - t.task_cost < u.e_min:
            mask.append(0)
            continue
        # 协同需求:如果还需 UGV,忽略(简化版)
        mask.append(1)
    return mask


# =====================================================================
# 策略:选分值最高的 K 个,最终用第一列(简化版脉冲 Top-K)
# =====================================================================
def select_action_pulse(scores, mask, K=3):
    """
    scores: [n_task] 网络输出的分值,(模拟脉冲,未真正网络)
    mask:   [n_task] 0/1 可选的
    K:      最多可选 K 个(脉冲上限)
    """
    valid = [(i, s) for i, (s, m) in enumerate(zip(scores, mask)) if m == 1]
    if not valid:
        return None
    # 排序取 Top-K
    valid.sort(key=lambda x: -x[1])
    # 只激活前 K 个,这里简化返回最强的 1 个
    return valid[0][0]


# =====================================================================
# 训练循环(概念版,用随机探索 + ε-greedy)
# =====================================================================
def train_concept(inst, episodes=500):
    env = Env(inst)
    best_makespan = float('inf')
    best_routes = None

    for ep in range(episodes):
        env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # 选一架还有能力干活的 UAV
            candidates = []
            for u in inst.uavs:
                mask = get_action_mask(env, u.id)
                if any(mask):
                    candidates.append((u.id, mask))
            if not candidates:
                break

            # ε-greedy:探索 vs 利用
            uav_id, mask = random.choice(candidates)
            if random.random() < 0.3:   # 30% 随机探索
                valid = [i for i, m in enumerate(mask) if m]
                task_id = random.choice(valid)
            else:
                # 用伪评分 = 距离的倒数(离得近分值高)
                scores = [1.0 / (1 + hypot(
                    inst.tasks[i].x - env.uav_pos[uav_id][0],
                    inst.tasks[i].y - env.uav_pos[uav_id][1]
                )) if mask[i] else 0.0 for i in range(len(inst.tasks))]
                task_id = select_action_pulse(scores, mask, K=3)

            if task_id is None:
                continue

            reward, done, info = env.step(uav_id, task_id)
            total_reward += reward

        if total_reward < 0 and abs(total_reward) < best_makespan:
            best_makespan = abs(total_reward)
            best_routes = {k: list(v) for k, v in env.routes.items()}

        if ep % 50 == 0:
            print(f"  轮次 {ep:4d}: makespan={abs(total_reward):.1f}  最佳={best_makespan:.1f}")

    return best_makespan, best_routes


if __name__ == "__main__":
    from model_skeleton import make_random_instance
    inst = make_random_instance(n_uav=3, n_ugv=0, n_task=8, seed=1)
    best_ms, best_routes = train_concept(inst, episodes=500)
    print(f"\n最佳 makespan = {best_ms:.1f}")
    for uid, seq in best_routes.items():
        print(f"  UAV{uid}: {seq}")
