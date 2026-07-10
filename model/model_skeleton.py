"""
UAV/UGV 任务调度建模 —— 骨架脚本
先教你 dataclass 怎么定义实体,以及怎么预计算距离/时间表。
这个文件不依赖 gurobi,直接就能跑,先看数据结构。
"""

from dataclasses import dataclass, field
from math import hypot

@dataclass
class UAV:
    id: int
    x: float                 # 起始位置 x
    y: float                 # 起始位置 y
    speed: float             # 速度(固定)
    e_full: float = 100.0    # 满电量
    e_min: float = 10.0      # 安全电量下限
    hover_cost: float = 1.0  # 悬停等待每单位时间耗电


@dataclass
class UGV:
    id: int
    x: float
    y: float
    speed: float             
    charge_rate: float = 5.0 # 给 UAV 充电的速率


@dataclass
class Task:
    id: int
    x: float
    y: float
    dur: float               # 任务耗时
    e: float                 # 时间窗最早开始
    l: float                 # 时间窗最晚结束
    need_uav: int = 1        # 需要几架 UAV (c=1, b=2)
    need_ugv: int = 0        # 需要几台 UGV (a=1)
    task_cost: float = 5.0   # 执行该任务的耗电

@dataclass
class Instance:
    uavs: list = field(default_factory=list)
    ugvs: list = field(default_factory=list)
    tasks: list = field(default_factory=list)

    def dist(self, a, b):
        """任意两个带 x,y 的任务之间的欧氏距离"""
        return hypot(a.x - b.x, a.y - b.y)

    def build_travel_table(self):
        """
        预计算:每架 UAV, UGV从每个任务飞到每个任务要多久。
        return:
            dict,key=(uav_id, from_task_id, to_task_id) -> 时间
            dict,key=(ugv_id, from_task_id, to_task_id) -> 时间
        """
        uav_travel = {}
        for u in self.uavs:
            for a in self.tasks:
                for b in self.tasks:
                    if a.id != b.id:
                        uav_travel[(u.id, a.id, b.id)] = self.dist(a, b) / u.speed

        ugv_travel = {}
        for g in self.ugvs:
            for a in self.tasks:
                for b in self.tasks:
                    if a.id != b.id:
                        ugv_travel[(g.id, a.id, b.id)] = self.dist(a, b) / g.speed

        return uav_travel, ugv_travel

def make_small_instance():
    inst = Instance()

    # 2 架 UAV,速度不同
    inst.uavs = [
        UAV(id=0, x=0,  y=0,  speed=2.0),
        UAV(id=1, x=10, y=0,  speed=1.5),
    ]
    # 1 台 UGV(慢)
    inst.ugvs = [
        UGV(id=0, x=5, y=5, speed=1.0),
    ]
    # 三个任务:a 要 UAV+UGV, b 要 2 UAV, c 单独 1 UAV
    inst.tasks = [
        Task(id=0, x=3, y=4, dur=5, e=0,  l=50, need_uav=1, need_ugv=1),  # a
        Task(id=1, x=8, y=2, dur=3, e=0,  l=50, need_uav=2, need_ugv=0),  # b
        Task(id=2, x=6, y=9, dur=4, e=10, l=60, need_uav=1, need_ugv=0),  # c
    ]
    return inst


def main():
    inst = make_small_instance()

    print("=== 实体一览 ===")
    for u in inst.uavs:
        print(u)          
    for g in inst.ugvs:
        print(g)
    for t in inst.tasks:
        print(t)

    print("\n=== 预计算的转场时间表 (uav, from, to) -> time ===")
    travel = inst.build_travel_table()
    for key, val in travel.items():
        print(f"  UAV{key[0]}: 任务{key[1]} -> 任务{key[2]}  用时 {val:.2f}")

    # 单独取某个字段也很方便
    print("\n=== 访问字段示例 ===")
    a = inst.tasks[0]
    print(f"任务a: 位置=({a.x},{a.y}), 时间窗=[{a.e},{a.l}], 需要 {a.need_uav}UAV+{a.need_ugv}UGV")


if __name__ == "__main__":
    main()
