"""
共享数据层:UAV/UGV/Charger/Task 实体 + 算例生成 + 预计算表。
所有模型变体统一从此导入,不再各自复制 model_skeleton。
"""

from dataclasses import dataclass, field
from math import hypot
import random


# =====================================================================
# 实体类
# =====================================================================
@dataclass
class UAV:
    id: int
    x: float
    y: float
    speed: float
    e_full: float = 100.0
    e_min: float = 10.0
    hover_cost: float = 1.0


@dataclass
class UGV:
    id: int
    x: float
    y: float
    speed: float
    charge_rate: float = 5.0


@dataclass
class Charger:
    """充电车:只充电,不执行任务"""
    id: int
    x: float
    y: float
    speed: float
    charge_rate: float = 8.0
    charge_time: float = 2.0


@dataclass
class Task:
    id: int
    x: float
    y: float
    dur: float
    e: float = 0.0
    l: float = 1e6
    need_uav: int = 1
    need_ugv: int = 0
    task_cost: float = 5.0
    preds: list = field(default_factory=list)


@dataclass
class Instance:
    uavs: list = field(default_factory=list)
    ugvs: list = field(default_factory=list)
    chargers: list = field(default_factory=list)
    tasks: list = field(default_factory=list)

    def dist(self, a, b):
        return hypot(a.x - b.x, a.y - b.y)

    def build_travel_table(self):
        """返回 (uav_travel, ugv_travel, charger_travel) 三张表"""
        def _tbl(agents):
            tbl = {}
            for a in agents:
                for b in self.tasks:
                    tbl[(a.id, -1, b.id)] = self.dist(a, b) / a.speed
                    tbl[(a.id, b.id, -2)] = self.dist(b, a) / a.speed
                for t1 in self.tasks:
                    for t2 in self.tasks:
                        if t1.id != t2.id:
                            tbl[(a.id, t1.id, t2.id)] = self.dist(t1, t2) / a.speed
            return tbl
        return _tbl(self.uavs), _tbl(self.ugvs), _tbl(self.chargers)


# =====================================================================
# 算例工厂
# =====================================================================
def make_small_instance(n_charger=0):
    """小算例:4任务,2UAV+1UGV(+可选充电车)"""
    inst = Instance()
    inst.uavs = [
        UAV(id=0, x=0,  y=0,  speed=2.0),
        UAV(id=1, x=10, y=0,  speed=1.5),
    ]
    inst.ugvs = [
        UGV(id=0, x=5, y=5, speed=1.0),
    ]
    if n_charger > 0:
        inst.chargers = [
            Charger(id=0, x=3, y=3, speed=0.8, charge_rate=10.0),
        ]
        inst.uavs = [
            UAV(id=0, x=0,  y=0,  speed=2.0, e_full=20.0, e_min=0.0),
            UAV(id=1, x=10, y=0,  speed=1.5, e_full=20.0, e_min=0.0),
        ]
        tc = 8.0
    else:
        tc = 5.0
    inst.tasks = [
        Task(id=0, x=3, y=4, dur=5, need_uav=1, need_ugv=1, task_cost=tc),
        Task(id=1, x=8, y=2, dur=3, need_uav=2, preds=[0], task_cost=tc),
        Task(id=2, x=6, y=9, dur=4, need_uav=1, preds=[1], task_cost=tc),
        Task(id=3, x=1, y=1, dur=2, need_uav=1, task_cost=tc),
    ]
    return inst


def make_random_instance(n_uav=3, n_ugv=1, n_charger=0, n_task=8,
                         area=100.0, horizon=200.0, seed=0):
    rng = random.Random(seed)
    inst = Instance()
    e_full = area * n_task * 2.0
    inst.uavs = [
        UAV(id=i, x=0.0, y=0.0, speed=rng.uniform(1.5, 2.5),
            e_full=e_full, e_min=0.0)
        for i in range(n_uav)
    ]
    inst.ugvs = [
        UGV(id=g, x=area / 2, y=area / 2, speed=rng.uniform(0.8, 1.2))
        for g in range(n_ugv)
    ]
    inst.chargers = [
        Charger(id=c, x=rng.uniform(0, area), y=rng.uniform(0, area),
                speed=rng.uniform(0.6, 1.0), charge_rate=rng.uniform(5, 15))
        for c in range(n_charger)
    ]
    tasks = []
    for j in range(n_task):
        e = rng.uniform(0, horizon * 0.5)
        dur = rng.uniform(3, 8)
        l = e + rng.uniform(dur + 20, horizon)
        need_uav = rng.choice([1, 1, 1, 2])
        need_ugv = rng.choice([0, 0, 1])
        tasks.append(Task(
            id=j, x=rng.uniform(0, area), y=rng.uniform(0, area),
            dur=dur, e=e, l=l,
            need_uav=need_uav, need_ugv=need_ugv,
        ))
    inst.tasks = tasks
    return inst
