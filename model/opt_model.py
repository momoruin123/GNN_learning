from model_skeleton import make_small_instance

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None  # 本地没装也能 import 这个文件,方便看结构


class OptModel:
    def __init__(self, inst):
        self.ins = inst
        self.m = gp.Model("uav_ugv_sched")

        # 预计算表(数据层算好,这里查)
        self.travel, self.ugv_travel = inst.build_travel_table()   # 两张表:UAV / UGV

        # 节点集合:任务 id 列表 + 两个虚拟点
        self.task_ids = [t.id for t in inst.tasks]
        self.START = -1     # 虚拟起点
        self.END = -2       # 虚拟终点
        self.nodes = self.task_ids + [self.START, self.END]

        # big-M:时间上界。取所有任务最晚时间窗 + 余量。宁可算紧一点
        self.BIG_T = max(t.l for t in inst.tasks) + 100.0

        # 变量容器(统一放这里,key=元组,不塞进实体类)
        self.x = {}      # x[i,a,b] 路径 0/1
        self.arr = {}    # arr[i,j] UAV i 到达任务 j 的时间
        self.s = {}      # s[j] 任务开始时间(任务级,协同共享)
        self.wait = {}   # wait[i,j] 等待
        self.E = {}      # E[i,j] UAV i 到达 j 时剩余电量

    # -----------------------------------------------------------------
    # 变量
    # -----------------------------------------------------------------
    def build_vars(self):
        ins = self.ins
        for u in ins.uavs:
            i = u.id
            # 路径变量:节点两两之间(不含自环)
            for a in self.nodes:
                for b in self.nodes:
                    if a == b:
                        continue
                    self.x[i, a, b] = self.m.addVar(
                        vtype=GRB.BINARY, name=f"x_{i}_{a}_{b}")
            # 到达时间 / 等待 / 电量:只对真实任务
            for t in ins.tasks:
                j = t.id
                self.arr[i, j] = self.m.addVar(lb=0, name=f"arr_{i}_{j}")
                self.wait[i, j] = self.m.addVar(lb=0, name=f"wait_{i}_{j}")
                self.E[i, j] = self.m.addVar(
                    lb=u.e_min, ub=u.e_full, name=f"E_{i}_{j}")

        # 任务开始时间:任务级(所有参与者共享,协同自动同步)
        for t in ins.tasks:
            self.s[t.id] = self.m.addVar(lb=0, name=f"s_{t.id}")

        self.m.update()

    # -----------------------------------------------------------------
    # 块1:协同需求(每个任务要几架 UAV / UGV)
    # -----------------------------------------------------------------
    def add_cooperation(self):
        # "UAV i 参与任务 j" = i 从某处进入 j,即 sum_a x[i,a,j] == 1
        for t in self.ins.tasks:
            j = t.id
            # 参与 j 的 UAV 数量 = 需求
            self.m.addConstr(
                gp.quicksum(self.x[u.id, a, j]
                            for u in self.ins.uavs
                            for a in self.nodes if a != j)
                >= t.need_uav,
                name=f"coop_uav_{j}")
            # UGV 同理(此处省略 UGV 路径变量,真实实现要给 UGV 也建一套 x)
            # self.m.addConstr(... >= t.need_ugv, name=f"coop_ugv_{j}")

    # -----------------------------------------------------------------
    # 块2:流量守恒(每架 UAV 的路径要连通,不断头)
    # -----------------------------------------------------------------
    def add_flow(self):
        for u in self.ins.uavs:
            i = u.id
            # 每架 UAV 从 START 出发一次
            self.m.addConstr(
                gp.quicksum(self.x[i, self.START, b]
                            for b in self.nodes if b != self.START) == 1,
                name=f"flow_start_{i}")
            # 每架 UAV 回到 END 一次
            self.m.addConstr(
                gp.quicksum(self.x[i, a, self.END]
                            for a in self.nodes if a != self.END) == 1,
                name=f"flow_end_{i}")
            # 每个任务节点:该 UAV 若进入,则必须离开(入度 = 出度)
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    gp.quicksum(self.x[i, a, j] for a in self.nodes if a != j)
                    == gp.quicksum(self.x[i, j, b] for b in self.nodes if b != j),
                    name=f"flow_bal_{i}_{j}")

    # -----------------------------------------------------------------
    # 块3:时间递推(= MTZ 破环 + 顺序 + 转场,一条顶三用)
    # -----------------------------------------------------------------
    def add_time(self):
        # 到达 b 的时间 >= 任务 a 结束时间 + 转场时间(a、b 都是真实任务时)
        # f[a] = s[a] + dur[a]
        for u in self.ins.uavs:
            i = u.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    tt = self.travel[i, a.id, b.id]
                    self.m.addConstr(
                        self.arr[i, b.id] >=
                        self.s[a.id] + a.dur + tt
                        - self.BIG_T * (1 - self.x[i, a.id, b.id]),
                        name=f"time_{i}_{a.id}_{b.id}")
            # 从 START 直接到任务 b:到达时间 >= 起点飞过去的时间
            for b in self.ins.tasks:
                d0 = self.ins.dist(u, b) / u.speed
                self.m.addConstr(
                    self.arr[i, b.id] >=
                    d0 - self.BIG_T * (1 - self.x[i, self.START, b.id]),
                    name=f"time_start_{i}_{b.id}")

    # -----------------------------------------------------------------
    # 块4:时间窗
    # -----------------------------------------------------------------
    def add_time_window(self):
        for t in self.ins.tasks:
            self.m.addConstr(self.s[t.id] >= t.e, name=f"tw_e_{t.id}")
            self.m.addConstr(self.s[t.id] + t.dur <= t.l, name=f"tw_l_{t.id}")

    # -----------------------------------------------------------------
    # 块5:等待(早到者等,开始时间 >= 所有参与者到达)
    # -----------------------------------------------------------------
    def add_wait(self):
        for u in self.ins.uavs:
            i = u.id
            for t in self.ins.tasks:
                j = t.id
                visits = gp.quicksum(self.x[i, a, j]
                                     for a in self.nodes if a != j)
                # 只有 i 参与 j 时才要求 s[j] >= arr[i,j];不参与用 big-M 放松
                self.m.addConstr(
                    self.s[j] >= self.arr[i, j] - self.BIG_T * (1 - visits),
                    name=f"wait_sync_{i}_{j}")
                # 等待时长 = 开始 - 到达(不参与时无意义,可不约束)
                self.m.addConstr(
                    self.wait[i, j] >= self.s[j] - self.arr[i, j]
                    - self.BIG_T * (1 - visits),
                    name=f"wait_def_{i}_{j}")

    # -----------------------------------------------------------------
    # 块6:能量递推 + 下限
    # -----------------------------------------------------------------
    def add_energy(self):
        for u in self.ins.uavs:
            i = u.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    # 简化耗电:飞行按距离,任务按 task_cost
                    move = self.ins.dist(a, b)     # 可换成 move_cost 表
                    self.m.addConstr(
                        self.E[i, b.id] <=
                        self.E[i, a.id] - move - b.task_cost
                        + u.e_full * (1 - self.x[i, a.id, b.id]),
                        name=f"energy_{i}_{a.id}_{b.id}")
            # 从起点出发满电(到第一个任务时的电量约束)
            for b in self.ins.tasks:
                move0 = self.ins.dist(u, b)
                self.m.addConstr(
                    self.E[i, b.id] <=
                    u.e_full - move0 - b.task_cost
                    + u.e_full * (1 - self.x[i, self.START, b.id]),
                    name=f"energy_start_{i}_{b.id}")

    # -----------------------------------------------------------------
    # 目标 + 求解
    # -----------------------------------------------------------------
    def set_objective(self):
        # 例:最小化 makespan(最后一个任务结束时间)
        makespan = self.m.addVar(lb=0, name="makespan")
        for t in self.ins.tasks:
            self.m.addConstr(makespan >= self.s[t.id] + t.dur)
        self.m.setObjective(makespan, GRB.MINIMIZE)

    def build(self):
        self.build_vars()
        self.add_cooperation()   # 块1
        self.add_flow()          # 块2
        self.add_time()          # 块3
        self.add_time_window()   # 块4
        self.add_wait()          # 块5
        self.add_energy()        # 块6
        self.set_objective()

    def solve(self):
        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            print(f"最优 makespan = {self.m.objVal:.2f}")
            for t in self.ins.tasks:
                print(f"  任务{t.id} 开始={self.s[t.id].X:.2f}")


def main():
    inst = make_small_instance()
    model = OptModel(inst)
    model.build()
    model.solve()


if __name__ == "__main__":
    main()
