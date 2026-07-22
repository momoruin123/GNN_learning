"""
UAV/UGV 协同任务调度 —— 建模层
=====================================
结构说明:
  - 数据层 model_skeleton.py 负责实体和预计算(距离/时间表)
  - 本文件只负责 Gurobi:变量 + 约束(分块) + 目标 + 求解 + 输出
  - 每个约束一个方法,build() 里按顺序调用。
    想临时关掉某块 -> 在 build() 里注释掉那一行即可,方便逐块调试。
  - 想扩展模型规模 -> 只改 make_random_instance 的参数,本文件不用动。

节点约定:
  对每架 UAV,路径走在 nodes 上:nodes = 所有任务id + [START, END]
  x[i,a,b]=1 表示 UAV i 从节点 a 直飞节点 b。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.entities import make_random_instance, make_small_instance
import gurobipy as gp
from gurobipy import GRB


class OptModel:
    def __init__(self, inst):
        self.ins = inst
        self.m = gp.Model("uav_ugv_sched")

        # 预计算表:(agent_id, from, to) -> 时间
        self.uav_travel, self.ugv_travel, self.charger_travel = inst.build_travel_table()

        # 节点集合:任务 id + 两个虚拟点
        self.task_ids = [t.id for t in inst.tasks]
        self.task_of = {t.id: t for t in inst.tasks}   # id -> Task,查参数用
        self.START = -1
        self.END = -2
        self.nodes = self.task_ids + [self.START, self.END]

        # 变量容器(统一放这里,key=元组)
        self.x_uav = {}      # x[i,a,b]  UAV 路径 0/1
        self.arr_uav = {}    # arr_uav[i,j]  UAV i 到达任务 j 的时间
        self.s = {}          # s[j]      任务开始时间(任务级,协同共享)
        self.E = {}          # E[i,j]    UAV i 到达 j 时剩余电量

        # UGV 一套平行变量(UGV 不耗电,不建能量)
        self.x_ugv = {}     # x_ugv[g,a,b] UGV 路径 0/1
        self.arr_ugv = {}   # arr_ugv[g,j] UGV g 到达任务 j 的时间

        # 充电车变量(只充电,不执行任务)
        self.x_charger = {}     # x_charger[c,a,b] 充电车路径
        self.arr_charger = {}   # arr_charger[c,j] 充电车到达时间
        self.t_charger_end = {} # t_charger_end[c,j] 充电完成离开时间
        self.meet = {}          # meet[i,c,j] UAV i 与充电车 c 在 j 会合充电

    # =================================================================
    # 变量
    # =================================================================
    def build_vars(self):
        ins = self.ins
        # ---- UGV 路径 + 到达时间----

        for uav in ins.uavs:
            i = uav.id
            for a in self.nodes:
                for b in self.nodes:
                    if a == b:
                        continue
                    # 禁止非法方向的边(从源头杜绝子回路/空跑):
                    #   START 只能出、不能进;END 只能进、不能出
                    #   但保留 START->END(表示该 UAV 待命不出动)
                    if b == self.START:      # 任何边指向 START -> 非法
                        continue
                    if a == self.END:        # 从 END 出发 -> 非法
                        continue
                    self.x_uav[i, a, b] = self.m.addVar(
                        vtype=GRB.BINARY, name=f"x_{i}_{a}_{b}")
            for t in ins.tasks:
                j = t.id
                self.arr_uav[i, j] = self.m.addVar(lb=0, name=f"arr_{i}_{j}")
                self.E[i, j] = self.m.addVar(
                    lb=uav.e_min, ub=uav.e_full, name=f"E_{i}_{j}")

        for t in ins.tasks:
            self.s[t.id] = self.m.addVar(lb=0, name=f"s_{t.id}")

        # ---- UGV 路径 + 到达时间----
        for ugv in ins.ugvs:
            ugv_id = ugv.id
            for a in self.nodes:
                for b in self.nodes:
                    if a == b:
                        continue
                    if b == self.START or a == self.END:
                        continue
                    self.x_ugv[ugv_id, a, b] = self.m.addVar(
                        vtype=GRB.BINARY, name=f"x_ugv_{ugv_id}_{a}_{b}")
            for t in ins.tasks:
                self.arr_ugv[ugv_id, t.id] = self.m.addVar(
                    lb=0, name=f"arr_ugv_{ugv_id}_{t.id}")

        # ---- 充电车路径 + 到达时间 ----
        for ch in ins.chargers:
            c = ch.id
            for a in self.nodes:
                for b in self.nodes:
                    if a == b:
                        continue
                    if b == self.START or a == self.END:
                        continue
                    self.x_charger[c, a, b] = self.m.addVar(
                        vtype=GRB.BINARY, name=f"x_ch_{c}_{a}_{b}")
            for t in ins.tasks:
                self.arr_charger[c, t.id] = self.m.addVar(
                    lb=0, name=f"arr_ch_{c}_{t.id}")
                self.t_charger_end[c, t.id] = self.m.addVar(
                    lb=0, name=f"t_ch_end_{c}_{t.id}")

        # ---- 会合变量 meet[i,c,j] ----
        for uav in ins.uavs:
            i = uav.id
            for ch in ins.chargers:
                c = ch.id
                for t in ins.tasks:
                    j = t.id
                    self.meet[i, c, j] = self.m.addVar(
                        vtype=GRB.BINARY, name=f"meet_{i}_{c}_{j}")

        self.m.update()

    # UAV i 对任务 j 的入度
    def _visits_uav(self, i, j):
        return gp.quicksum(self.x_uav[i, a, j] for a in self.nodes
                           if a != j and (i, a, j) in self.x_uav)

    # UGV i 对任务 j 的入度
    def _visits_ugv(self, i, j):
        return gp.quicksum(self.x_ugv[i, a, j] for a in self.nodes
                           if a != j and (i, a, j) in self.x_ugv)

    # 充电车 c 对任务 j 的入度
    def _visits_charger(self, c, j):
        return gp.quicksum(self.x_charger[c, a, j] for a in self.nodes
                           if a != j and (c, a, j) in self.x_charger)

    # 充电车 c 从任务 j 的出度
    def _leave_charger(self, c, j):
        return gp.quicksum(self.x_charger[c, j, a] for a in self.nodes
                           if a != j and (c, j, a) in self.x_charger)

    # UAV i 对任务 j 的出度
    def _leave_uav(self, i, j):
        return gp.quicksum(self.x_uav[i, j, a] for a in self.nodes
                           if a != j and (i, j, a) in self.x_uav)

    # UGV i 对任务 j 的出度
    def _leave_ugv(self, i, j):
        return gp.quicksum(self.x_ugv[i, j, a] for a in self.nodes
                           if a != j and (i, j, a) in self.x_ugv)
    # =================================================================
    # 块1:协同需求(每个任务要几架 UAV / 几台 UGV)
    # =================================================================
    def add_cooperation(self):
        for t in self.ins.tasks:
            self.m.addConstr(
                gp.quicksum(self._visits_uav(uav.id, t.id) for uav in self.ins.uavs)
                == t.need_uav,
                name=f"coop_uav_{t.id}")
            if t.need_ugv > 0:
                self.m.addConstr(
                    gp.quicksum(self._visits_ugv(ugv.id, t.id)
                                for ugv in self.ins.ugvs)
                    == t.need_ugv,
                    name=f"coop_ugv_{t.id}")

    # =================================================================
    # 块2:流量守恒
    # =================================================================
    def add_flow(self):
        # UAV 流量守恒
        for uav in self.ins.uavs:
            i = uav.id
            # 起点出度=1
            self.m.addConstr(
                gp.quicksum(self.x_uav[i, self.START, b]
                            for b in self.nodes
                            if b != self.START and (i, self.START, b) in self.x_uav) == 1,
                name=f"flow_start_{i}")
            # 终点入度=1
            self.m.addConstr(
                gp.quicksum(self.x_uav[i, a, self.END]
                            for a in self.nodes
                            if a != self.END and (i, a, self.END) in self.x_uav) == 1,
                name=f"flow_end_{i}")
            # 任务出入相等
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    self._visits_uav(i, j) == self._leave_uav(i, j),
                    name=f"flow_bal_{i}_{j}")

        # UGV 流量守恒
        for ugv in self.ins.ugvs:
            ugv_id = ugv.id
            self.m.addConstr(
                gp.quicksum(self.x_ugv[ugv_id, self.START, b]
                            for b in self.nodes
                            if b != self.START and (ugv_id, self.START, b) in self.x_ugv) == 1,
                name=f"flow_start_ugv_{ugv_id}")
            self.m.addConstr(
                gp.quicksum(self.x_ugv[ugv_id, a, self.END]
                            for a in self.nodes
                            if a != self.END and (ugv_id, a, self.END) in self.x_ugv) == 1,
                name=f"flow_end_ugv_{ugv_id}")
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    self._visits_ugv(ugv_id, j) == self._leave_ugv(ugv_id, j),
                    name=f"flow_bal_ugv_{ugv_id}_{j}")

        # 充电车流量守恒
        for ch in self.ins.chargers:
            c = ch.id
            self.m.addConstr(
                gp.quicksum(self.x_charger[c, self.START, b]
                            for b in self.nodes
                            if b != self.START and (c, self.START, b) in self.x_charger) == 1,
                name=f"flow_start_ch_{c}")
            self.m.addConstr(
                gp.quicksum(self.x_charger[c, a, self.END]
                            for a in self.nodes
                            if a != self.END and (c, a, self.END) in self.x_charger) == 1,
                name=f"flow_end_ch_{c}")
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    self._visits_charger(c, j) == self._leave_charger(c, j),
                    name=f"flow_bal_ch_{c}_{j}")

    # =================================================================
    # 块3:时间递推(= 破环 + 顺序 + 转场)。用 Indicator:x=1 时才强制
    # =================================================================
    def add_time(self):
        for uav in self.ins.uavs:
            i = uav.id
            # 任务a -> 任务b:到达时间 >= a到达 + 任务持续时间 + 飞行时间
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    tt = self.uav_travel[i, a.id, b.id]
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a.id, b.id], True,
                        self.arr_uav[i, b.id] >= self.s[a.id] + a.dur + tt,
                        name=f"time_{i}_{a.id}_{b.id}")
            # START -> 任务 b:到达时间 >= 从起点飞过去的时间
            for b in self.ins.tasks:
                tt0 = self.uav_travel[i, -1, b.id]
                self.m.addGenConstrIndicator(
                    self.x_uav[i, self.START, b.id], True,
                    self.arr_uav[i, b.id] >= tt0,
                    name=f"time_start_{i}_{b.id}")

        # UGV 时间递推(和 UAV 同构)
        for ugv in self.ins.ugvs:
            gid = ugv.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    tt = self.ugv_travel[gid, a.id, b.id]
                    self.m.addGenConstrIndicator(
                        self.x_ugv[gid, a.id, b.id], True,
                        self.arr_ugv[gid, b.id] >= self.s[a.id] + a.dur + tt,
                        name=f"time_ugv_{gid}_{a.id}_{b.id}")
            for b in self.ins.tasks:
                tt = self.ugv_travel[gid, self.START, b.id]
                self.m.addGenConstrIndicator(
                    self.x_ugv[gid, self.START, b.id], True,
                    self.arr_ugv[gid, b.id] >= tt,
                    name=f"time_start_ugv_{gid}_{b.id}")

        # 充电车时间递推(自己的时钟,不绑任务时间)
        for ch in self.ins.chargers:
            c = ch.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    tt = self.charger_travel[c, a.id, b.id]
                    self.m.addGenConstrIndicator(
                        self.x_charger[c, a.id, b.id], True,
                        self.arr_charger[c, b.id] >= self.t_charger_end[c, a.id] + tt,
                        name=f"time_ch_{c}_{a.id}_{b.id}")
            # 充电耗时:任务点停留时间 >= 到达 + 充电时间
            for a in self.ins.tasks:
                charge_need = gp.quicksum(
                    self.meet[uav.id, c, a.id]
                    for uav in self.ins.uavs
                    if (uav.id, c, a.id) in self.meet)
                self.m.addConstr(
                    self.t_charger_end[c, a.id] >= self.arr_charger[c, a.id] + ch.charge_time * charge_need,
                    name=f"charge_dur_{c}_{a.id}")
            for b in self.ins.tasks:
                tt = self.charger_travel[c, self.START, b.id]
                self.m.addGenConstrIndicator(
                    self.x_charger[c, self.START, b.id], True,
                    self.arr_charger[c, b.id] >= tt,
                    name=f"time_start_ch_{c}_{b.id}")

    # =================================================================
    # 块4:时间窗
    # =================================================================
    def add_time_window(self):
        for t in self.ins.tasks:
            self.m.addConstr(self.s[t.id] >= t.e, name=f"tw_e_{t.id}")
            self.m.addConstr(self.s[t.id] + t.dur <= t.l, name=f"tw_l_{t.id}")

    # =================================================================
    # 块4.5:先后顺序(前置任务结束 <= 本任务开始)
    #   无 big-M,硬约束,最干净。preds 来自 Task.preds
    # =================================================================
    def add_precedence(self):
        for t in self.ins.tasks:
            for p in t.preds:
                p_task = self.task_of[p]
                self.m.addConstr(
                    self.s[t.id] >= self.s[p] + p_task.dur,
                    name=f"prec_{p}_before_{t.id}")

    # =================================================================
    # 块5:等待(早到者等,开始时间 >= 所有参与者到达)
    #   Indicator 条件须是单个二元变量,故对每条入边 x[i,a,j] 分别写
    # =================================================================
    def add_wait(self):
        for uav in self.ins.uavs:
            i = uav.id
            for t in self.ins.tasks:
                j = t.id
                for a in self.nodes:
                    if a == j or (i, a, j) not in self.x_uav:
                        continue
                    # 若 UAV i 经 a 进入 j(x=1),则任务开始 >= 其到达
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a, j], True,
                        self.s[j] >= self.arr_uav[i, j],
                        name=f"wait_sync_{i}_{a}_{j}")

        # UGV 也要到齐
        for ugv in self.ins.ugvs:
            gid = ugv.id
            for t in self.ins.tasks:
                j = t.id
                for a in self.nodes:
                    if a == j or (gid, a, j) not in self.x_ugv:
                        continue
                    self.m.addGenConstrIndicator(
                        self.x_ugv[gid, a, j], True,
                        self.s[j] >= self.arr_ugv[gid, j],
                        name=f"wait_sync_ugv_{gid}_{a}_{j}")

        # 充电车也要同步(到齐后才能充电)
        for ch in self.ins.chargers:
            c = ch.id
            for t in self.ins.tasks:
                j = t.id
                for a in self.nodes:
                    if a == j or (c, a, j) not in self.x_charger:
                        continue
                    self.m.addGenConstrIndicator(
                        self.x_charger[c, a, j], True,
                        self.s[j] >= self.t_charger_end[c, j],
                        name=f"wait_sync_ch_{c}_{a}_{j}")

    # =================================================================
    # 块5.5:会合充电约束
    #   meet[i,c,j]=1 表示 UAV i 和充电车 c 在任务 j 会合充电
    # =================================================================
    def add_rendezvous(self):
        for uav in self.ins.uavs:
            i = uav.id
            for ch in self.ins.chargers:
                c = ch.id
                for t in self.ins.tasks:
                    j = t.id
                    m_ij = self.meet[i, c, j]
                    # 会合条件:UAV 必须访问 j
                    self.m.addConstr(m_ij <= self._visits_uav(i, j),
                                     name=f"meet_uav_{i}_{c}_{j}")
                    # 会合条件:充电车必须访问 j
                    self.m.addConstr(m_ij <= self._visits_charger(c, j),
                                     name=f"meet_ch_{i}_{c}_{j}")

    # =================================================================
    # 块6:能量递推 + 下限
    #   扩展点:等待悬停耗电 -> 加 wait 变量后,这里减 hover_cost*wait
    # =================================================================
    def add_energy(self):
        for uav in self.ins.uavs:
            i = uav.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id:
                        continue
                    move = self.ins.dist(a, b)
                    # 充电增益:所有充电车在 b 点给 UAV i 充电之和
                    charge_gain = gp.quicksum(
                        ch.charge_rate * self.meet[i, ch.id, b.id]
                        for ch in self.ins.chargers)
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a.id, b.id], True,
                        self.E[i, b.id] <= self.E[i, a.id] - move - b.task_cost + charge_gain,
                        name=f"energy_{i}_{a.id}_{b.id}")
            for b in self.ins.tasks:
                move0 = self.ins.dist(uav, b)
                charge_gain = gp.quicksum(
                    ch.charge_rate * self.meet[i, ch.id, b.id]
                    for ch in self.ins.chargers)
                self.m.addGenConstrIndicator(
                    self.x_uav[i, self.START, b.id], True,
                    self.E[i, b.id] <= uav.e_full - move0 - b.task_cost + charge_gain,
                    name=f"energy_start_{i}_{b.id}")

    # =================================================================
    # 目标
    # =================================================================
    def set_objective(self):
        makespan = self.m.addVar(lb=0, name="makespan")
        for t in self.ins.tasks:
            self.m.addConstr(makespan >= self.s[t.id] + t.dur)
        # 回程:UAV 做完最后一个任务后飞到 END 的时间也计入
        for uav in self.ins.uavs:
            i = uav.id
            for t in self.ins.tasks:
                j = t.id
                tt_back = self.uav_travel[i, j, -2]
                self.m.addGenConstrIndicator(
                    self.x_uav[i, j, self.END], True,
                    makespan >= self.s[j] + t.dur + tt_back,
                    name=f"return_{i}_{j}")
        self.makespan = makespan
        self.m.setObjective(makespan, GRB.MINIMIZE)

    def set_warm_start(self, routes_uav, routes_ugv, arr, s, e_dict):
        """
        MIP 热启动:只设路线上的边为 1,其他二元变量不碰。
        连续变量(arr/s/E)也设,让 Gurobi 拿到完整可行解。
        """
        START, END = self.START, self.END

        def _set_route(agent_id, seq, edges_dict):
            if not seq:
                edges_dict[agent_id, START, END].Start = 1.0
                return
            prev = START
            for tid in seq:
                edges_dict[agent_id, prev, tid].Start = 1.0
                prev = tid
            edges_dict[agent_id, prev, END].Start = 1.0

        for u in self.ins.uavs:
            _set_route(u.id, routes_uav.get(u.id, []), self.x_uav)
        for g in self.ins.ugvs:
            _set_route(g.id, routes_ugv.get(g.id, []), self.x_ugv)

        for tid, val in s.items():
            self.s[tid].Start = val
        for (i, tid), val in arr.items():
            if (i, tid) in self.arr_uav:
                self.arr_uav[i, tid].Start = val
        for (i, tid), val in e_dict.items():
            if (i, tid) in self.E:
                self.E[i, tid].Start = val
        self.m.update()

    # =================================================================
    # 组装 / 求解 / 输出
    # =================================================================
    def build(self):
        self.build_vars()
        self.add_cooperation()   # 块1
        self.add_flow()          # 块2
        self.add_time()          # 块3
        # self.add_time_window()   # 块4 (暂不需要时间窗)
        self.add_precedence()    # 块4.5 先后顺序
        self.add_wait()          # 块5
        self.add_rendezvous()    # 块5.5 会合充电
        self.add_energy()        # 块6
        self.set_objective()

    def solve(self, time_limit=None, mip_gap=None):
        if time_limit is not None:
            self.m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            self.m.Params.MIPGap = mip_gap
        self.m.optimize()
        if self.m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or self.m.SolCount > 0:
            self._report()
        elif self.m.status == GRB.INFEASIBLE:
            print("无可行解 (INFEASIBLE)。计算 IIS 找冲突约束...")
            self.m.computeIIS()
            print("以下约束互相冲突:")
            for c in self.m.getConstrs():
                if c.IISConstr:
                    print("  ", c.ConstrName)
        else:
            print("未找到解, status =", self.m.status)

    def _route_of(self, i, edges):
        """沿 edges[i,a,b]=1 还原一个个体的路线(START->...->END)
        edges 传 self.x_uav(UAV)或 self.x_ugv(UGV)"""
        route = [self.START]
        cur = self.START
        while cur != self.END:
            nxt = None
            for b in self.nodes:
                if b != cur and (i, cur, b) in edges and edges[i, cur, b].X > 0.5:
                    nxt = b
                    break
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
        return route

    def _fmt_route(self, route):
        return " -> ".join(
            "S" if p == self.START else "E" if p == self.END else f"T{p}"
            for p in route)

    def _report(self):
        print(f"\n最优 makespan = {self.makespan.X:.2f}")
        # print("任务时间:")
        # for t in self.ins.tasks:
        #     prec = f"  前置={t.preds}" if t.preds else ""
        #     print(f"  任务{t.id}: 开始={self.s[t.id].X:6.2f}  "
        #           f"结束={self.s[t.id].X + t.dur:6.2f}{prec}")
        print("UAV 路线:")
        for uav in self.ins.uavs:
            route = self._route_of(uav.id, self.x_uav)
            print(f"  UAV{uav.id}: {self._fmt_route(route)}")
        print("UGV 路线:")
        for ugv in self.ins.ugvs:
            route = self._route_of(ugv.id, self.x_ugv)
            print(f"  UGV{ugv.id}: {self._fmt_route(route)}")
        if self.ins.chargers:
            print("充电车 路线:")
            for ch in self.ins.chargers:
                route = self._route_of(ch.id, self.x_charger)
                print(f"  CH{ch.id}: {self._fmt_route(route)}")
            # 诊断:充电会合
            print("\n[诊断] 会合充电 meet[i,c,j]=1:")
            for uav in self.ins.uavs:
                i = uav.id
                for ch in self.ins.chargers:
                    c = ch.id
                    for t in self.ins.tasks:
                        j = t.id
                        if (i, c, j) in self.meet and self.meet[i, c, j].X > 0.5:
                            print(f"  UAV{i} + CH{c} @ T{j}")

        # ---- 诊断:每架 UAV 到达各任务的时间 vs 任务开始 ----
        # print("\n[诊断] UAV 到达时间 arr / 任务开始 s:")
        # for uav in self.ins.uavs:
        #     i = uav.id
        #     for t in self.ins.tasks:
        #         if self._visits_uav(i, t.id).getValue() > 0.5:
        #             ok = "OK" if self.s[t.id].X >= self.arr_uav[i,t.id].X - 1e-4 else "!!!违反 s<arr"
        #             print(f"  UAV{i} 到达 T{t.id}: arr={self.arr_uav[i,t.id].X:6.2f}"
        #                   f"  (任务开始 s={self.s[t.id].X:6.2f})  {ok}")

        # ---- 诊断:所有激活的边 + 每个任务被访问次数 ----
        # print("\n[诊断] 激活的边 (x=1):")
        # for uav in self.ins.uavs:
        #     i = uav.id
        #     for a in self.nodes:
        #         for b in self.nodes:
        #             if a != b and (i, a, b) in self.x_uav and self.x_uav[i, a, b].X > 0.5:
        #                 na = "S" if a == self.START else "E" if a == self.END else f"T{a}"
        #                 nb = "S" if b == self.START else "E" if b == self.END else f"T{b}"
        #                 print(f"  UAV{i}: {na} -> {nb}")
        # print("[诊断] 每个任务被访问次数 (need_uav):")
        # for t in self.ins.tasks:
        #     cnt = sum(self.x_uav[uav.id, a, t.id].X
        #               for uav in self.ins.uavs
        #               for a in self.nodes
        #               if a != t.id and (uav.id, a, t.id) in self.x_uav)
        #     print(f"  任务{t.id}: 访问={cnt:.0f}  需求={t.need_uav}")

        self._report_timeline()

    def _report_timeline(self):
        """按时间线展示各 UAV/UGV 的飞行-等待-执行-回程,以及每个任务的参与者。"""
        print(f"\n{'='*60}")
        print(f"  时间线  (makespan = {self.makespan.X:.2f})")
        print(f"{'='*60}")

        # ---- 任务窗口 ----
        print("\n[任务]  开始→结束      参与个体")
        for t in sorted(self.ins.tasks, key=lambda x: self.s[x.id].X):
            members = []
            for uav in self.ins.uavs:
                if self._visits_uav(uav.id, t.id).getValue() > 0.5:
                    members.append(f"UAV{uav.id}")
            for ugv in self.ins.ugvs:
                if self._visits_ugv(ugv.id, t.id).getValue() > 0.5:
                    members.append(f"UGV{ugv.id}")
            for ch in self.ins.chargers:
                if self._visits_charger(ch.id, t.id).getValue() > 0.5:
                    members.append(f"CH{ch.id}")
            start = self.s[t.id].X
            end = start + t.dur
            print(f"  T{t.id:2d}  [{start:6.2f} → {end:6.2f}]  {', '.join(members)}")

        # ---- 个体时间线 ----
        def _agent_timeline(agent_id, edges, travel_table, arr_dict, agent_label, is_charger=False):
            route_nodes = self._route_of(agent_id, edges)
            if len(route_nodes) < 3:
                return
            print(f"\n[{agent_label}]")
            prev_node = self.START
            time_now = 0.0
            for node in route_nodes[1:]:
                if node in self.task_ids:
                    tt = travel_table[agent_id, prev_node, node]
                    arr_real = time_now + tt
                    print(f"  [{time_now:6.2f}] ── 飞行 {tt:5.2f} ──→ 到达 T{node} arr={arr_real:6.2f}")
                    if is_charger:
                        ch_obj = next(ch for ch in self.ins.chargers if ch.id == agent_id)
                        end_val = self.t_charger_end[agent_id, node].X
                        print(f"  [{arr_val:6.2f}] ~~ 充电 dur={ch_obj.charge_time:.1f} ~~ → [{end_val:6.2f}]")
                        time_now = end_val
                    else:
                        time_now = self.s[node].X
                        t = self.task_of[node]
                        print(f"  [{time_now:6.2f}] ══ 执行 T{node} dur={t.dur:.2f} → [{time_now+t.dur:6.2f}]")
                        time_now += t.dur
                elif node == self.END:
                    tt = travel_table[agent_id, prev_node, -2]
                    print(f"  [{time_now:6.2f}] ── 回程 {tt:5.2f} ──→ END")
                prev_node = node

        for uav in self.ins.uavs:
            _agent_timeline(uav.id, self.x_uav, self.uav_travel, self.arr_uav, f"UAV{uav.id}")

        for ugv in self.ins.ugvs:
            _agent_timeline(ugv.id, self.x_ugv, self.ugv_travel, self.arr_ugv, f"UGV{ugv.id}")

        for ch in self.ins.chargers:
            _agent_timeline(ch.id, self.x_charger, self.charger_travel, self.arr_charger, f"CH{ch.id}", is_charger=True)

        print(f"\n{'='*60}")


def main():
    # inst = make_random_instance(n_uav=3, n_ugv=1, n_task=8, seed=1)
    inst = make_small_instance()
    model = OptModel(inst)
    model.build()
    model.solve(time_limit=60)


if __name__ == "__main__":
    main()
