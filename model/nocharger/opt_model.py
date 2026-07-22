"""
UAV/UGV 协同任务调度 —— 建模层(无充电车)
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
        self.uav_travel, self.ugv_travel, _ = inst.build_travel_table()
        self.task_ids = [t.id for t in inst.tasks]
        self.task_of = {t.id: t for t in inst.tasks}
        self.START = -1
        self.END = -2
        self.nodes = self.task_ids + [self.START, self.END]
        self.x_uav = {}
        self.arr_uav = {}
        self.s = {}
        self.E = {}
        self.x_ugv = {}
        self.arr_ugv = {}

    # =================================================================
    # 变量
    # =================================================================
    def build_vars(self):
        ins = self.ins
        for uav in ins.uavs:
            i = uav.id
            for a in self.nodes:
                for b in self.nodes:
                    if a == b: continue
                    if b == self.START or a == self.END: continue
                    self.x_uav[i, a, b] = self.m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{a}_{b}")
            for t in ins.tasks:
                j = t.id
                self.arr_uav[i, j] = self.m.addVar(lb=0, name=f"arr_{i}_{j}")
                self.E[i, j] = self.m.addVar(lb=uav.e_min, ub=uav.e_full, name=f"E_{i}_{j}")
        for t in ins.tasks:
            self.s[t.id] = self.m.addVar(lb=0, name=f"s_{t.id}")
        for ugv in ins.ugvs:
            gid = ugv.id
            for a in self.nodes:
                for b in self.nodes:
                    if a == b: continue
                    if b == self.START or a == self.END: continue
                    self.x_ugv[gid, a, b] = self.m.addVar(vtype=GRB.BINARY, name=f"x_ugv_{gid}_{a}_{b}")
            for t in ins.tasks:
                self.arr_ugv[gid, t.id] = self.m.addVar(lb=0, name=f"arr_ugv_{gid}_{t.id}")
        self.m.update()

    def _visits_uav(self, i, j):
        return gp.quicksum(self.x_uav[i, a, j] for a in self.nodes if a != j and (i, a, j) in self.x_uav)
    def _visits_ugv(self, i, j):
        return gp.quicksum(self.x_ugv[i, a, j] for a in self.nodes if a != j and (i, a, j) in self.x_ugv)
    def _leave_uav(self, i, j):
        return gp.quicksum(self.x_uav[i, j, a] for a in self.nodes if a != j and (i, j, a) in self.x_uav)
    def _leave_ugv(self, i, j):
        return gp.quicksum(self.x_ugv[i, j, a] for a in self.nodes if a != j and (i, j, a) in self.x_ugv)

    # =================================================================
    # 块1:协同
    # =================================================================
    def add_cooperation(self):
        for t in self.ins.tasks:
            self.m.addConstr(
                gp.quicksum(self._visits_uav(uav.id, t.id) for uav in self.ins.uavs) == t.need_uav,
                name=f"coop_uav_{t.id}")
            if t.need_ugv > 0:
                self.m.addConstr(
                    gp.quicksum(self._visits_ugv(ugv.id, t.id) for ugv in self.ins.ugvs) == t.need_ugv,
                    name=f"coop_ugv_{t.id}")

    # =================================================================
    # 块2:流量守恒
    # =================================================================
    def add_flow(self):
        for uav in self.ins.uavs:
            i = uav.id
            self.m.addConstr(
                gp.quicksum(self.x_uav[i, self.START, b] for b in self.nodes
                            if b != self.START and (i, self.START, b) in self.x_uav) == 1,
                name=f"flow_start_{i}")
            self.m.addConstr(
                gp.quicksum(self.x_uav[i, a, self.END] for a in self.nodes
                            if a != self.END and (i, a, self.END) in self.x_uav) == 1,
                name=f"flow_end_{i}")
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    self._visits_uav(i, j) == self._leave_uav(i, j),
                    name=f"flow_bal_{i}_{j}")
        for ugv in self.ins.ugvs:
            gid = ugv.id
            self.m.addConstr(
                gp.quicksum(self.x_ugv[gid, self.START, b] for b in self.nodes
                            if b != self.START and (gid, self.START, b) in self.x_ugv) == 1,
                name=f"flow_start_ugv_{gid}")
            self.m.addConstr(
                gp.quicksum(self.x_ugv[gid, a, self.END] for a in self.nodes
                            if a != self.END and (gid, a, self.END) in self.x_ugv) == 1,
                name=f"flow_end_ugv_{gid}")
            for t in self.ins.tasks:
                j = t.id
                self.m.addConstr(
                    self._visits_ugv(gid, j) == self._leave_ugv(gid, j),
                    name=f"flow_bal_ugv_{gid}_{j}")

    # =================================================================
    # 块3:时间递推(Indicator)
    # =================================================================
    def add_time(self):
        for uav in self.ins.uavs:
            i = uav.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id: continue
                    tt = self.uav_travel[i, a.id, b.id]
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a.id, b.id], True,
                        self.arr_uav[i, b.id] >= self.s[a.id] + a.dur + tt,
                        name=f"time_{i}_{a.id}_{b.id}")
            for b in self.ins.tasks:
                tt0 = self.uav_travel[i, -1, b.id]
                self.m.addGenConstrIndicator(
                    self.x_uav[i, self.START, b.id], True,
                    self.arr_uav[i, b.id] >= tt0,
                    name=f"time_start_{i}_{b.id}")
        for ugv in self.ins.ugvs:
            gid = ugv.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id: continue
                    tt = self.ugv_travel[gid, a.id, b.id]
                    self.m.addGenConstrIndicator(
                        self.x_ugv[gid, a.id, b.id], True,
                        self.arr_ugv[gid, b.id] >= self.s[a.id] + a.dur + tt,
                        name=f"time_ugv_{gid}_{a.id}_{b.id}")
            for b in self.ins.tasks:
                tt = self.ugv_travel[gid, -1, b.id]
                self.m.addGenConstrIndicator(
                    self.x_ugv[gid, self.START, b.id], True,
                    self.arr_ugv[gid, b.id] >= tt,
                    name=f"time_start_ugv_{gid}_{b.id}")

    def add_precedence(self):
        for t in self.ins.tasks:
            for p in t.preds:
                p_task = self.task_of[p]
                self.m.addConstr(self.s[t.id] >= self.s[p] + p_task.dur,
                                 name=f"prec_{p}_before_{t.id}")

    def add_wait(self):
        for uav in self.ins.uavs:
            i = uav.id
            for t in self.ins.tasks:
                j = t.id
                for a in self.nodes:
                    if a == j or (i, a, j) not in self.x_uav: continue
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a, j], True,
                        self.s[j] >= self.arr_uav[i, j],
                        name=f"wait_sync_{i}_{a}_{j}")
        for ugv in self.ins.ugvs:
            gid = ugv.id
            for t in self.ins.tasks:
                j = t.id
                for a in self.nodes:
                    if a == j or (gid, a, j) not in self.x_ugv: continue
                    self.m.addGenConstrIndicator(
                        self.x_ugv[gid, a, j], True,
                        self.s[j] >= self.arr_ugv[gid, j],
                        name=f"wait_sync_ugv_{gid}_{a}_{j}")

    def add_energy(self):
        for uav in self.ins.uavs:
            i = uav.id
            for a in self.ins.tasks:
                for b in self.ins.tasks:
                    if a.id == b.id: continue
                    move = self.ins.dist(a, b)
                    self.m.addGenConstrIndicator(
                        self.x_uav[i, a.id, b.id], True,
                        self.E[i, b.id] <= self.E[i, a.id] - move - b.task_cost,
                        name=f"energy_{i}_{a.id}_{b.id}")
            for b in self.ins.tasks:
                move0 = self.ins.dist(uav, b)
                self.m.addGenConstrIndicator(
                    self.x_uav[i, self.START, b.id], True,
                    self.E[i, b.id] <= uav.e_full - move0 - b.task_cost,
                    name=f"energy_start_{i}_{b.id}")

    def set_objective(self):
        makespan = self.m.addVar(lb=0, name="makespan")
        for t in self.ins.tasks:
            self.m.addConstr(makespan >= self.s[t.id] + t.dur)
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

    # =================================================================
    # 组装 / 求解 / 输出
    # =================================================================
    def build(self):
        self.build_vars()
        self.add_cooperation()
        self.add_flow()
        self.add_time()
        self.add_precedence()
        self.add_wait()
        self.add_energy()
        self.set_objective()

    def solve(self, time_limit=None, mip_gap=None):
        if time_limit is not None: self.m.Params.TimeLimit = time_limit
        if mip_gap is not None: self.m.Params.MIPGap = mip_gap
        self.m.optimize()
        if self.m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or self.m.SolCount > 0:
            self._report()
        else:
            print("无解, status =", self.m.status)

    def _route_of(self, i, edges):
        route = [self.START]
        cur = self.START
        while cur != self.END:
            nxt = None
            for b in self.nodes:
                if b != cur and (i, cur, b) in edges and edges[i, cur, b].X > 0.5:
                    nxt = b; break
            if nxt is None: break
            route.append(nxt)
            cur = nxt
        return route

    def _fmt_route(self, route):
        return " -> ".join("S" if p == self.START else "E" if p == self.END else f"T{p}" for p in route)

    def _report(self):
        print(f"\n最优 makespan = {self.makespan.X:.2f}")
        print("UAV 路线:")
        for uav in self.ins.uavs:
            route = self._route_of(uav.id, self.x_uav)
            print(f"  UAV{uav.id}: {self._fmt_route(route)}")
        print("UGV 路线:")
        for ugv in self.ins.ugvs:
            route = self._route_of(ugv.id, self.x_ugv)
            print(f"  UGV{ugv.id}: {self._fmt_route(route)}")
        self._report_timeline()

    def _report_timeline(self):
        print(f"\n{'='*60}")
        print(f"  时间线  (makespan = {self.makespan.X:.2f})")
        print(f"{'='*60}")
        print("\n[任务]  开始→结束      参与个体")
        for t in sorted(self.ins.tasks, key=lambda x: self.s[x.id].X):
            members = []
            for uav in self.ins.uavs:
                if self._visits_uav(uav.id, t.id).getValue() > 0.5:
                    members.append(f"UAV{uav.id}")
            for ugv in self.ins.ugvs:
                if self._visits_ugv(ugv.id, t.id).getValue() > 0.5:
                    members.append(f"UGV{ugv.id}")
            start = self.s[t.id].X
            end = start + t.dur
            print(f"  T{t.id:2d}  [{start:6.2f} → {end:6.2f}]  {', '.join(members)}")

        def _agent_timeline(agent_id, edges, travel_table, arr_dict, label):
            route = self._route_of(agent_id, edges)
            if len(route) < 3: return
            print(f"\n[{label}]")
            prev = self.START
            t_now = 0.0
            for node in route[1:]:
                if node in self.task_ids:
                    tt = travel_table[agent_id, prev, node]
                    arr_real = t_now + tt
                    print(f"  [{t_now:6.2f}] ── 飞行 {tt:5.2f} ──→ 到达 T{node} arr={arr_real:6.2f}")
                    t_now = self.s[node].X
                    t = self.task_of[node]
                    print(f"  [{t_now:6.2f}] ══ 执行 T{node} dur={t.dur:.2f} → [{t_now+t.dur:6.2f}]")
                    t_now += t.dur
                elif node == self.END:
                    tt = travel_table[agent_id, prev, -2]
                    print(f"  [{t_now:6.2f}] ── 回程 {tt:5.2f} ──→ END")
                prev = node

        for uav in self.ins.uavs:
            _agent_timeline(uav.id, self.x_uav, self.uav_travel, self.arr_uav, f"UAV{uav.id}")
        for ugv in self.ins.ugvs:
            _agent_timeline(ugv.id, self.x_ugv, self.ugv_travel, self.arr_ugv, f"UGV{ugv.id}")
        print(f"\n{'='*60}")


def main():
    inst = make_small_instance()
    model = OptModel(inst)
    model.build()
    model.solve(time_limit=60)


if __name__ == "__main__":
    main()
