"""
热启动对照实验:
  A: 无热启动(基线)
  B: 最优解热启动(把最优解重新喂给 Gurobi)
  C: 可行但不优(贪心)
  D: 贼烂解(随机乱填)

对比:求解时间、MIP gap 收敛、节点数。
"""

import time
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.entities import make_random_instance


# =====================================================================
# 辅助函数:构造路线 → Gurobi 变量 start 值
# =====================================================================
def _resolve_order(inst, routes):
    """沿路线推导 s[j] = max(参与者到达, 前置约束)"""
    task_of = {t.id: t for t in inst.tasks}
    arr = {}
    for u in inst.uavs:
        i = u.id
        t_now = 0.0
        cur = u
        for tid in routes.get(i, []):
            t = task_of[tid]
            t_now += inst.dist(cur, t) / u.speed
            arr[i, tid] = t_now
            t_now += t.dur
            cur = t

    s = {}
    for t in inst.tasks:
        candidates = [0.0]
        for p in t.preds:
            candidates.append(s.get(p, 0) + task_of[p].dur)
        for u in inst.uavs:
            if (u.id, t.id) in arr:
                candidates.append(arr[u.id, t.id])
        s[t.id] = max(candidates)

    arr2 = {}
    for u in inst.uavs:
        i = u.id
        t_now = 0.0
        cur = u
        for tid in routes.get(i, []):
            t = task_of[tid]
            t_now += inst.dist(cur, t) / u.speed
            arr2[i, tid] = t_now
            t_now = max(t_now, s[tid]) + t.dur
            cur = t
    return arr2, s


def _routes_to_names(inst, routes_uav, routes_ugv, arr, s, e_dict):
    """路线 → {变量名字符串: 值}"""
    START, END = -1, -2
    vals = {}

    for u in inst.uavs:
        i = u.id
        seq = routes_uav.get(i, [])
        if not seq:
            vals[f"x_{i}_{START}_{END}"] = 1.0
            continue
        prev = START
        for tid in seq:
            vals[f"x_{i}_{prev}_{tid}"] = 1.0
            prev = tid
        vals[f"x_{i}_{prev}_{END}"] = 1.0

    for g in inst.ugvs:
        gid = g.id
        seq = routes_ugv.get(gid, [])
        if not seq:
            vals[f"x_ugv_{gid}_{START}_{END}"] = 1.0
            continue
        prev = START
        for tid in seq:
            vals[f"x_ugv_{gid}_{prev}_{tid}"] = 1.0
            prev = tid
        vals[f"x_ugv_{gid}_{prev}_{END}"] = 1.0

    for tid, val in s.items():
        vals[f"s_{tid}"] = val
    for (i, tid), val in arr.items():
        vals[f"arr_{i}_{tid}"] = val
    for (i, tid), val in e_dict.items():
        vals[f"E_{i}_{tid}"] = val

    return vals


def greedy_routes(inst):
    """最近邻贪心"""
    ru = {u.id: [] for u in inst.uavs}
    assigned = set()
    for u in inst.uavs:
        cur = u
        while True:
            cand = [(t, inst.dist(cur, t)) for t in inst.tasks if t.id not in assigned]
            if not cand:
                break
            t, _ = min(cand, key=lambda x: x[1])
            ru[u.id].append(t.id)
            assigned.add(t.id)
            cur = t
    rg = {g.id: [] for g in inst.ugvs}
    for g in inst.ugvs:
        for t in inst.tasks:
            if t.need_ugv > 0 and t.id not in [x for v in rg.values() for x in v]:
                rg[g.id].append(t.id)
    return ru, rg


# =====================================================================
# 实验主流程
# =====================================================================
def _collect_from_model(model):
    vals = {}
    for var in model.m.getVars():
        vals[var.VarName] = var.X
    return vals


def _feed_warm_start(model, var_vals):
    cnt = 0
    for var in model.m.getVars():
        if var.VarName in var_vals:
            var.Start = var_vals[var.VarName]
            cnt += 1
    return cnt


def _solve_and_report(model, label, time_limit=60, mip_gap=0.05):
    model.m.Params.TimeLimit = time_limit
    model.m.Params.MIPGap = mip_gap
    # 诊断:热启动是否被加载
    n_start = model.m.NumStart
    n_set = sum(1 for v in model.m.getVars() if v.Start > 0.5)
    print(f"  [{label}] Start加载={n_start}, 已设变量={n_set}")
    t0 = time.time()
    model.m.optimize()
    elapsed = time.time() - t0

    obj = model.m.ObjVal if model.m.SolCount > 0 else None
    gap = model.m.MIPGap if model.m.SolCount > 0 else None
    gap_str = f"{gap*100:.1f}%" if gap is not None else 'N/A'
    print(f"\n  [{label}] makespan={obj:.2f}  " if obj else f"\n  [{label}] 无解  "
          f"耗时={elapsed:.1f}s  节点={model.m.NodeCount}  gap={gap_str}")
    return {'label': label, 'obj': obj, 'time': elapsed,
            'nodes': model.m.NodeCount, 'gap': gap}


def main():
    from opt_model import OptModel

    inst = make_random_instance(n_uav=3, n_ugv=1, n_charger=1, n_task=8, seed=1)
    task_of = {t.id: t for t in inst.tasks}

    # ---- A: 无热启动 ----
    print("\n[A] 无热启动(基线)")
    ma = OptModel(inst)
    ma.build()
    ra = _solve_and_report(ma, "A-无热启动", 120)

    # 从 A 解提取路线/时间(用 Python 引用,不靠字符串)
    opt_routes_uav = {}
    opt_routes_ugv = {}
    opt_arr = {}
    opt_s = {}
    opt_e = {}
    if ma.m.SolCount > 0:
        for u in inst.uavs:
            r = ma._route_of(u.id, ma.x_uav)
            opt_routes_uav[u.id] = [n for n in r if n >= 0]
        for g in inst.ugvs:
            r = ma._route_of(g.id, ma.x_ugv)
            opt_routes_ugv[g.id] = [n for n in r if n >= 0]
        for tid in ma.task_ids:
            opt_s[tid] = ma.s[tid].X
        for u in inst.uavs:
            for tid in opt_routes_uav[u.id]:
                opt_arr[u.id, tid] = ma.arr_uav[u.id, tid].X
                opt_e[u.id, tid] = ma.E[u.id, tid].X

    # ---- B: 最优热启动 ----
    print("\n[B] 最优解热启动")
    mb = OptModel(inst)
    mb.build()
    mb.set_warm_start(opt_routes_uav, opt_routes_ugv, opt_arr, opt_s, opt_e)
    rb = _solve_and_report(mb, "B-最优热启动", 120)

    # ---- C: 贪心热启动 ----
    print("\n[C] 贪心热启动")
    ru, rg = greedy_routes(inst)
    arr, s = _resolve_order(inst, ru)
    e_d = {}
    for u in inst.uavs:
        e = u.e_full
        cur = u
        for tid in ru.get(u.id, []):
            e -= inst.dist(cur, task_of[tid]) + task_of[tid].task_cost
            e_d[u.id, tid] = max(e, u.e_min)
            cur = task_of[tid]
    mc = OptModel(inst)
    mc.build()
    mc.set_warm_start(ru, rg, arr, s, e_d)
    rc = _solve_and_report(mc, "C-贪心热启动", 120)

    # ---- D: 烂解热启动 ----
    print("\n[D] 烂解热启动")
    rng = random.Random(0)
    bad_ru = {}
    for u in inst.uavs:
        subset = rng.sample(range(len(inst.tasks)),
                            rng.randint(1, len(inst.tasks)))
        rng.shuffle(subset)
        bad_ru[u.id] = subset
    bad_rg = {}
    for g in inst.ugvs:
        tids = [t.id for t in inst.tasks if t.need_ugv > 0]
        if tids:
            bad_rg[g.id] = tids[:1]
    bad_arr = {(u.id, t): 0.0 for u in inst.uavs for t in bad_ru.get(u.id, [])}
    bad_s = {t.id: 0.0 for t in inst.tasks}
    bad_e = {(u.id, tid): u.e_full * 0.5 for u in inst.uavs for tid in bad_ru.get(u.id, [])}
    md = OptModel(inst)
    md.build()
    md.set_warm_start(bad_ru, bad_rg, bad_arr, bad_s, bad_e)
    rd = _solve_and_report(md, "D-烂解热启动", 120)

    # ---- E: 部分最优(随机丢 50% 变量,其余不设) ----
    print("\n[E] 部分最优热启动(随机只给 50%)")
    rng = random.Random(42)
    # 随机扔一半 route
    partial_ru = {}
    for uid, seq in opt_routes_uav.items():
        partial_ru[uid] = [t for t in seq if rng.random() > 0.5]
    partial_rg = {}
    for gid, seq in opt_routes_ugv.items():
        partial_rg[gid] = [t for t in seq if rng.random() > 0.5]
    partial_arr = {(i, tid): v for (i, tid), v in opt_arr.items()
                   if rng.random() > 0.5 and tid in partial_ru.get(i, [])}
    partial_s = {tid: v for tid, v in opt_s.items() if rng.random() > 0.5}
    partial_e = {(i, tid): v for (i, tid), v in opt_e.items()
                 if rng.random() > 0.5 and tid in partial_ru.get(i, [])}
    me = OptModel(inst)
    me.build()
    me.set_warm_start(partial_ru, partial_rg, partial_arr, partial_s, partial_e)
    re = _solve_and_report(me, "E-部分最优50%", 120)

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    print(f"{'实验':<14} {'makespan':>10} {'耗时(s)':>8} {'节点':>8} {'Gap':>8}")
    print(f"{'-'*48}")
    for r in [ra, rb, rc, rd, re]:
        ms = f"{r['obj']:.2f}" if r['obj'] is not None else 'N/A'
        gp = f"{r['gap']*100:.1f}%" if r['gap'] is not None else 'N/A'
        print(f"  {r['label']:<14} {ms:>10} {r['time']:>8.1f} {r['nodes']:>8} {gp:>8}")


if __name__ == "__main__":
    main()
