"""
数据生成脚本:多线程并行求解多样化调度算例,保存 GNN 训练数据。
运行: python model/nocharger/generate_data.py
线程数: 自动 = CPU 核数/2,可--workers 手动指定
"""

import json, time, os, sys, random
from multiprocessing import Pool, cpu_count

# path for both main and worker processes
def _setup_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for p in (here, root):
        if p not in sys.path:
            sys.path.insert(0, p)

_setup_path()
from core.entities import UAV, UGV, Task, Instance

# ====================== 配置组 ======================
CONFIG_GROUPS = [
    ("XS",  (2, 2),  (0, 1),  (3, 5),   30.0),
    ("S",   (2, 3),  (0, 1),  (5, 10),  50.0),
    ("M",   (3, 5),  (1, 2),  (10, 18), 100.0),
    ("L",   (5, 8),  (1, 3),  (18, 30), 150.0),
]

OUT_DIR = "data"
N_SAMPLES = 200
TIME_LIMIT = 120
MIP_GAP = 0.01
RNG_SEED = 42
N_WORKERS = max(1, cpu_count() // 2)  # 默认一半核


# ====================== 实例生成 ======================
def generate_diverse_instance(seed):
    rng = random.Random(seed)
    label, (n_uav_lo, n_uav_hi), (n_ugv_lo, n_ugv_hi), \
        (n_task_lo, n_task_hi), area = rng.choice(CONFIG_GROUPS)
    n_uav = rng.randint(n_uav_lo, n_uav_hi)
    n_ugv = rng.randint(n_ugv_lo, n_ugv_hi)
    n_task = rng.randint(n_task_lo, n_task_hi)
    e_full = area * n_task * 3.0

    inst = Instance()
    inst.uavs = [
        UAV(id=i, x=0.0, y=0.0, speed=rng.uniform(1.5, 2.5),
            e_full=e_full, e_min=0.0)
        for i in range(n_uav)
    ]
    inst.ugvs = [
        UGV(id=g, x=rng.uniform(0, area), y=rng.uniform(0, area),
            speed=rng.uniform(0.8, 1.2))
        for g in range(n_ugv)
    ]

    tasks = []
    for j in range(n_task):
        dur = rng.uniform(3, 10)
        need_uav = rng.choices([1, 1, 1, 2, 2, 3], weights=[5, 4, 3, 2, 1, 1])[0]
        need_uav = min(need_uav, n_uav)
        need_ugv = 0
        if n_ugv > 0:
            need_ugv = rng.choices([0, 0, 0, 0, 1, 1, 1], weights=[5, 4, 3, 2, 2, 1, 1])[0]
        tasks.append(Task(
            id=j, x=rng.uniform(0, area), y=rng.uniform(0, area),
            dur=dur, need_uav=need_uav, need_ugv=need_ugv,
            task_cost=rng.uniform(3, 10),
        ))

    all_ids = list(range(n_task))
    rng.shuffle(all_ids)
    n_prec = int(n_task * 0.3)
    for k in range(0, n_prec * 2, 2):
        if k + 1 >= len(all_ids): break
        a, b = all_ids[k], all_ids[k + 1]
        if b not in tasks[a].preds and a not in tasks[b].preds:
            tasks[b].preds.append(a)

    inst.tasks = tasks
    config = {"label": label, "n_uav": n_uav, "n_ugv": n_ugv,
              "n_task": n_task, "area": area}
    return inst, config


# ====================== 单实例求解(worker进程) ======================
def _solve_one(args):
    inst, config, seed = args
    _setup_path()
    from opt_model import OptModel

    t0 = time.time()
    try:
        model = OptModel(inst)
        model.build()
        model.solve(time_limit=TIME_LIMIT, mip_gap=MIP_GAP)
    except Exception as e:
        return {"seed": seed, "error": str(e), "solve_time": time.time() - t0}

    if model.m.SolCount == 0:
        return {"seed": seed, "error": "no_solution", "solve_time": time.time() - t0}

    route_uav = {}
    for uav in inst.uavs:
        r = model._route_of(uav.id, model.x_uav)
        route_uav[uav.id] = [n for n in r if n >= 0]
    route_ugv = {}
    for ugv in inst.ugvs:
        r = model._route_of(ugv.id, model.x_ugv)
        route_ugv[ugv.id] = [n for n in r if n >= 0]

    task_feat = []
    for t in inst.tasks:
        task_feat.append({
            "id": t.id, "x": round(t.x, 2), "y": round(t.y, 2),
            "dur": round(t.dur, 2), "need_uav": t.need_uav,
            "need_ugv": t.need_ugv, "preds": t.preds,
            "task_cost": round(t.task_cost, 2),
        })

    return {
        "seed": seed, "config": config,
        "makespan": round(model.makespan.X, 4),
        "task_start": {t.id: round(model.s[t.id].X, 4) for t in inst.tasks},
        "route_uav": route_uav, "route_ugv": route_ugv,
        "task_feat": task_feat,
        "uav_speed": [round(u.speed, 2) for u in inst.uavs],
        "ugv_speed": [round(g.speed, 2) for g in inst.ugvs],
        "solve_time": round(time.time() - t0, 2),
        "gap": round(model.m.MIPGap, 4) if model.m.SolCount > 0 else None,
    }


# ====================== 主函数 ======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 先生成所有实例
    print(f"生成 {N_SAMPLES} 个多样化实例...")
    instances = []
    for idx in range(N_SAMPLES):
        seed = RNG_SEED + idx
        inst, config = generate_diverse_instance(seed)
        instances.append((inst, config, seed))

    # 2. 并行求解
    print(f"使用 {N_WORKERS} 个工作进程并行求解...")
    with Pool(N_WORKERS) as pool:
        results_raw = pool.map(_solve_one, instances)

    # 3. 汇总
    results = []
    solved, failed = 0, 0
    for r in results_raw:
        if "error" in r:
            failed += 1
        else:
            results.append(r)
            solved += 1

    # 4. 保存
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n保存到 {out_path}")
    print(f"成功: {solved}  失败/无解: {failed}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=N_WORKERS, help="并行进程数")
    p.add_argument("--samples", type=int, default=N_SAMPLES, help="总算例数")
    args = p.parse_args()
    N_WORKERS = args.workers
    N_SAMPLES = args.samples
    main()
