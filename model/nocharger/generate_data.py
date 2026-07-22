"""
数据生成脚本:多线程并行求解多样化算例,保存完整训练数据。
运行: python model/nocharger/generate_data.py
输出: data/YYYYMMDD/data_XXXXXX.pkl (每个算例) + summary.json
"""

import json, time, os, sys, pickle, random
from datetime import datetime
from multiprocessing import Pool, cpu_count

def _setup_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for p in (here, root):
        if p not in sys.path:
            sys.path.insert(0, p)

_setup_path()
from core.entities import UAV, UGV, Task, Instance

# ====================== 配置 ======================
CONFIG_GROUPS = [
    ("S",   (2, 3),  (1, 2),  (3, 5),  50.0),
    ("M",   (3, 5),  (1, 2),  (5, 7),  100.0),
    ("L",   (5, 8),  (2, 3),  (7, 10), 150.0),
]

N_SAMPLES = 2
TIME_LIMIT = 120
MIP_GAP = 0.01
RNG_SEED = 42
N_WORKERS = max(1, cpu_count() // 2)
RESUME = True              # 断点续传:已有 .pkl 则跳过


# ====================== 实例生成 ======================
def generate_diverse_instance(seed):
    rng = random.Random(seed)
    label, (n_uav_lo, n_uav_hi), (n_ugv_lo, n_ugv_hi), \
        (n_task_lo, n_task_hi), area = rng.choice(CONFIG_GROUPS)
    n_uav = rng.randint(n_uav_lo, n_uav_hi)
    n_ugv = rng.randint(n_ugv_lo, n_ugv_hi)
    n_task = rng.randint(n_task_lo, n_task_hi)
    e_full = area * 2

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
        tasks.append(Task(id=j, x=rng.uniform(0, area), y=rng.uniform(0, area),
                          dur=dur, need_uav=need_uav, need_ugv=need_ugv,
                          task_cost=rng.uniform(3, 10)))

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


# ====================== Worker 求解 ======================
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
        return {"seed": seed, "error": "no_solution", "config": config,
                "solve_time": time.time() - t0}

    # ---- 提取全部变量值 ----
    # 二进制路径变量
    x_uav = {}
    for (i, a, b), var in model.x_uav.items():
        x_uav[f"UAV{i}_{a}_{b}"] = int(round(var.X))
    x_ugv = {}
    for (g, a, b), var in model.x_ugv.items():
        x_ugv[f"UGV{g}_{a}_{b}"] = int(round(var.X))

    # 连续变量
    arr_uav = {f"UAV{i}_T{j}": round(v.X, 4) for (i, j), v in model.arr_uav.items()}
    arr_ugv = {f"UGV{g}_T{j}": round(v.X, 4) for (g, j), v in model.arr_ugv.items()}
    s_vals = {f"T{tid}": round(v.X, 4) for tid, v in model.s.items()}
    e_vals = {f"UAV{i}_T{j}": round(v.X, 4) for (i, j), v in model.E.items()}

    # 路线(精简版,方便快速读)
    route_uav = {}
    for uav in inst.uavs:
        r = model._route_of(uav.id, model.x_uav)
        route_uav[uav.id] = [n for n in r if n >= 0]
    route_ugv = {}
    for ugv in inst.ugvs:
        r = model._route_of(ugv.id, model.x_ugv)
        route_ugv[ugv.id] = [n for n in r if n >= 0]

    # 任务特征(GNN 输入)
    task_feat = []
    for t in inst.tasks:
        task_feat.append({
            "id": t.id, "x": round(t.x, 2), "y": round(t.y, 2),
            "dur": round(t.dur, 2), "need_uav": t.need_uav,
            "need_ugv": t.need_ugv, "preds": t.preds,
            "task_cost": round(t.task_cost, 2),
            "s": round(model.s[t.id].X, 4),      # 实际开始时间
        })

    elapsed = round(time.time() - t0, 2)

    # ---- 组装完整数据包 ----
    data = {
        # ---- 输入特征 ----
        "config": config,
        "task_feat": task_feat,
        "uav_speed": [round(u.speed, 2) for u in inst.uavs],
        "uav_e_full": [u.e_full for u in inst.uavs],
        "ugv_speed": [round(g.speed, 2) for g in inst.ugvs],

        # ---- 标签/输出 ----
        "makespan": round(model.makespan.X, 4),
        "task_start": {t.id: round(model.s[t.id].X, 4) for t in inst.tasks},
        "route_uav": route_uav,
        "route_ugv": route_ugv,

        # ---- 完整变量(全量,用于高级训练) ----
        "x_uav": x_uav,
        "x_ugv": x_ugv,
        "arr_uav": arr_uav,
        "arr_ugv": arr_ugv,
        "s": s_vals,
        "E": e_vals,

        # ---- 求解元信息 ----
        "seed": seed,
        "solve_time": elapsed,
        "gap": round(model.m.MIPGap, 4),
        "status": model.m.Status,
        "nodes": model.m.NodeCount,
    }

    return data


# ====================== 主函数 ======================
def main():
    # 日期标签目录(精确到秒,每次运行独立)
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", date_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=" * 55)
    print(f"  数据生成  |  输出: {out_dir}")
    print(f"  算例数: {N_SAMPLES}  |  进程: {N_WORKERS}")
    print(f"=" * 55)
    t_start = time.time()

    # 1. 生成所有实例
    print(f"\n[1/3] 生成 {N_SAMPLES} 个实例...")
    instances = []
    for idx in range(N_SAMPLES):
        seed = RNG_SEED + idx
        inst, config = generate_diverse_instance(seed)
        instances.append((inst, config, seed))

    # 2. 并行求解,完成一个立刻保存
    print(f"[2/3] 并行求解 ({N_WORKERS} workers, 边跑边存)...")
    t0 = time.time()
    solved, failed, skipped = 0, 0, 0
    err_count = {}
    summary = []

    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_solve_one, instances):
            seed = r.get("seed", -1)
            if "error" in r:
                failed += 1
                reason = r["error"][:40]
                err_count[reason] = err_count.get(reason, 0) + 1
                continue

            fname = f"data_{seed:06d}.pkl"
            fpath = os.path.join(out_dir, fname)

            if RESUME and os.path.exists(fpath):
                skipped += 1
                solved += 1
                continue

            solved += 1
            with open(fpath, "wb") as f:
                pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)

            summary.append({
                "seed": seed, "config": r["config"],
                "makespan": r["makespan"],
                "solve_time": r["solve_time"], "gap": r["gap"],
                "route_uav": r["route_uav"],
            })
            print(f"  V seed={seed:6d}  ms={r['makespan']:.1f}"
                  f"  [{solved}/{N_SAMPLES}]")

    # 3. 写汇总 json
    print(f"\n[3/3] 保存 summary.json ...")
    sm_path = os.path.join(out_dir, "summary.json")
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n保存到 {out_dir}/")
    print(f"  成功: {solved}  失败/无解: {failed}"
          + (f"  跳过: {skipped}" if skipped else ""))
    print(f"  总耗时: {time.time() - t_start:.0f}s")

    # 4. 写运行日志
    total_elapsed = time.time() - t_start
    ms_list = [s["makespan"] for s in summary if "makespan" in s]
    log_path = os.path.join(out_dir, "run.log")
    with open(log_path, "w") as f:
        f.write(f"=== 数据生成运行日志 ===\n")
        f.write(f"开始时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时:     {total_elapsed:.0f}s\n")
        f.write(f"工作进程:   {N_WORKERS}\n")
        f.write(f"时间限制:   {TIME_LIMIT}s/实例\n")
        f.write(f"MIP gap:    {MIP_GAP}\n\n")
        f.write(f"--- 结果统计 ---\n")
        f.write(f"总算例:     {N_SAMPLES}\n")
        f.write(f"成功:       {solved}\n")
        f.write(f"失败:       {failed}\n")
        if skipped: f.write(f"跳过(已有): {skipped}\n")
        f.write(f"成功率:     {solved/(solved+failed)*100:.1f}%" if solved+failed else "N/A")
        f.write(f"\n\n--- 失败原因 ---\n")
        for reason, cnt in sorted(err_count.items(), key=lambda x: -x[1]):
            f.write(f"  {reason:<20} x{cnt}\n")
        if ms_list:
            f.write(f"\n--- makespan 统计 ---\n")
            f.write(f"  min:    {min(ms_list):.1f}\n")
            f.write(f"  max:    {max(ms_list):.1f}\n")
            f.write(f"  mean:   {sum(ms_list)/len(ms_list):.1f}\n")
            f.write(f"  median: {sorted(ms_list)[len(ms_list)//2]:.1f}\n")
        solve_times = [s.get("solve_time", 0) for s in summary]
        if solve_times:
            f.write(f"\n--- 求解时间统计 (s) ---\n")
            f.write(f"  min:    {min(solve_times):.1f}\n")
            f.write(f"  max:    {max(solve_times):.1f}\n")
            f.write(f"  mean:   {sum(solve_times)/len(solve_times):.1f}\n")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=N_WORKERS)
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    args = p.parse_args()
    N_WORKERS = args.workers
    N_SAMPLES = args.samples
    main()
