"""
运行脚本：加载真实 Soho 数据，跑一次仿真，输出时序曲线与冲突热点图。

命令行参数按四层参数体系分组(见下方 argparse 分组注释)：
    Context      —— 场景/时间窗口，仿真内固定不变
    Experimental —— 活动强度(骑手/行人数量)，视为 activity-intensity 参数，
                     不是现实人口真值；基准范围由外部证据(TfL PCL flow、
                     Little's Law 换算)约束，再用模型自身的非退化/非饱和
                     诊断校验，最终结论稳健性通过单因素敏感性扫描检验
                     （empirically informed，非 empirical calibration）
    Intervention —— 空间干预(停车点/暗厨房/装卸区)与运营干预(打包单)
    Behavioural  —— 机制内部参数(platform_squeeze/jam_density等)，已校准
                     锁定，不通过命令行暴露，改动需回到 model.py 默认值

用法：
    python run.py                              # S1 基线场景
    python run.py --scenario S2_adverse
    python run.py --riders 80 --pedestrians 3000   # 人口敏感性扫描的一个档位
    python run.py --add-parking 15                 # Intervention 1
    python run.py --micro-hubs 5 --hub-share 0.5    # Intervention 2
    python run.py --clear-loading-bays              # 探索性干预(不进正式矩阵)
    python run.py --batch-probability 0.3           # 运营干预：打包单
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from environment import load_environment
from interventions import add_legal_parking, select_micro_hubs
from model import SohoDeliveryModel

# ⚠️ 改成你自己的路径
DATA_DIR = "/Users/wusiyi/Documents/CASA/dis/DATA"

# 人口敏感性扫描的固定档位(方法论里写清楚：不是随意取值，是以当前默认值为
# 中心的数量级括号测试 + Little's Law/地铁客流粗略核对给出的上界)
RIDER_SENSITIVITY_LEVELS = (20, 40, 80, 160)
PEDESTRIAN_SENSITIVITY_LEVELS = (300, 1000, 3000, 8000)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---------------------------------------------------------------
    # Layer 1 — Context parameters：场景/时间窗口，单次仿真内固定不变
    # ---------------------------------------------------------------
    g_context = ap.add_argument_group("Context parameters")
    g_context.add_argument("--scenario", default="S1_baseline",
                            choices=["S1_baseline", "S2_adverse"],
                            help="S1=校准场景(白天/干燥)；S2=干预压力测试场景(天黑/下雨)")
    g_context.add_argument("--minutes", type=float, default=180,
                            help="仿真时长（分钟）。研究场景 17:00-20:00 即 180 分钟")
    g_context.add_argument("--dt", type=float, default=5.0, help="时间步长（秒）")
    g_context.add_argument("--seed", type=int, default=42)
    g_context.add_argument("--data-dir", default=DATA_DIR)

    # ---------------------------------------------------------------
    # Layer 2 — Experimental parameters：活动强度，非现实人口真值
    # ---------------------------------------------------------------
    g_exp = ap.add_argument_group(
        "Experimental (activity-intensity) parameters",
        "基准值(40/300)不是校准到现实密度的结果——这类小范围、特定时段的骑手/"
        "行人密度没有可查的公开数据。用 --riders/--pedestrians 手动指定档位，"
        "配合模型自身的 quadrant_diagnostics()/congestion_index 非退化检查，"
        "做单因素敏感性扫描。"
    )
    g_exp.add_argument("--riders", type=int, default=40,
                        help=f"在途骑手数。敏感性扫描固定档位：{RIDER_SENSITIVITY_LEVELS}")
    g_exp.add_argument("--pedestrians", type=int, default=300,
                        help=f"行人数。敏感性扫描固定档位：{PEDESTRIAN_SENSITIVITY_LEVELS}")

    # ---------------------------------------------------------------
    # Layer 3a — Intervention parameters（空间干预）
    # ---------------------------------------------------------------
    g_spatial = ap.add_argument_group("Spatial intervention parameters")
    g_spatial.add_argument("--add-parking", type=int, default=0,
                            help="Intervention 1：新增合法自行车停车点数量（0=不启用）")
    g_spatial.add_argument("--parking-near-hubs", action="store_true",
                            help="配合--micro-hubs使用：新增的停车点优先覆盖暗厨房集中点"
                                 "周边的缺口，而不是按全局POI权重排序——用于测试"
                                 "'集中化+配套停车'这一组合干预")
    g_spatial.add_argument("--micro-hubs", type=int, default=0,
                            help="Intervention 2：暗厨房集中点数量（0=不启用）")
    g_spatial.add_argument("--hub-share", type=float, default=0.5,
                            help="订单流向集中点的比例，0-1（固定扫描范围：0/0.25/0.5/0.75/1.0）")
    g_spatial.add_argument("--clear-loading-bays", action="store_true",
                            help="探索性干预：装卸区清空（效果有限，不进正式矩阵，仅作背景）")

    # ---------------------------------------------------------------
    # Layer 3b — Intervention parameters（运营干预）
    # ---------------------------------------------------------------
    g_op = ap.add_argument_group("Operational intervention parameters")
    g_op.add_argument("--batch-probability", type=float, default=0.0,
                       help="打包单比例，0-1（0=不启用，默认）")

    # ---------------------------------------------------------------
    # Layer 4 — Behavioural parameters（决策矩阵/路径层阈值，敏感性检验用）
    # ---------------------------------------------------------------
    g_thresh = ap.add_argument_group(
        "Behavioural threshold parameters (sensitivity testing only)",
        "默认值等于已锁定的建模假设，不需要改动即可复现主结果；"
        "单独调整某一个、其余留默认，用于单因素敏感性检验（§4.7.2）"
    )
    g_thresh.add_argument("--stress-threshold", type=float, default=0.5,
                           help="2×2矩阵：压力轴阈值")
    g_thresh.add_argument("--congestion-threshold", type=float, default=0.5,
                           help="2×2矩阵：拥堵轴阈值")
    g_thresh.add_argument("--stress-trigger-threshold", type=float, default=0.7,
                           help="路径层：压力触发抄人行道的阈值")
    g_thresh.add_argument("--congestion-trigger-threshold", type=float, default=0.8,
                           help="路径层：拥堵触发抄人行道的阈值")
    g_thresh.add_argument("--land-use-threshold", type=float, default=0.8,
                           help="路径层：land_use_mix强制触发的阈值")
    g_thresh.add_argument("--jam-density", type=float, default=1.0,
                           help="拥堵公式：每米路宽多少agent视为完全拥堵（congestion_index的分母）；"
                                "此前只在model.py内部有默认值，未暴露为CLI参数，现补上以支持敏感性检验")
    g_thresh.add_argument("--land-use-weight", type=float, default=0.3,
                           help="2×2矩阵：land_use_mix对stress/congestion阈值的收紧系数"
                                "（decide_baseline_tendency里 adj = threshold*(1-weight*land_use_mix)）；"
                                "此前是agents.py里硬编码的无名magic number，现暴露为CLI参数")
    g_thresh.add_argument("--parking-coverage-weight", type=float, default=1.0,
                           help="路径层：合法停车覆盖度(连续值)对land_use_mix的折减权重；"
                                "取代原来'50m内有停车点=完全不触发'的二元判断")

    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "sim_output")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 环境层 ----------
    print("加载环境层...")
    env = load_environment(args.data_dir)

    micro_hubs = None
    if args.micro_hubs > 0:
        micro_hubs = select_micro_hubs(env, n=args.micro_hubs)
        print(f"Intervention 2：选定 {len(micro_hubs)} 个暗厨房集中点"
              f"（hub_share={args.hub_share:g}）")

    if args.add_parking > 0:
        print(f"[调试] 加干预前 cycle_parking_nodes 数量：{len(env.cycle_parking_nodes)}")
        near_nodes = micro_hubs if args.parking_near_hubs else None
        env = add_legal_parking(env, n=args.add_parking, near_nodes=near_nodes)
        print(f"[调试] 加干预后 cycle_parking_nodes 数量：{len(env.cycle_parking_nodes)}")
        mode = "绑定暗厨房集中点周边" if near_nodes else "按全局POI权重排序"
        print(f"Intervention 1：新增 {args.add_parking} 个合法停车点（{mode}）")

    # ---------- 模型 ----------
    print(f"\n初始化模型（场景 {args.scenario}，"
          f"riders={args.riders}, pedestrians={args.pedestrians}）...")
    model = SohoDeliveryModel(
        env=env,
        n_riders=args.riders,
        n_pedestrians=args.pedestrians,
        dt=args.dt,
        seed=args.seed,
        scenario=args.scenario,
        clear_loading_bays=args.clear_loading_bays,
        micro_hubs=micro_hubs,
        hub_share=args.hub_share,
        batch_probability=args.batch_probability,
        stress_threshold=args.stress_threshold,
        congestion_threshold=args.congestion_threshold,
        stress_trigger_threshold=args.stress_trigger_threshold,
        congestion_trigger_threshold=args.congestion_trigger_threshold,
        land_use_threshold=args.land_use_threshold,
        jam_density=args.jam_density,
        land_use_weight=args.land_use_weight,
        parking_coverage_weight=args.parking_coverage_weight,
    )

    n_steps = int(args.minutes * 60 / args.dt)
    print(f"运行 {n_steps} 步（{args.minutes:.0f} 分钟，dt={args.dt}s）...")

    for i in range(n_steps):
        model.step()
        if (i + 1) % max(n_steps // 10, 1) == 0:
            print(f"  {100 * (i + 1) // n_steps:3d}%  "
                  f"冲突累计 {len(model.conflict_events)}  "
                  f"完成配送 {len(model.completed_deliveries)}")

    # ---------- 象限诊断 + 拥堵饱和检查 ----------
    print("\n2×2决策矩阵象限占比:")
    for k, v in model.quadrant_diagnostics().items():
        print(f"  {k}: {v}")

    df_model_check = model.datacollector.get_model_vars_dataframe()
    print("\ncongestion_index 分布(检查是否饱和):")
    print(df_model_check["mean_congestion_index"]
          .describe(percentiles=[.25, .5, .75, .9, .99]).to_string())

    # ---------- 结果 ----------
    tag = args.scenario
    tag += f"_seed{args.seed}"    
    if args.riders != 40:
        tag += f"_riders{args.riders}"
    if args.pedestrians != 300:
        tag += f"_peds{args.pedestrians}"
    if args.clear_loading_bays:
        tag += "_clearbays"
    if args.add_parking > 0:
        tag += f"_parking{args.add_parking}"
    if args.micro_hubs > 0:
        tag += f"_hub{args.micro_hubs}"
        if args.hub_share != 0.5:
            tag += f"_share{args.hub_share:g}"
    if args.batch_probability > 0:
        tag += f"_batch{args.batch_probability:g}"
    if args.stress_threshold != 0.5:
        tag += f"_st{args.stress_threshold:g}"
    if args.congestion_threshold != 0.5:
        tag += f"_ct{args.congestion_threshold:g}"
    if args.stress_trigger_threshold != 0.7:
        tag += f"_stt{args.stress_trigger_threshold:g}"
    if args.congestion_trigger_threshold != 0.8:
        tag += f"_ctt{args.congestion_trigger_threshold:g}"
    if args.land_use_threshold != 0.8:
        tag += f"_lut{args.land_use_threshold:g}"

    df_model = df_model_check
    df_conf = pd.DataFrame(model.conflict_events)
    df_deliv = pd.DataFrame(model.completed_deliveries)

    df_model.to_csv(os.path.join(out_dir, f"timeseries_{tag}.csv"), index=False)
    df_conf.to_csv(os.path.join(out_dir, f"conflicts_{tag}.csv"), index=False)
    df_deliv.to_csv(os.path.join(out_dir, f"deliveries_{tag}.csv"), index=False)

    summarise(df_model, df_conf, df_deliv, tag)
    plot(model, df_model, df_conf, tag, out_dir)
    print(f"\n输出已写入 {out_dir}")


def summarise(df_model, df_conf, df_deliv, tag):
    print("\n" + "=" * 60)
    print(f"场景 {tag}")
    print("=" * 60)
    print(f"冲突事件总数        {len(df_conf)}")
    if len(df_conf):
        for k, v in df_conf["kind"].value_counts().items():
            print(f"    {k:24s} {v}")
    print(f"完成配送            {len(df_deliv)}")
    if len(df_deliv):
        print(f"超时比例            {df_deliv['overdue'].mean():.1%}"
              f"  (机制健康检查，非首要结果指标)")
        print(f"人行道占用均值      {df_deliv['pavement_time_s'].mean():.1f}s")
        conflicts_per_delivery = len(df_conf) / len(df_deliv)
        print(f"每单离散冲突事件数  {conflicts_per_delivery:.2f}  (仅用于空间热点图，量级受dt影响，不作为核心指标)")
        exposure_per_delivery = df_deliv["pavement_risk_exposure"].mean()
        print(f"每单人行道风险暴露量 {exposure_per_delivery:.3f}  (核心结果指标，连续量，不是次数)")
        severe_per_delivery = df_deliv["severe_conflicts"].mean()
        total_severe = df_deliv["severe_conflicts"].sum()
        print(f"每单严重冲突次数    {severe_per_delivery:.3f}  (核心结果指标，PCL D/E级，总数{int(total_severe)})")
    print(f"平均压力水平        {df_model['mean_stress_level'].mean():.3f}")
    print(f"平均拥堵指数        {df_model['mean_congestion_index'].mean():.3f}")
    print(f"人行道骑行比例      {df_model['pct_on_pavement'].mean():.1%}  (核心结果指标)")


def plot(model, df_model, df_conf, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 6))

    # --- 左：时序 ---
    ax1 = fig.add_subplot(1, 2, 1)
    x = df_model["sim_minutes"]
    ax1.plot(x, df_model["mean_stress_level"], label="stress_level", color="#d94801")
    ax1.plot(x, df_model["mean_congestion_index"], label="congestion_index", color="grey")
    ax1.plot(x, df_model["pct_on_pavement"], label="% on pavement", color="#08519c")
    ax1.set_xlabel("Simulated minutes (17:00 - 20:00)")
    ax1.set_ylabel("Value (0-1)")
    ax1.set_title(f"Rider state over time - {tag}")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    # --- 右：冲突热点 ---
    ax2 = fig.add_subplot(1, 2, 2)
    env = model.env
    for u, v in env.graph.edges():
        x0, y0 = env.node_xy.get(u, (None, None))
        x1, y1 = env.node_xy.get(v, (None, None))
        if None not in (x0, y0, x1, y1):
            ax2.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=0.6, zorder=1)

    if len(df_conf):
        ax2.scatter(df_conf["x"], df_conf["y"], s=12, alpha=0.35,
                    color="#d94801", zorder=2, label=f"conflicts (n={len(df_conf)})")
        ax2.legend(loc="upper right")

    ax2.set_aspect("equal")
    ax2.set_title(f"Conflict hotspots - {tag}")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"result_{tag}.png")
    plt.savefig(path, dpi=180)
    print(f"图已保存：{path}")


if __name__ == "__main__":
    main()
