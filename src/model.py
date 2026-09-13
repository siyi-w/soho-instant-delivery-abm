"""
Soho 外卖骑手—行人 ABM 主模型。

流程（每个 step）：
    环境层 → 更新双重压力 → 2×2 决策矩阵 → 三个行为层
    → 移动与冲突检测 → 记录状态

相对 scaffold 版的改动：
1. 环境层接入真实数据（environment.SohoEnvironment），不再用 placeholder
2. 订单按 POI 权重生成，时间预算依路径长度推算，而非均匀随机
3. 拥堵按"路段"而非"节点"统计，符合内生拥堵的设计
4. 行人流量按 TfL PCL 量级设定，随 land_use_mix 在路段间分配
5. 支持 S1/S2 场景与干预开关（micro-hub / loading bay）
6. 冲突事件带坐标，可直接做空间热点分析
"""

from __future__ import annotations

import mesa
from mesa.datacollection import DataCollector

from agents import PedestrianAgent, RiderAgent
from environment import SohoEnvironment
from pedestrian_params import SCENARIOS, FLOW_REFERENCE_PPH
from collections import defaultdict

DWELL_TIME_PER_STOP = 170.0   # 秒，取餐/交付停靠平均耗时
# 依据：Possible (2021) Pedal Me GPS实测，到达-完成交付均值2min50s (N=40,000)
class SohoDeliveryModel(mesa.Model):

    def __init__(
        self,
        env: SohoEnvironment,
        n_riders: int = 40,
        n_pedestrians: int = 300,
        dt: float = 5.0,                    # 秒
        seed: int | None = None,
        scenario: str = "S1_baseline",
        platform_squeeze: float = 0.8,      # 平台承诺时长 / 自由流耗时
        jam_density: float = 1.0,           # 每米路宽多少 agent 视为完全拥堵
        base_ped_flow_pph: float | None = None,
        micro_hubs: set | None = None,      # 干预 A：微枢纽所在节点
        hub_share: float = 0.5,             # 订单起点流向 micro_hubs 的比例（其余仍走分散POI）
        # 混合强度参数：不是"真实数据校准值"，是空间格局反事实的假设强度。
        # 固定的扫描范围：0.0 / 0.25 / 0.5 / 0.75 / 1.0（0=完全不启用，1=全部走集中点），
        # 默认 0.5 保持向后兼容；实际测试 Intervention 2 时应跑完这整组，
        # 而不是只取默认值——结果对这个比例是否敏感，本身就是需要报告的发现
        clear_loading_bays: bool = False,   # 干预 B：装卸区清空
        batch_probability: float = 0.0,     # 打包单比例：0=不启用（默认，行为与之前完全一致）
        # 依据：Intouch Insight (2024/2025) 暗访研究显示美国市场约12%的外卖订单由
        # 顺路带单的骑手完成；Gonzalez-Jimenez et al. (2022, CSCW) 访谈研究显示骑手
        # 普遍反映"现在几乎都是双单"。两项研究量级差异较大、且均非伦敦本地数据，故此处只作为
        # contrast construction参数（方向确认、量级自定），不做精确校准，
        # 具体测试值由实验设计决定，与 S2 天气参数的处理方式一致
        stress_threshold: float = 0.5,              # 2×2矩阵：压力轴阈值
        congestion_threshold: float = 0.5,          # 2×2矩阵：拥堵轴阈值
        stress_trigger_threshold: float = 0.7,      # 路径层：压力触发抄人行道的阈值
        congestion_trigger_threshold: float = 0.8,  # 路径层：拥堵触发抄人行道的阈值
        land_use_threshold: float = 0.8,            # 路径层：land_use_mix强制触发的阈值
        land_use_weight: float = 0.3,               # 2×2矩阵：land_use_mix对阈值的收紧系数
        # adj_threshold = threshold * (1 - land_use_weight * land_use_mix)，
        # 原先是agents.py里decide_baseline_tendency()内部硬编码的magic number，
        # 未暴露、未做敏感性检验；现补上，和上面五个阈值同样处理——
        # 以上六个均为建模假设，不是校准到现实数据的量，单因素敏感性检验见方法论4.7.2
        parking_coverage_weight: float = 1.0,        # 路径层：合法停车覆盖度对land_use_mix的折减权重
        # effective_land_use = land_use_mix * (1 - parking_coverage_weight * parking_coverage)
        # 取代原来"50m内有停车点=完全不触发/50m外=完全触发"的二元判断，
        # 改为连续的距离衰减覆盖度；同样是模型内部构造，靠敏感性检验站住
        congestion_can_force: bool = True,           # 路径消融检验：关闭拥堵触发路径
        land_use_can_force: bool = True,             # 路径消融检验：关闭land_use触发路径
        stress_can_force: bool = True,                # 路径消融检验：关闭压力触发路径
        # 三个开关用于pathway ablation：单独关闭某条路径，检验"某个干预效果
        # 很强"是否只是因为该路径在模型里被赋予了主导地位（见方法论4.8节）
    ):
        super().__init__(rng=seed)

        self.env = env
        self.dt = dt
        self.sim_scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.sim_time = 0.0                 # 已仿真秒数

        # ---------- 决策矩阵/行为层阈值 ----------
        self.stress_threshold = stress_threshold
        self.congestion_threshold = congestion_threshold
        self.stress_trigger_threshold = stress_trigger_threshold
        self.congestion_trigger_threshold = congestion_trigger_threshold
        self.land_use_threshold = land_use_threshold
        self.land_use_weight = land_use_weight
        self.parking_coverage_weight = parking_coverage_weight
        self.congestion_can_force = congestion_can_force
        self.land_use_can_force = land_use_can_force
        self.stress_can_force = stress_can_force

        # ---------- 干预设定 ----------
        self.micro_hubs = set(micro_hubs or [])
        self.hub_share = hub_share
        self.clear_loading_bays = clear_loading_bays
        self.batch_probability = batch_probability

        # ---------- 行人流量基准 ----------
        base = base_ped_flow_pph or FLOW_REFERENCE_PPH["peak_hour"]
        self.base_ped_flow_pph = base * self.sim_scenario.ped_flow_multiplier

        # ---------- 记录 ----------
        self.conflict_events = []
        self.completed_deliveries = []
        self._edge_occupancy = {}           # {(u,v): agent 数}，每步重算

        # ---------- 初始 agent ----------
        self.platform_squeeze = platform_squeeze
        self.jam_density = jam_density

        self.node_reassign_fails = defaultdict(int)
        self.node_congestion_log = defaultdict(list)
        self.route_length_log = []  # 存 (origin_node, route_length)

        for _ in range(n_riders):
            self._spawn_rider()
        n_pedestrians_effective = round(n_pedestrians * self.sim_scenario.ped_flow_multiplier)
        for _ in range(n_pedestrians_effective):
            self._spawn_pedestrian()

        # ---------- 数据收集 ----------
        self.datacollector = DataCollector(
            model_reporters={
                "sim_minutes": lambda m: m.sim_time / 60.0,
                "n_riders_active": lambda m: sum(
                    1 for a in m.agents_by_type[RiderAgent] if not a.finished),
                "mean_stress_level": lambda m: m._mean_attr(RiderAgent, "stress_level"),
                "mean_congestion_index": lambda m: m._mean_attr(RiderAgent, "congestion_index"),
                "pct_on_pavement": lambda m: m._pct_pavement(),
                "n_conflicts_this_step": lambda m: m._conflicts_this_step(),
                "cumulative_conflicts": lambda m: len(m.conflict_events),
                "deliveries_completed": lambda m: len(m.completed_deliveries),
                "pct_overdue": lambda m: m._pct_overdue(),
            },
            agent_reporters={
                "stress_level": lambda a: getattr(a, "stress_level", None),
                "congestion_index": lambda a: getattr(a, "congestion_index", None),
                "route_decision": lambda a: getattr(a, "route_decision", None),
                "route_trigger": lambda a: getattr(a, "route_trigger", None),
                "baseline_tendency": lambda a: getattr(a, "baseline_tendency", None),
            },
        )
        self.datacollector.collect(self)

    # ════════════════════════════════════════════════════════
    # Agent 生成
    # ════════════════════════════════════════════════════════

    def _generate_trip(self):
        env = self.env

        def pick_origin():
            if self.micro_hubs and self.random.random() < self.hub_share:
                return self.random.choice(sorted(self.micro_hubs))
            exclude = self.micro_hubs if self.hub_share > 0 else None
            return env.random_poi_node(self.random, exclude=exclude)

        batch_size = 2 if (self.batch_probability > 0
                           and self.random.random() < self.batch_probability) else 1

        origins = [pick_origin() for _ in range(batch_size)]
        # 送达点按 activity_density（住宅/办公/机构密度）加权，不再是纯均匀随机——
        # 现实里外卖不会均匀送到地图上任意一点，会送到"有人在的地方"
        destinations = [self._weighted_node_choice(env.activity_density) for _ in range(batch_size)]

        waypoints = origins + destinations
        route = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            leg = env.shortest_route(a, b)
            if len(leg) < 2:
                self.node_reassign_fails[origins[0]] += 1
                return None
            route.extend(leg[1:])

        orders = []
        for i in range(batch_size):
            own_leg = env.shortest_route(origins[i], destinations[i])
            if len(own_leg) < 2:
                self.node_reassign_fails[origins[0]] += 1
                return None
            orders.append({
                "time_budget": self._time_budget_for(own_leg),
                "delivered": False,
                "overdue": False,
            })

        route_len = sum(env.edge_length(u, v) for u, v in zip(route, route[1:]))
        self.route_length_log.append((origins[0], route_len))

        stops = [{"node": n, "kind": "pickup", "order_idx": i}
                 for i, n in enumerate(origins)]
        stops += [{"node": n, "kind": "dropoff", "order_idx": i}
                  for i, n in enumerate(destinations)]

        return route, stops, orders
    def _spawn_rider(self):
        trip = self._generate_trip()
        if trip is None:
            return None
        route, stops, orders = trip
        return RiderAgent(self, route, stops, orders, scenario=self.sim_scenario)

    def _spawn_pedestrian(self):
        env = self.env
        # 行人起讫点偏向高 activity_density 区域（住宅/办公/机构密集处人多）
        # —— 注意：不是 land_use_mix（那是餐饮密度，专用于骑手取餐点加权，
        # 混用会导致行人分布被"哪里有餐厅"而不是"哪里有人"决定，见方法论讨论）
        origin = self._weighted_node_choice(env.activity_density)
        destination = self._weighted_node_choice(env.activity_density)
        route = env.shortest_route(origin, destination)
        if len(route) < 2:
            return None
        return PedestrianAgent(self, route)

    def _weighted_node_choice(self, density: dict, default: float = 0.1):
        """按给定的density字典（land_use_mix 或 activity_density）加权抽节点。

        泛化自原来只认 land_use_mix 的版本——现在调用方明确传入用哪个
        密度字典，避免"骑手取餐" "骑手送达" "行人起讫" 这几件语义不同的
        事情，混用同一个变量。
        """
        env = self.env
        nodes = env.nodes
        if not density:
            return self.random.choice(nodes)
        weights = [density.get(n, default) + default for n in nodes]
        total = sum(weights)
        r = self.random.random() * total
        acc = 0.0
        for n, w in zip(nodes, weights):
            acc += w
            if r <= acc:
                return n
        return nodes[-1]

    def respawn_pedestrian(self, ped):
        """行人到达终点后重新投放，维持稳定的人流密度。"""
        env = self.env
        origin = ped.current_node
        destination = self._weighted_node_choice(env.activity_density)
        route = env.shortest_route(origin, destination)
        if len(route) >= 2:
            ped._init_movement(route)

    # ════════════════════════════════════════════════════════
    # 行人流量（TfL PCL 量级）
    # ════════════════════════════════════════════════════════

    def pedestrian_flow_pph(self, edge):
        """
        某路段的行人小时流量 (people per hour)。

        以 TfL High Street 高峰量级为基准，按该路段的 land_use_mix
        在路段间分配 —— 餐饮密集路段人流高，支路人流低。
        """
        u, v = edge
        mix_u = self.env.land_use_mix.get(u, 0.3)
        mix_v = self.env.land_use_mix.get(v, 0.3)
        mix = (mix_u + mix_v) / 2.0
        # 0.15~1.0 的分配系数：即使最冷清的支路也有基本人流
        factor = 0.15 + 0.85 * mix
        return self.base_ped_flow_pph * factor

    # ════════════════════════════════════════════════════════
    # 内生拥堵（按路段统计）
    # ════════════════════════════════════════════════════════

    def _update_congestion(self):
     occ = {}
     for a in self.agents:
        edge = getattr(a, "current_edge", None)
        if edge is not None:
            occ[edge] = occ.get(edge, 0) + 1
     self._edge_occupancy = occ

     for a in self.agents_by_type[RiderAgent]:
        if a.finished:
            continue
        edge = a.current_edge
        if edge is None:
            a.update_congestion_index(0, 1.0)
            continue

        count = occ.get(edge, 1)
        width = self.env.carriageway_width(*edge)

        if self.env.has_loading_bay(edge[0]) and not self.clear_loading_bays:
            width = max(width - 2.0, 0.5)

        a.update_congestion_index(count, width)
        self.node_congestion_log[edge[0]].append(a.congestion_index)   # 加这行
    # ════════════════════════════════════════════════════════
    # 事件记录
    # ════════════════════════════════════════════════════════

    def log_conflict(self, rider, kind, severity="minor"):
        x, y = rider.position_xy()
        self.conflict_events.append({
            "step": self.steps,
            "sim_time_s": self.sim_time,
            "rider_id": rider.unique_id,
            "kind": kind,
            "severity": severity,
            "edge": rider.current_edge,
            "node": rider.current_node,
            "x": x,
            "y": y,
            "stress_level": rider.stress_level,
            "congestion_index": rider.congestion_index,
            "land_use_mix": rider.land_use_mix,
            "route_trigger": rider.route_trigger,
        })

    def reassign_rider(self, rider):
        """
        骑手完成一趟(单单或打包单)后立即接下一趟。

        骑手是持续工作的个体，一个 3 小时班次内完成多单；
        若完成即移除，车队规模会迅速衰减，无法维持稳态。

        若一时找不到可行路径(_generate_trip 返回 None)，本步跳过——
        rider 保持 finished 状态，下一次 step() 会再尝试一次
        (与原实现在这种边界情况下的容错方式一致)。
        """
        trip = self._generate_trip()
        if trip is None:
            rider._finalized = False   # 允许下一步再次尝试派单
            return

        route, stops, orders = trip
        rider.start_new_trip(route, stops, orders, scenario=self.sim_scenario)
        
    def _time_budget_for(self, route):
        env = self.env
        route_len = sum(env.edge_length(u, v) for u, v in zip(route, route[1:]))
        free_flow_time = route_len / RiderAgent.BASE_SPEED
        realistic_time = free_flow_time + DWELL_TIME_PER_STOP
        squeeze = self.platform_squeeze / self.sim_scenario.order_rate_multiplier
        return max(realistic_time * squeeze, 60.0)

    def on_delivery_complete(self, rider, order):
        """
        每送达一单调用一次(打包单会调用多次，每单各一次)。
        不在这里触发派单——整趟(单单或打包单的最后一单)真正结束时，
        由 RiderAgent.step() 在 self.finished 变 True 后调用 reassign_rider。
        """
        self.completed_deliveries.append({
            "rider_id": rider.unique_id,
            "origin_node": rider.stops[0]["node"],
            "elapsed_s": rider.elapsed_time,
            "time_budget_s": order["time_budget"],
            "overdue": order["overdue"],
            "distance_m": rider.distance_travelled,
            "pavement_time_s": rider.pavement_time,
            "red_light_violations": rider.red_light_violations,
            "pavement_conflicts": rider.pavement_conflicts,
            "pavement_risk_exposure": rider.pavement_risk_exposure,
            "severe_conflicts": rider.severe_conflicts,
        })

    # ════════════════════════════════════════════════════════
    # DataCollector 辅助
    # ════════════════════════════════════════════════════════

    def _mean_attr(self, agent_type, attr):
        vals = [getattr(a, attr) for a in self.agents_by_type[agent_type]
                if not getattr(a, "finished", False)]
        return sum(vals) / len(vals) if vals else 0.0

    def _pct_pavement(self):
        active = [a for a in self.agents_by_type[RiderAgent] if not a.finished]
        if not active:
            return 0.0
        n = sum(1 for a in active if a.route_decision == "shortcut_pavement")
        return n / len(active)

    def _pct_overdue(self):
        active = [a for a in self.agents_by_type[RiderAgent]]
        if not active:
            return 0.0
        return sum(1 for a in active if a.overdue) / len(active)

    def _conflicts_this_step(self):
        return sum(1 for e in self.conflict_events if e["step"] == self.steps)

    # ════════════════════════════════════════════════════════
    # 校准诊断
    # ════════════════════════════════════════════════════════

    def quadrant_diagnostics(self):
        """
        检查 2×2 决策矩阵是否四个象限都被激活。

        若某象限占比过低（<5%），说明对应的阈值或 jam_density
        需要重新校准 —— 否则矩阵实际退化，干预也无从体现。
        """
        df = self.datacollector.get_agent_vars_dataframe()
        if "baseline_tendency" not in df.columns or df.empty:
            return {}
        share = df["baseline_tendency"].value_counts(normalize=True).to_dict()
        quadrants = ["full_compliance", "forced_minor_violation",
                     "active_violation", "combined_high_risk"]
        out = {q: share.get(q, 0.0) for q in quadrants}
        out["_congestion_p99"] = float(df["congestion_index"].quantile(0.99))
        out["_degenerate"] = sum(1 for q in quadrants if out[q] < 0.05) >= 2
        return out

    # ════════════════════════════════════════════════════════
    # 主循环
    # ════════════════════════════════════════════════════════

    def step(self):
        # 1. 内生拥堵（需邻域信息，先于 agent step 计算）
        self._update_congestion()

        # 2. 骑手：压力 → 决策矩阵 → 行为层 → 移动
        self.agents_by_type[RiderAgent].shuffle_do("step")

        # 3. 行人移动
        self.agents_by_type[PedestrianAgent].shuffle_do("step")

        # 骑手车队规模固定：完成一单后由 on_delivery_complete 立即派下一单

        self.sim_time += self.dt
        self.datacollector.collect(self)
