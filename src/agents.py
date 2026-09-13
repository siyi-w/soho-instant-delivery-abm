"""
Soho 外卖骑手—行人 ABM 的 agent 类。

改动要点（相对 scaffold 版）：
1. agent 真正在路网上移动：沿 route 按 speed*dt 推进，可跨路段结算
2. 位置精确到路段内部，冲突可落在路段而非仅交叉口
3. 随机数统一走 self.model.random，保证 seed 可复现
4. 冲突判定接入 TfL PCL：由行人拥挤度推出受限移动概率
5. 时间单位统一为秒，速度单位 m/s

Mesa 3.x API：Agent.__init__(model) 自动注册并分配 unique_id。
"""

from __future__ import annotations

import mesa

from pedestrian_params import (
    clear_footway_width,
    crowding_ppmm,
    ppmm_to_restricted,
)


# ══════════════════════════════════════════════════════════════
# 移动混入：沿路网 route 连续推进
# ══════════════════════════════════════════════════════════════

class NetworkMover:
    """
    位置表示 = (current_node, next_node, dist_into_edge)

    与只记 current_node 相比，agent 在路段中间也有确定位置，
    冲突检测才能落在"路段"上——Soho 的人车冲突主要发生在
    路段中段（骑行上人行道），而非交叉口。
    """

    def _init_movement(self, route):
        self.route = list(route)
        self.route_index = 0
        self.dist_into_edge = 0.0
        self.current_node = self.route[0] if self.route else None
        self.next_node = self.route[1] if len(self.route) > 1 else None
        self.distance_travelled = 0.0
        self.finished = len(self.route) <= 1
        self.just_crossed_node = False

    @property
    def current_edge(self):
        """当前所在路段 (u, v)；已到终点时为 None。"""
        if self.next_node is None:
            return None
        return (self.current_node, self.next_node)

    def move_along(self, distance):
        """沿 route 前进 distance 米，可跨越多个路段。"""
        env = self.model.env
        remaining = distance
        self.just_crossed_node = False

        while remaining > 0 and not self.finished:
            if self.next_node is None:
                self.finished = True
                break

            edge_len = max(env.edge_length(self.current_node, self.next_node), 1e-6)
            left_on_edge = edge_len - self.dist_into_edge

            if remaining < left_on_edge:
                self.dist_into_edge += remaining
                self.distance_travelled += remaining
                remaining = 0.0
            else:
                self.distance_travelled += left_on_edge
                remaining -= left_on_edge
                self.route_index += 1
                self.current_node = self.next_node
                self.dist_into_edge = 0.0
                self.just_crossed_node = True
                if self.route_index + 1 < len(self.route):
                    self.next_node = self.route[self.route_index + 1]
                else:
                    self.next_node = None
                    self.finished = True

    def position_xy(self):
        """当前位置的投影坐标，用于绘图与空间统计。"""
        env = self.model.env
        x0, y0 = env.node_xy.get(self.current_node, (0.0, 0.0))
        if self.next_node is None:
            return x0, y0
        x1, y1 = env.node_xy.get(self.next_node, (x0, y0))
        L = max(env.edge_length(self.current_node, self.next_node), 1e-6)
        t = min(self.dist_into_edge / L, 1.0)
        return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


# ══════════════════════════════════════════════════════════════
# 行人
# ══════════════════════════════════════════════════════════════

class PedestrianAgent(NetworkMover, mesa.Agent):
    """
    行人个体。用于占据人行道空间并参与冲突判定。

    注意：路段级行人流量强度由 model 按 TfL PCL 量级设定
    （见 model.pedestrian_flow_pph），个体 agent 只是抽样代表，
    不需要按真实人数 1:1 生成。
    """

    BASE_SPEED = 1.35          # m/s，成人常速步行

    def __init__(self, model, route, speed=None):
        super().__init__(model)
        self._init_movement(route)
        self.speed = speed if speed is not None else self.BASE_SPEED

    def step(self):
        if self.finished:
            self.model.respawn_pedestrian(self)
            return
        self.move_along(self.speed * self.model.dt)


# ══════════════════════════════════════════════════════════════
# 骑手
# ══════════════════════════════════════════════════════════════

class RiderAgent(NetworkMover, mesa.Agent):
    """
    核心行为 agent。双重压力（stress_level × congestion_index）
    经 2×2 决策矩阵后，触发三个相互独立的行为层：route / node / speed。
    """

    BASE_SPEED = 4.2           # m/s ≈ 15.1 km/h / 9.4mph
# 依据：Allen et al. (2018) 实测伦敦市中心自行车/摩托车配送速度10-15mph，
# 取值接近该范围下限，反映Soho密集街区保守通行条件

    SEVERE_PPMM_THRESHOLD = 27.0
    # 依据：TfL Pedestrian Comfort Guidance PCL 等级表——ppmm=27 是 D 级起点，
    # D/E 级官方受限移动比例均为 1.00("流量完全断流")。用作"严重/真实冲突"
    # 与"轻微受限"的分界，不是自定义阈值，是官方表里已有的分级临界点。

    def __init__(self, model, route, stops, orders, scenario=None):
        """
        route  : 完整物理路径(节点序列)。单单时就是 [起点,...,终点]；
                 打包单时是"取1→[取2]→送1→送2"依次拼接的完整路径。
        stops  : [{"node":, "kind": "pickup"/"dropoff", "order_idx":}, ...]，
                 按到达顺序排列，stops[0] 必是第一个取餐点(等于 route[0])。
        orders : [{"time_budget":, "delivered": False, "overdue": False}, ...]，
                 每单独立的时间预算(按该单自己"取→送"的路径长度算，
                 不含打包带来的绕路)；deliveries_done 之类的生涯统计不放这里。
        """
        super().__init__(model)
        self.deliveries_done = 0     # agent 整个生涯的累计完成单数，不随每趟重置
        self.start_new_trip(route, stops, orders, scenario=scenario)

    def start_new_trip(self, route, stops, orders, scenario=None):
        """开始一趟新行程(单单或打包单)，重置所有"本趟"状态。"""
        self._init_movement(route)

        self.stops = list(stops)
        self.next_stop_idx = 1     # stop 0(第一个取餐点)就是出生位置，下面直接处理
        self.orders = [dict(o) for o in orders]

        env = self.model.env
        self.origin_node = self.route[0] if self.route else None
        self.destination_node = self.route[-1] if self.route else None
        self.parking_coverage = env.legal_parking_coverage(self.origin_node)

        # --- 双重压力 ---
        self.elapsed_time = 0.0
        self.stress_level = 0.0
        self.congestion_index = 0.0
        self.overdue = False

        # --- 环境调节量(第一个取餐点的 land_use_mix；后续取餐点在途中经过时刷新) ---
        self.land_use_mix = env.land_use_mix.get(self.origin_node, 0.5)

        # --- 场景 ---
        self.sim_scenario = scenario

        # --- 行为层状态 ---
        self.route_decision = "compliant"
        self.route_trigger = None
        self.baseline_tendency = "full_compliance"
        self.base_speed = self.BASE_SPEED
        if scenario is not None:
            self.base_speed *= scenario.speed_multiplier
        self.speed = self.base_speed
        self.k = 0.5                                # 速度层压力系数

        # --- 本趟记录(每单送达时会清零，见 _process_stops) ---
        self.red_light_violations = 0
        self.pavement_conflicts = 0                 # 离散事件计数，仅用于空间热点图
        self.pavement_risk_exposure = 0.0           # 连续暴露量累加，核心结果指标用这个
        self.severe_conflicts = 0                   # PCL D/E级严重冲突次数，核心结果指标
        self.pavement_time = 0.0                    # 占用人行道累计时长(s)

        self._finalized = False                     # 整趟完成→重新派单，只触发一次

    # ------------------------------------------------------------------
    # 双重压力
    # ------------------------------------------------------------------
    def update_stress_level(self):
        """
        压力水平由"当前尚未送达的订单里，剩余时间比例最小的那一单"驱动——
        打包单场景下，骑手的紧迫感来自最快要超时的那一单，不是均值。
        """
        open_orders = [o for o in self.orders if not o["delivered"]]
        if not open_orders:
            self.stress_level = 0.0
            self.overdue = False
            return

        best_remaining_ratio = 1.0
        any_overdue = False
        for o in open_orders:
            budget = max(o["time_budget"], 1e-6)
            remaining = max(budget - self.elapsed_time, 0.0)
            if remaining <= 0:
                o["overdue"] = True
            if o["overdue"]:
                any_overdue = True
            best_remaining_ratio = min(best_remaining_ratio, remaining / budget)

        self.stress_level = min(max(1 - best_remaining_ratio, 0.0), 1.0)
        self.overdue = any_overdue

    def update_congestion_index(self, local_agent_count, effective_width,
                                jam_density=None):
        """
        内生拥堵：由 agent 密度与有效路宽算出，不依赖外部交通流数据。

        jam_density = 每米路宽上多少个 agent 视为完全拥堵。

        ⚠️ 这是关键校准参数。模型中的 agent 是真实人流的抽样代表，
        并非 1:1，故 jam_density 必须校准到使决策矩阵四象限都被激活；
        取值过高会让 congestion 永远低于阈值，2×2 矩阵退化为 1×2。
        用 model.quadrant_diagnostics() 检查。
        """
        jam = jam_density if jam_density is not None else self.model.jam_density
        if effective_width <= 0:
            self.congestion_index = 1.0
        else:
            density = local_agent_count / effective_width
            self.congestion_index = min(density / jam, 1.0)

    # ------------------------------------------------------------------
    # 2×2 决策矩阵（阈值受 land_use_mix 调节）
    # ------------------------------------------------------------------
    def decide_baseline_tendency(self, stress_threshold=0.5, congestion_threshold=0.5,
                                 land_use_weight=0.3):
        adj_stress = stress_threshold * (1 - land_use_weight * self.land_use_mix)
        adj_congestion = congestion_threshold * (1 - land_use_weight * self.land_use_mix)

        low_stress = self.stress_level < adj_stress
        low_congestion = self.congestion_index < adj_congestion

        if low_stress and low_congestion:
            return "full_compliance"
        if low_stress and not low_congestion:
            return "forced_minor_violation"
        if not low_stress and low_congestion:
            return "active_violation"
        return "combined_high_risk"

    # ------------------------------------------------------------------
    # 行为层（三层相互独立触发）
    # ------------------------------------------------------------------
    def apply_route_layer(self, baseline_tendency,
                          stress_trigger_threshold=0.7,
                          congestion_trigger_threshold=0.8,
                          land_use_threshold=0.8,
                          parking_coverage_weight=1.0,
                          congestion_can_force=True,
                          land_use_can_force=True,
                          stress_can_force=True):
        """
        路径层：是否借用人行道。三条独立的触发路径：

          1. land_use_mix 过高 —— 合法路缘空间已被占满，无处可停
          2. 压力触发 —— 主动违规（active_violation / combined_high_risk）
          3. 拥堵触发 —— 车行道堵死，被迫绕行人行道

        三条路径各自有独立开关(congestion_can_force/land_use_can_force/
        stress_can_force)，用于路径消融检验(pathway ablation)：关闭某条
        路径后重新跑一遍，看结果对哪条路径最敏感——用来检验"某个干预效果
        很强"是不是因为该路径本身在模型里就被赋予了主导地位，而不是三条
        路径贡献均衡下的真实涌现结果。见方法论4.8节。

        第 1 条不再是"50m内有合法停车=完全不触发，50m外=完全触发"的
        二元判断，改用 parking_coverage（连续值，0=完全没有覆盖，1=就在
        停车点旁边）按 parking_coverage_weight 折减 land_use_mix 的
        有效值——覆盖度越高，land_use触发越不容易达标，但不是非黑即白。
        parking_coverage_weight 和 land_use_weight 是同一类模型内部构造，
        没有直接现实依据，靠敏感性检验站住（见方法论4.5节）。
        """
        effective_land_use = self.land_use_mix * (1 - parking_coverage_weight * self.parking_coverage)
        forced_by_land_use = (
            land_use_can_force
            and effective_land_use > land_use_threshold
        )
        triggered_by_stress = (
            stress_can_force
            and self.stress_level >= stress_trigger_threshold
            and baseline_tendency in ("active_violation", "combined_high_risk")
        )
        forced_by_congestion = (
            congestion_can_force
            and self.congestion_index >= congestion_trigger_threshold
        )

        if forced_by_land_use or triggered_by_stress or forced_by_congestion:
            self.route_decision = "shortcut_pavement"
            self.route_trigger = (
                "land_use" if forced_by_land_use else
                "congestion" if forced_by_congestion else "stress"
            )
        else:
            self.route_decision = "compliant"
            self.route_trigger = None

    def apply_node_layer(self, base_rate=0.05):
        """节点层：每次经过交叉口独立抽签，P(闯红灯)。"""
        if not self.just_crossed_node:
            return False
        p = base_rate * (1 + 2 * self.stress_level)
        if self.sim_scenario is not None:
            p *= (2 - self.sim_scenario.visibility)     # 能见度差 → 风险感知下降
        return self.model.random.random() < min(p, 1.0)

    def apply_speed_layer(self, congestion_drag=0.7):
        """
        速度层：连续倍率，非离散档位。

        两个方向相反的作用：
          压力 ↑ → 骑手加速（算法压力的直接体现）
          拥堵 ↑ → 实际可达速度下降

        拥堵项不可省略：没有它，内生拥堵不会影响任何结果，
        "拥堵 → 耗时增加 → 压力上升 → 违规"这条反馈环不成立。
        """
        stress_boost = 1 + self.k * self.stress_level
        congestion_penalty = 1 - congestion_drag * self.congestion_index
        self.speed = self.base_speed * stress_boost * max(congestion_penalty, 0.15)
        if self.sim_scenario is not None:
            self.speed *= (0.5 + 0.5 * self.sim_scenario.friction)

    # ------------------------------------------------------------------
    # 人行道风险暴露量（TfL PCL）
    # ------------------------------------------------------------------
    def pavement_conflict_exposure(self):
        """
        骑手占用人行道时，本时间步内造成的"行人受限移动"预期人数。

        依据 TfL Pedestrian Comfort Guidance：
            ppmm = 行人每小时流量 / 60 / 有效人行道宽度
            受限移动比例 = PCL 曲线(ppmm)
        骑手自身占据的宽度先从有效宽度中扣除。

        返回 (exposure, ppmm)：
          exposure：连续期望值(预期受限人数)，不封顶——用于核心指标
                    pavement_risk_exposure（轻微到严重全谱的暴露量代理）
          ppmm：当前路段拥挤度，用于在 step() 里判定是否达到 PCL D/E 级
                （ppmm≥27，TfL 官方表里"流量完全断流、受限比例=1.00"的临界点），
                作为"严重/真实冲突"（severe_conflicts）的判定依据
        """
        edge = self.current_edge
        if edge is None:
            return 0.0, 0.0

        env = self.model.env
        total_w = env.footway_width(*edge)
        if total_w <= 0:
            return 0.0, 0.0

        clear_w = clear_footway_width(total_w)
        if self.sim_scenario is not None:
            clear_w *= self.sim_scenario.footway_width_multiplier

        RIDER_OCCUPANCY = 0.75          # m，骑手＋车把占用宽度
        clear_w = max(clear_w - RIDER_OCCUPANCY, 0.05)

        flow_pph = self.model.pedestrian_flow_pph(edge)
        ppmm = crowding_ppmm(flow_pph, clear_w)
        p_restricted = ppmm_to_restricted(ppmm)

        # PCL 给的是"每名行人受限的比例"，按本时间步遭遇的行人数折算
        expected_peds = flow_pph / 3600.0 * self.model.dt
        return p_restricted * expected_peds, ppmm  # 不夹到1.0，保留原始期望值


    # ------------------------------------------------------------------
    # 到站处理(取餐 / 送达)
    # ------------------------------------------------------------------
    def _process_stops(self):
        """
        移动之后检查是否到达了下一个待处理的停靠点(取餐或送达)。
        一步内移动距离较大时，可能连续经过多个停靠点，故用 while。
        """
        env = self.model.env
        while (self.next_stop_idx < len(self.stops)
               and self.current_node == self.stops[self.next_stop_idx]["node"]):
            stop = self.stops[self.next_stop_idx]

            if stop["kind"] == "pickup":
                # 打包单的第二个取餐点：刷新落脚处的 land_use_mix / 合法停车覆盖度，
                # 因为路缘空间竞争是"当前所在位置"的属性，不是出发点固定不变的
                self.land_use_mix = env.land_use_mix.get(stop["node"], 0.5)
                self.parking_coverage = env.legal_parking_coverage(stop["node"])
            else:  # dropoff
                # 送达点同样刷新——之前这里完全没有更新land_use_mix/停车覆盖度，
                # 意味着骑手在送餐环节的路缘竞争压力被忽略了，只反映了最近一次
                # 取餐点的状态。现在送达时也按"当前位置"重新评估
                self.land_use_mix = env.land_use_mix.get(stop["node"], 0.5)
                self.parking_coverage = env.legal_parking_coverage(stop["node"])
                order = self.orders[stop["order_idx"]]
                order["delivered"] = True
                self.model.on_delivery_complete(self, order)
                self.deliveries_done += 1
                # 这一单送完了，"本单范围"的统计清零，下一单从零开始计
                # (打包单场景下这几个字段代表"这一单送达前发生的事"，不是整趟累计)
                self.pavement_time = 0.0
                self.pavement_conflicts = 0
                self.pavement_risk_exposure = 0.0
                self.severe_conflicts = 0
                self.red_light_violations = 0
                self.distance_travelled = 0.0

            self.next_stop_idx += 1

    # ------------------------------------------------------------------
    # 主步骤
    # ------------------------------------------------------------------
    def step(self):
        if self.finished:
            # 整趟(单单或打包单的最后一单)已送完，接下一趟；只触发一次
            if not self._finalized:
                self._finalized = True
                self.model.reassign_rider(self)
            return

        # 1. 双重压力（congestion_index 已由 model 在本步之前写入）
        self.update_stress_level()

        # 2. 决策矩阵
        self.baseline_tendency = self.decide_baseline_tendency(
            stress_threshold=self.model.stress_threshold,
            congestion_threshold=self.model.congestion_threshold,
            land_use_weight=self.model.land_use_weight,
        )

        # 3. 三个行为层
        self.apply_route_layer(
            self.baseline_tendency,
            stress_trigger_threshold=self.model.stress_trigger_threshold,
            congestion_trigger_threshold=self.model.congestion_trigger_threshold,
            land_use_threshold=self.model.land_use_threshold,
            parking_coverage_weight=self.model.parking_coverage_weight,
            congestion_can_force=self.model.congestion_can_force,
            land_use_can_force=self.model.land_use_can_force,
            stress_can_force=self.model.stress_can_force,
        )
        ran_red_light = self.apply_node_layer()
        self.apply_speed_layer()

        # 4. 违规与冲突记录
        if ran_red_light:
            self.red_light_violations += 1
            self.model.log_conflict(self, kind="red_light_violation", severity="minor")

        if self.route_decision == "shortcut_pavement":
            self.pavement_time += self.model.dt
            exposure, ppmm = self.pavement_conflict_exposure()
            self.pavement_risk_exposure += exposure   # 核心指标：连续累加，不封顶
            if self.model.random.random() < min(exposure, 1.0):
                self.pavement_conflicts += 1            # 离散事件计数：仅用于空间热点图
                self.model.log_conflict(self, kind="pavement_conflict", severity="minor")
            if ppmm >= self.SEVERE_PPMM_THRESHOLD:
                # PCL D/E 级：官方受限比例已达1.00，不再抽签，直接计一次严重冲突
                self.severe_conflicts += 1
                self.model.log_conflict(self, kind="severe_pavement_conflict", severity="severe")

        # 5. 移动
        self.move_along(self.speed * self.model.dt)
        self.elapsed_time += self.model.dt

        # 6. 到站处理：可能这一步内正好取到/送到，记录并推进 stop 指针
        self._process_stops()
