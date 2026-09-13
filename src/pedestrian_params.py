"""
行人参数配置 —— 基于 TfL《Pedestrian Comfort Guidance for London》(2010, v2 2019)

用途：为 Soho 外卖配送 ABM 提供有官方出处的行人密度与冲突概率参数。

引用方式：
    Transport for London (2019) Pedestrian Comfort Guidance for London.
    Technical Guide, Version 2. London: TfL. (First edition by Atkins, 2010)
    https://content.tfl.gov.uk/pedestrian-comfort-guidance-technical-guide.pdf

数据基础：TfL 道路网络（TLRN）75+ 站点的 CCTV 观测、行人速度测量、
受限移动记录、通行间距测量与使用者感知问卷。

核心指标 ppmm = pedestrians per metre of clear footway width per minute
    ppmm = 每小时行人数 / 60 / 有效人行道宽度(m)
"""

from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 1. PCL 等级表：拥挤度 → 受限移动比例
# ══════════════════════════════════════════════════════════════
# "受限移动"(Restricted Movement) 官方定义：行人不得不改变速度、改变路线、
# 发生"擦肩"(shoulder brushing) 或与其他使用者碰撞。
# —— 该定义与本研究的"微观空间冲突"高度对应，故直接用作冲突概率基准。

PCL_BANDS = [
    # (等级, ppmm下限, ppmm上限, 受限移动比例)
    ("A+",  0.0,  3.0, 0.03),
    ("A",   3.0,  5.0, 0.13),
    ("A-",  6.0,  8.0, 0.22),
    ("B+",  9.0, 11.0, 0.31),   # ← TfL 推荐的各类区域最低标准
    ("B",  12.0, 14.0, 0.41),
    ("B-", 15.0, 17.0, 0.50),
    ("C+", 18.0, 20.0, 0.59),
    ("C",  21.0, 23.0, 0.69),
    ("C-", 24.0, 26.0, 0.78),
    ("D",  27.0, 35.0, 1.00),
    ("E",  35.0, 999.0, 1.00),
]

# 用于插值的锚点（取各等级区间中值）
_PCL_ANCHORS = [
    (1.5, 0.03), (4.0, 0.13), (7.0, 0.22), (10.0, 0.31),
    (13.0, 0.41), (16.0, 0.50), (19.0, 0.59), (22.0, 0.69),
    (25.0, 0.78), (31.0, 1.00), (40.0, 1.00),
]

PCL_RECOMMENDED = "B+"   # TfL 推荐标准
PCL_MIN_ACCEPTABLE_WITH_CAFE = "C+"
# 指南说明：外摆咖啡座带来的街道活力可以补偿略低的舒适度，
# 但即使如此，高峰时段 PCL 也不应低于 C+。
# → 可作为本研究 al fresco 干预情景的约束条件。


# TfL 原表以整数 ppmm 给出区间（如 A: 3-5, A-: 6-8），
# 区间之间存在空隙。模型中 ppmm 为连续值，故改用连续上界判定。
_PCL_UPPER_BOUNDS = [
    (3.0,  "A+"), (6.0,  "A"),  (9.0,  "A-"), (12.0, "B+"),
    (15.0, "B"),  (18.0, "B-"), (21.0, "C+"), (24.0, "C"),
    (27.0, "C-"), (35.0, "D"),
]


def ppmm_to_pcl(ppmm: float, tfl_rounding: bool = True) -> str:
    """
    拥挤度(ppmm) → PCL 等级

    tfl_rounding=True 时先四舍五入到整数再分级，与 TfL 官方
    计算表的显示与分级方式一致（表中 ppmm 均为整数）。
    设为 False 则按连续值分级，适合模型内部使用。
    """
    v = round(ppmm) if tfl_rounding else ppmm
    for upper, label in _PCL_UPPER_BOUNDS:
        if v < upper:
            return label
    return "E"


def ppmm_to_restricted(ppmm: float) -> float:
    """
    拥挤度(ppmm) → 受限移动比例 [0,1]

    在 TfL 实测锚点之间做线性插值，得到连续函数，
    供 ABM 每个 step 计算行人-骑手冲突概率使用。
    """
    if ppmm <= _PCL_ANCHORS[0][0]:
        return _PCL_ANCHORS[0][1]
    for (x0, y0), (x1, y1) in zip(_PCL_ANCHORS, _PCL_ANCHORS[1:]):
        if x0 <= ppmm <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (ppmm - x0) / (x1 - x0)
    return 1.0


def crowding_ppmm(people_per_hour: float, clear_width_m: float) -> float:
    """TfL 官方拥挤度公式"""
    if clear_width_m <= 0:
        return float("inf")
    return people_per_hour / 60.0 / clear_width_m


# ══════════════════════════════════════════════════════════════
# 2. 有效人行道宽度：标准缓冲与街道家具折减
# ══════════════════════════════════════════════════════════════

BUILDING_EDGE_BUFFER = 0.20   # m，建筑侧标准缓冲
KERB_EDGE_BUFFER     = 0.20   # m，路缘侧标准缓冲
BODY_ELLIPSE_WIDTH   = 0.60   # m，标准人体椭圆宽度
BODY_ELLIPSE_DEPTH   = 0.45   # m，标准人体椭圆深度
MIN_USABLE_GAP       = 0.60   # m，小于此值的残余空间不计入有效宽度
MAX_RESTRICTED_LENGTH = 6.0   # m，DfT 规定的受限人行道最大连续长度

# 街道家具折减规则（单位：m）
# 说明中标注 "buffer" 者为在家具外缘另加的缓冲；
# 标注 "total_reduction" 者为对有效宽度的整体折减量。
FURNITURE_BUFFERS = {
    # —— 与 Soho / 本研究直接相关 ——
    # ⚠️ 墙体类家具（外摆、平行自行车停放、装卸区）：TfL 规定其"视同新的
    #    墙体或路缘"，所需的 200mm 由标准边缘缓冲提供，不重复计算，
    #    故此处 buffer 取 0，只扣除家具自身占用宽度。
    "cafe_seating": {
        "mode": "buffer", "value": 0.0,
        "note": "外摆座椅视同墙体，其 200mm 由标准边缘缓冲提供。"
                "需注意实际占用具有弹性（顾客/商家会加椅子），"
                "且广告牌等附加障碍会进一步压缩宽度。"
                "→ 本研究 al fresco 情景的量化依据",
    },
    "loading_bay_segregated": {
        "mode": "buffer", "value": 0.0,
        "note": "有路缘分隔的装卸区：行人只使用主人行道段，"
                "路缘 200mm 由标准边缘缓冲提供",
    },
    "loading_bay_shared_surface": {
        "mode": "buffer", "value": 0.0,
        "note": "共享路面装卸区：行人倾向使用全宽。"
                "需分别评估【有车停放】与【无车】两种状态。"
                "TfL 明确指出该泊位可能在行人高峰时段运营，"
                "或虽不在运营时段但存在违规使用 "
                "→ 可直接引用以支撑本研究的违规停靠假设",
    },
    "cycle_parking_parallel": {
        "mode": "buffer", "value": 0.0,
        "note": "平行于道路的自行车停放视同墙体，"
                "其 200mm 由标准边缘缓冲提供",
    },
    "cycle_parking_diagonal": {
        "mode": "total_reduction", "value": 2.00,
        "note": "斜向自行车停放，有效宽度折减约 2000mm",
    },
    "cycle_parking_perpendicular": {
        "mode": "total_reduction", "value": 2.50,
        "note": "垂直自行车停放，有效宽度折减约 2500mm",
    },
    # —— 其他常见家具 ——
    "market_stall": {
        "mode": "buffer", "value": 1.40,
        "note": "摊位尺寸外加 1400mm（浏览与排队）；"
                "若双侧开放则为摊位宽度 + 2800mm",
    },
    "individual_vendor": {
        "mode": "buffer", "value": 0.50,
        "note": "单个流动摊贩，摊位尺寸外加 500mm",
    },
    "bench_against_wall": {
        "mode": "buffer", "value": 0.50,
        "note": "座椅宽度外，朝向就座方向另加 500mm（腿、包等）",
    },
    "bench_mid_footway": {
        "mode": "buffer", "value": 0.70,
        "note": "位于人行道中部：就座侧 500mm + 非就座侧 200mm；"
                "若双向就座则为 1000mm",
    },
    "guard_rail": {
        "mode": "buffer", "value": 0.20,
        "note": "护栏外 200mm；部分位置的等候行为会进一步压缩",
    },
    "wayfinding_sign": {
        "mode": "total_reduction", "value": 2.00,
        "note": "地图型指路牌，双面阅读占用约 2m²，"
                "在繁忙站点会显著增加碰撞与绕行",
    },
    "atm": {
        "mode": "buffer", "value": 2.25,
        "note": "排队占用 1500–3000mm，取中值；应经现场核实",
    },
    "tree": {
        "mode": "total_reduction", "value": 0.40,
        "note": "种植区外每侧 200mm",
    },
    "multiple_posts": {
        "mode": "buffer", "value": 0.20,
        "note": "300mm 内多个立柱形成类似护栏的障碍，靠墙或靠路缘时缓冲 200mm",
    },
    "multiple_posts_mid_footway": {
        "mode": "buffer", "value": 0.40,
        "note": "位于人行道中部的立柱群/信号箱：立柱宽度 + 400mm（两侧各 200mm）",
    },
}


def clear_footway_width(
    total_width_m: float,
    furniture: Optional[list] = None,
    has_building_edge: bool = True,
    has_kerb_edge: bool = True,
    unusable_gaps_m: float = 0.0,
) -> float:
    """
    计算有效人行道宽度（TfL 方法）

    参数
    ----
    total_width_m : 人行道总宽
    furniture     : [(家具类型, 家具自身占用宽度m), ...]
                    家具类型须为 FURNITURE_BUFFERS 的键
    unusable_gaps_m : 家具之间小于 0.6m 的残余空间总和（不计入有效宽度）

    注意：若家具紧贴墙体或路缘，则该侧标准缓冲不重复计算，
          应将对应的 has_building_edge / has_kerb_edge 设为 False。
    """
    w = total_width_m
    if has_building_edge:
        w -= BUILDING_EDGE_BUFFER
    if has_kerb_edge:
        w -= KERB_EDGE_BUFFER

    for ftype, own_width in (furniture or []):
        rule = FURNITURE_BUFFERS[ftype]
        if rule["mode"] == "buffer":
            w -= (own_width + rule["value"])
        else:  # total_reduction
            w -= rule["value"]

    w -= unusable_gaps_m
    return max(w, 0.0)


# ══════════════════════════════════════════════════════════════
# 3. 区域类型与流量参考值
# ══════════════════════════════════════════════════════════════
# Soho 归类为 High Street：以零售与餐饮场所为主的区域。
# TfL 界定该类型的行人高峰为周六 14:00–18:00，
# 但明确说明工作日流量通常处于相似水平
# —— 此句为本研究采用工作日傍晚 High Street 量级参数的依据。

AREA_TYPE = "High Street"
AREA_TYPE_PEAK_OFFICIAL = "Saturday 14:00-18:00"
AREA_TYPE_NOTE = "TfL: weekday flows often have similar levels"

# TfL 指南示例中 High Street 站点的流量量级（pph = people per hour）
FLOW_REFERENCE_PPH = {
    "average":            1800,
    "peak_hour":          2800,
    "average_of_max":     5400,   # 最繁忙 10 秒样本的均值，用于压力校核
}

# 推荐最小人行道总宽（含街道家具空间）
RECOMMENDED_TOTAL_WIDTH = {
    "low_flow":    {"pph": "<600",      "width_m": 2.9, "no_furniture_m": 2.6},
    "active_flow": {"pph": "600-1200",  "width_m": 4.2, "no_furniture_m": 3.3},
    "high_flow":   {"pph": ">1200",     "width_m": 5.3, "no_furniture_m": 3.3},
}


# ══════════════════════════════════════════════════════════════
# 4. 场景参数
# ══════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """
    场景设定参数（外生给定，不参与拟合）

    ⚠️ 与校准参数严格区分：stress_level 响应系数、各行为层阈值
       属校准参数，两个场景取值相同，否则构成过拟合。
    """
    name: str
    description: str

    # —— 环境 ——
    visibility: float = 1.0        # 能见度系数 → route / node 层
    friction: float = 1.0          # 路面摩擦系数 → speed 层
    speed_multiplier: float = 1.0  # 骑手基础速度折减

    # —— 需求与人流 ——
    order_rate_multiplier: float = 1.0    # 订单生成率
    ped_flow_multiplier: float = 1.0      # 行人流量
    footway_width_multiplier: float = 1.0 # 有效宽度折减（撑伞、避水洼等）

    # —— 基准行人流量（pph）——
    base_ped_flow_pph: float = FLOW_REFERENCE_PPH["peak_hour"]

    # —— 场景语境（用于记录与论文写作）——
    months: str = ""
    light_condition: str = ""
    weather: str = ""

    def effective_ped_flow(self) -> float:
        return self.base_ped_flow_pph * self.ped_flow_multiplier

    def street_ppmm(self, total_width_m: float, furniture=None, **kw) -> float:
        """给定街道断面，返回本场景下的拥挤度"""
        w = clear_footway_width(total_width_m, furniture, **kw)
        w *= self.footway_width_multiplier
        return crowding_ppmm(self.effective_ped_flow(), w)

    def conflict_probability(self, total_width_m: float, furniture=None, **kw) -> float:
        """给定街道断面，返回本场景下的行人受限移动（冲突）概率"""
        return ppmm_to_restricted(self.street_ppmm(total_width_m, furniture, **kw))


SCENARIOS = {
    "S1_baseline": Scenario(
        name="S1_baseline",
        description="基线场景：用于参数校准与机制验证（场景内部条件一致）",
        visibility=1.0,
        friction=1.0,
        speed_multiplier=1.0,
        order_rate_multiplier=1.0,
        ped_flow_multiplier=1.0,
        footway_width_multiplier=1.0,
        months="September 1-15",
        light_condition="Daylight",
        weather="Dry",
    ),
    "S2_adverse": Scenario(
        name="S2_adverse",
        description="不利场景：用于干预压力测试与设施配置标准",
        visibility=0.65,               # 黑暗（有路灯）
        friction=0.75,                 # 湿滑路面
        speed_multiplier=0.875,
        order_rate_multiplier=1.20,    # 雨天订单上升
        ped_flow_multiplier=0.80,      # 雨天出行减少
        footway_width_multiplier=0.85, # 撑伞、避水洼导致有效宽度下降
        months="November",
        light_condition="Darkness (street lights lit)",
        weather="Rain",
    ),
}

# ⚠️ 注意 order_rate 与 ped_flow 方向相反：雨天订单上升但行人减少。
#    两个反向效应叠加后的净效果无法由推理得出，必须通过模拟确定
#    —— 这是本研究采用 ABM 的核心理由之一。


# ══════════════════════════════════════════════════════════════
# 5. 敏感性分析范围
# ══════════════════════════════════════════════════════════════

SENSITIVITY_RANGES = {
    "base_ped_flow_pph":      [1800, 2800, 5400],   # TfL 三档实测量级
    "order_rate_multiplier":  [0.8, 1.0, 1.2],
    "visibility":             [0.5, 0.65, 0.8, 1.0],
    "friction":               [0.7, 0.85, 1.0],
    "footway_width_multiplier": [0.7, 0.85, 1.0],
}


# ══════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("TfL PCL 参数自检")
    print("=" * 68)

    # 复现指南 p.9 全部四个示例，验证公式与分级函数
    print("\n[验证] 复现 TfL 指南 p.9 示例（预期值取自指南 Print Sheet）")
    cases = [
        # (标签, 总宽, 家具列表, 不可用空隙, 路缘缓冲, 预期有效宽, 预期ppmm, 预期PCL)
        ("A 无家具",   9.7, [],                                 0.0,  True,  9.30, 5,  "A"),
        ("B 多件家具", 8.3, [("cycle_parking_parallel", 2.5),
                             ("multiple_posts_mid_footway", 0.6)], 0.45, True,  3.95, 12, "B"),
        ("C 单件家具", 6.9, [("cycle_parking_parallel", 2.5)],   0.0,  True,  4.00, 12, "B"),
        ("D 全宽",     6.6, [],                                 0.0,  True,  6.20, 8,  "A-"),
    ]
    for label, tw, furn, gap, kerb, exp_w, exp_p, exp_pcl in cases:
        w = clear_footway_width(tw, furn, unusable_gaps_m=gap, has_kerb_edge=kerb)
        p = crowding_ppmm(FLOW_REFERENCE_PPH["peak_hour"], w)
        ok_w = abs(w - exp_w) < 0.02
        ok_p = abs(p - exp_p) < 0.6
        ok_l = ppmm_to_pcl(p) == exp_pcl
        flag = "OK " if (ok_w and ok_p and ok_l) else "!! "
        print(f"  {flag}{label:12s} 有效宽 {w:5.2f}m (预期 {exp_w})  "
              f"{p:5.1f} ppmm (预期 {exp_p})  PCL {ppmm_to_pcl(p):3s} (预期 {exp_pcl})")

    # 分级函数连续性检查
    print("\n[验证] PCL 分级连续性（原表区间存在空隙，须无 E 误判）")
    gaps = [x / 2 for x in range(0, 70)]
    bad = [x for x in gaps if x < 35 and ppmm_to_pcl(x) == "E"]
    print(f"  0–35 ppmm 范围内误判为 E 的取值：{bad if bad else '无'}")

    # Soho 典型窄街：总宽 2.5m，一侧有外摆座椅占 1.0m
    # 注意：窄支路的行人流量远低于 High Street 主街量级，
    #      此处用 800 pph（活跃流量下限）而非 2800 pph
    NARROW_STREET_PPH = 800
    print(f"\n[应用] Soho 典型窄街（总宽 2.5m，外摆座椅占 1.0m，{NARROW_STREET_PPH} pph）")
    for key, sc in SCENARIOS.items():
        furn = [("cafe_seating", 1.0)]
        w = clear_footway_width(2.5, furn) * sc.footway_width_multiplier
        flow = NARROW_STREET_PPH * sc.ped_flow_multiplier
        ppmm = crowding_ppmm(flow, w)
        print(f"  {sc.name:14s} 有效宽 {w:.2f}m  {ppmm:6.1f} ppmm  "
              f"PCL {ppmm_to_pcl(ppmm):3s}  冲突概率 {ppmm_to_restricted(ppmm):.0%}  "
              f"[{sc.light_condition}, {sc.weather}]")

    # 装卸区干预对比
    print(f"\n[干预] 装卸区被占用 vs 清空（总宽 4.0m）")
    sc = SCENARIOS["S1_baseline"]
    for label, furn in [
        ("装卸区被车辆占用", [("loading_bay_shared_surface", 2.0)]),
        ("装卸区清空",       []),
    ]:
        ppmm = sc.street_ppmm(4.0, furn)
        print(f"  {label:18s} {ppmm:6.1f} ppmm  PCL {ppmm_to_pcl(ppmm):3s}  "
              f"冲突概率 {ppmm_to_restricted(ppmm):.0%}")
