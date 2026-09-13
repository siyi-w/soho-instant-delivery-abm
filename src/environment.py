"""
环境层 —— 把磁盘上的真实 Soho 数据装配成模型可用的对象。

替代 model.py 中的 _make_placeholder_* 系列函数。

输入文件（相对 DATA_DIR）：
    raw/soho_bike_network.graphml   骑手路网
    raw/soho_walk_network.graphml   行人路网（可选）
    soho_food_pois.csv              餐饮 POI（订单生成点）
    raw/soho_loading_bays.geojson   装卸区（干预情景用，可选）
    raw/soho_cycle_parking.geojson  自行车停放（可选）
    soho_boundary.geojson           Soho 边界（可选，用于边界效应诊断）

设计原则：任一可选文件缺失时降级但不崩溃，便于先跑通再补数据。
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from shapely.geometry import Point


# ══════════════════════════════════════════════════════════════
# 路段几何默认值
# ══════════════════════════════════════════════════════════════
# OSM 的 width 标签覆盖率极低，故按 highway 类型给默认值。
# 数值参考 TfL PCL 推荐宽度与 Soho 实际街道尺度，属可校准参数。

DEFAULT_CARRIAGEWAY_WIDTH = {      # 车行道宽度 (m)
    "motorway": 10.0, "trunk": 9.0, "primary": 8.0, "secondary": 7.0,
    "tertiary": 6.5, "residential": 5.5, "living_street": 4.5,
    "unclassified": 5.5, "service": 4.0, "pedestrian": 6.0,
    "footway": 0.0, "path": 0.0, "cycleway": 3.0, "steps": 0.0,
}

DEFAULT_FOOTWAY_WIDTH = {          # 单侧人行道总宽 (m)
    "primary": 3.5, "secondary": 3.0, "tertiary": 2.8,
    "residential": 2.2, "living_street": 2.0, "unclassified": 2.2,
    "service": 1.8, "pedestrian": 6.0, "footway": 2.5,
    "cycleway": 0.0, "path": 2.0, "steps": 1.5,
}

FALLBACK_CARRIAGEWAY = 5.5
FALLBACK_FOOTWAY = 2.2


def _first(value, default=None):
    """OSM 标签可能是列表（路段合并所致），取第一个值。"""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _as_float(value, default=None):
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def _assign_edge_widths(G) -> int:
    """给每条边写入 carriageway_width / footway_width。

    从 load_environment() 中抽出，供 build_environment_for_radius()
    （多半径扫描，路网现场抓取而非从 graphml 文件读入）复用，避免逻辑重复。
    返回值：有多少条边的宽度来自 OSM width 标签（其余用类型默认值）。
    """
    n_from_tag = 0
    for u, v, d in G.edges(data=True):
        hw = _first(d.get("highway"), "residential")

        tagged = _as_float(_first(d.get("width")))
        if tagged and 0.5 < tagged < 50:
            d["carriageway_width"] = tagged
            n_from_tag += 1
        else:
            d["carriageway_width"] = DEFAULT_CARRIAGEWAY_WIDTH.get(
                hw, FALLBACK_CARRIAGEWAY)

        d["footway_width"] = DEFAULT_FOOTWAY_WIDTH.get(hw, FALLBACK_FOOTWAY)
    return n_from_tag


# ══════════════════════════════════════════════════════════════
# 环境对象
# ══════════════════════════════════════════════════════════════

@dataclass
class SohoEnvironment:
    """模型的环境层。所有空间数据在此汇总，模型只读不写。"""

    graph: nx.MultiDiGraph
    poi_nodes: list = field(default_factory=list)        # [(node, weight), ...]
    land_use_mix: dict = field(default_factory=dict)     # {node: 0-1} 餐饮密度，专用于取餐点加权+land_use_threshold触发
    activity_density: dict = field(default_factory=dict) # {node: 0-1} 住宅/办公/机构密度，专用于骑手终点+行人起讫点加权
    loading_bay_nodes: set = field(default_factory=set)
    cycle_parking_nodes: set = field(default_factory=set)
    soho_boundary: object = None                         # Soho 边界 polygon（已投影到图的CRS）

    # 派生索引
    nodes: list = field(default_factory=list)
    node_xy: dict = field(default_factory=dict)          # {node: (x, y)} 投影坐标(m)

    def __post_init__(self):
        self.nodes = list(self.graph.nodes)
        self.node_xy = {
            n: (d.get("x", 0.0), d.get("y", 0.0))
            for n, d in self.graph.nodes(data=True)
        }

    # ---------------- 路段属性 ----------------
    def edge_data(self, u, v):
        """取 u→v 之间最短的那条平行边。"""
        candidates = self.graph.get_edge_data(u, v)
        if not candidates:
            return {}
        return min(candidates.values(), key=lambda d: d.get("length", 1e9))

    def edge_length(self, u, v) -> float:
        return float(self.edge_data(u, v).get("length", 30.0))

    def carriageway_width(self, u, v) -> float:
        return float(self.edge_data(u, v).get("carriageway_width", FALLBACK_CARRIAGEWAY))

    def footway_width(self, u, v) -> float:
        """人行道总宽（未扣缓冲）。有效宽度由 pedestrian_params 计算。"""
        return float(self.edge_data(u, v).get("footway_width", FALLBACK_FOOTWAY))

    def has_loading_bay(self, node) -> bool:
        return node in self.loading_bay_nodes

    def has_nearby_cycle_parking(self, node, radius_m=15.0):
        """节点周边radius_m内是否有自行车停车点（用于路径层：land_use触发是否被合法停车位抵消）。"""
        if node not in self.node_xy or not self.cycle_parking_nodes:
            return False
        x0, y0 = self.node_xy[node]
        for cp in self.cycle_parking_nodes:
            if cp not in self.node_xy:
                continue
            x1, y1 = self.node_xy[cp]
            if (x0 - x1) ** 2 + (y0 - y1) ** 2 <= radius_m ** 2:
                return True
        return False

    def legal_parking_coverage(self, node, radius_m=15.0):
        """
        连续版本的停车覆盖度 ∈ [0,1]，取代 has_nearby_cycle_parking() 的
        二元(在/不在radius_m内)判断。

        用节点到最近一个合法停车点的距离做线性衰减：距离为0时覆盖度=1，
        距离达到radius_m时覆盖度=0，之间线性过渡——不再是"50m内完全没影响，
        50m外完全没有覆盖"这种一刀切。用最近距离而不是像land_use_mix那样
        累加多个点，因为骑手只需要一个能用的停车位，多个停车点不会增加
        "有地方停车"这件事本身的确定性。
        """
        if node not in self.node_xy or not self.cycle_parking_nodes:
            return 0.0
        x0, y0 = self.node_xy[node]
        min_d2 = min(
            (x0 - self.node_xy[cp][0]) ** 2 + (y0 - self.node_xy[cp][1]) ** 2
            for cp in self.cycle_parking_nodes if cp in self.node_xy
        )
        min_d = min_d2 ** 0.5
        if min_d >= radius_m:
            return 0.0
        return 1.0 - min_d / radius_m

    # ---------------- 路径 ----------------
    def shortest_route(self, origin, destination, weight="length"):
        """返回节点序列；不可达时返回 [origin]。"""
        if origin == destination:
            return [origin]
        try:
            return nx.shortest_path(self.graph, origin, destination, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [origin]

    def random_poi_node(self, rng, exclude=None):
        """
        按 POI 权重抽取一个取餐点。

        exclude: 可选的节点集合，抽取时跳过这些节点——用于暗厨房情景下
        把已经被"吸收"进 micro_hubs 的节点从常规 POI 池里排除，
        避免同一个节点既算作暗厨房、又算作独立餐厅重复计入。
        """
        pool = self.poi_nodes
        if exclude:
            pool = [(n, w) for n, w in pool if n not in exclude]
        if not pool:
            return rng.choice(self.nodes)
        nodes, weights = zip(*pool)
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for n, w in zip(nodes, weights):
            acc += w
            if r <= acc:
                return n
        return nodes[-1]

    # ---------------- 边界效应诊断 ----------------
    def compute_node_boundary_distances(self):
        """
        计算每个节点到 Soho 边界的距离，写入 graph 节点属性 dist_to_boundary。

        用于检查模型范围是否存在边界效应（骑手路径/拥堵/冲突是否在
        地图边缘因为"图外没有路网"而被人为扭曲）。若 soho_boundary
        未提供，直接跳过（不报错，保持"降级但不崩溃"的一致设计）。
        """
        if self.soho_boundary is None:
            return
        boundary_line = self.soho_boundary.boundary
        for node, data in self.graph.nodes(data=True):
            pt = Point(data['x'], data['y'])
            data['dist_to_boundary'] = pt.distance(boundary_line)


# ══════════════════════════════════════════════════════════════
# 加载
# ══════════════════════════════════════════════════════════════

def load_environment(
    data_dir: str,
    network_file: str = "raw/soho_bike_network_r1900.graphml",
    poi_file: str = "soho_food_pois_r1900.csv",
    loading_bay_file: str = "raw/soho_loading_bays_r1900.geojson",
    cycle_parking_file: str = "raw/soho_cycle_parking_r1900.geojson",
    boundary_file: str = "soho_boundary_r1900.geojson",
    land_use_radius_m: float = 100.0,
    activity_landuse_file: str = "raw/osm_landuse_r1900.geojson",
    activity_radius_m: float = 250.0,
    verbose: bool = True,
) -> SohoEnvironment:
    """从磁盘装配环境层。

    ⚠️ 上面5个文件名默认值已改成 _r1900 版本（配合你8月确认过的边界校准结果）。
    activity_landuse_file/activity_radius_m 是新增的两个参数，用于计算
    activity_density（住宅/办公/机构密度，见下方 SohoEnvironment 里的用法）。
    """

    def log(msg):
        if verbose:
            print(f"  [env] {msg}")

    # ---------- 1. 路网 ----------
    path = os.path.join(data_dir, network_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到路网文件：{path}\n"
            f"请确认 data_dir 与 network_file 参数是否正确。"
        )

    try:
        import osmnx as ox
        G = ox.load_graphml(path)
        G = ox.project_graph(G)          # 投影到米制，便于按距离移动
        log(f"路网已投影：{G.graph.get('crs')}")
    except ImportError:
        G = nx.read_graphml(path)
        warnings.warn("未安装 osmnx，路网未投影，长度单位可能不是米")

    # graphml 读入后属性是字符串，转成数值
    for _, _, d in G.edges(data=True):
        d["length"] = _as_float(_first(d.get("length")), 30.0)

    log(f"路网节点 {G.number_of_nodes()}，路段 {G.number_of_edges()}")

    # ---------- 2. 路段宽度 ----------
    n_from_tag = _assign_edge_widths(G)
    log(f"路段宽度：{n_from_tag} 条来自 OSM width 标签，其余用类型默认值")

    # ---------- 3. 餐饮 POI ----------
    poi_nodes = []
    poi_path = os.path.join(data_dir, poi_file)
    if os.path.exists(poi_path):
        poi_nodes = _snap_pois(G, poi_path, log)
    else:
        log(f"未找到 POI 文件 {poi_path}，订单起点将退化为随机节点")

    # ---------- 4. 土地利用混合度 ----------
    land_use_mix = _compute_land_use_mix(G, poi_nodes, land_use_radius_m)
    if land_use_mix:
        vals = list(land_use_mix.values())
        log(f"land_use_mix：均值 {sum(vals)/len(vals):.2f}，最大 {max(vals):.2f}")

    # ---------- 4b. 活动密度（住宅/办公/机构，用于骑手终点+行人起讫点加权） ----------
    activity_features = _load_activity_landuse(
        G, os.path.join(data_dir, activity_landuse_file),
        ACTIVITY_LANDUSE_CATEGORIES, log,
    )
    activity_density = _compute_activity_density(G, activity_features, activity_radius_m)
    if activity_density:
        vals = list(activity_density.values())
        log(f"activity_density：均值 {sum(vals)/len(vals):.2f}，最大 {max(vals):.2f}")

    # ---------- 5. 装卸区与自行车停放 ----------
    loading = _snap_geojson(G, os.path.join(data_dir, loading_bay_file), "装卸区", log)
    cycling = _snap_geojson(G, os.path.join(data_dir, cycle_parking_file), "自行车停放", log)

    # ---------- 6. Soho 边界（用于边界效应诊断） ----------
    soho_boundary = None
    boundary_path = os.path.join(data_dir, boundary_file)
    if os.path.exists(boundary_path):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(boundary_path)          # 原始通常是 CRS84（经纬度）
            graph_crs = G.graph.get("crs")
            if graph_crs is not None:
                gdf = gdf.to_crs(graph_crs)              # 重新投影到和路网一样的CRS
            soho_boundary = gdf.geometry.iloc[0]
            log(f"边界已加载并投影到 {graph_crs}")
        except Exception as e:
            log(f"边界读取失败（{type(e).__name__}: {e}），dist_to_boundary 将不可用")
    else:
        log(f"未找到边界文件 {boundary_path}，dist_to_boundary 将不可用")

    env = SohoEnvironment(
        graph=G,
        poi_nodes=poi_nodes,
        land_use_mix=land_use_mix,
        activity_density=activity_density,
        loading_bay_nodes=loading,
        cycle_parking_nodes=cycling,
        soho_boundary=soho_boundary,
    )
    env.compute_node_boundary_distances()   # 写入 dist_to_boundary（若无边界数据则安全跳过）
    return env


# ---------------- 内部工具 ----------------

def _snap_pois(G, poi_path, log):
    try:
        import pandas as pd
    except ImportError:
        log("未安装 pandas，跳过 POI")
        return []
    df = pd.read_csv(poi_path)
    return _snap_pois_from_df(G, df, log)


def _snap_pois_from_df(G, df, log):
    """从已经读入内存的 POI DataFrame 吸附到路网节点。

    从 _snap_pois() 拆出，供 build_environment_for_radius() 复用——
    多半径扫描时 POI 先按距圆心的距离过滤（见 _filter_poi_csv_by_radius），
    过滤后的 DataFrame 直接传进来，不需要再从磁盘读一次完整文件。
    """
    lon_col = next((c for c in df.columns if c.lower() in ("lon", "longitude")), None)
    lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude")), None)
    if lon_col is None or lat_col is None:
        log(f"POI 文件缺少经纬度列，现有列：{list(df.columns)[:8]}")
        return []

    try:
        import osmnx as ox
        from pyproj import Transformer
        # CSV 是 EPSG:4326，G 已投影到 graph_crs，必须先转到同一坐标系
        graph_crs = G.graph.get("crs")
        transformer = Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
        xs, ys = transformer.transform(df[lon_col].values, df[lat_col].values)
        nodes = ox.nearest_nodes(G, xs, ys)
    except Exception as e:
        log(f"osmnx 吸附失败（{type(e).__name__}），改用暴力最近点")
        nodes = _brute_force_snap(G, df[lon_col].values, df[lat_col].values)

    # 外卖店权重更高（订单频次），此处作为可校准参数
    type_weight = {"Takeaway/sandwich shop": 1.5,
                   "Restaurant/Cafe/Canteen": 1.0,
                   "Other catering premises": 0.6}
    if "BusinessType" in df.columns:
        df = df[df["BusinessType"].isin(type_weight.keys())].copy()
        weights = df["BusinessType"].map(type_weight).values
    else:
        weights = [1.0] * len(df)

    agg = {}
    for n, w in zip(nodes, weights):
        agg[n] = agg.get(n, 0.0) + float(w)

    log(f"POI {len(df)} 个吸附到 {len(agg)} 个节点")
    return list(agg.items())


def _brute_force_snap(G, lons, lats):
    """无 osmnx 时的退化方案（经纬度近似平面距离）。"""
    pts = [(n, d.get("x", 0.0), d.get("y", 0.0)) for n, d in G.nodes(data=True)]
    out = []
    for lon, lat in zip(lons, lats):
        best, best_d = None, float("inf")
        for n, x, y in pts:
            d2 = (x - lon) ** 2 + (y - lat) ** 2
            if d2 < best_d:
                best, best_d = n, d2
        out.append(best)
    return out


def _compute_land_use_mix(G, poi_nodes, radius_m):
    """
    land_use_mix ∈ [0,1]：节点周边餐饮 POI 密度的归一化值。

    代表"该处商业活动强度"——高值意味着装卸需求密集、
    合法路缘空间被占满，对应决策矩阵中阈值下调的情形。
    """
    if not poi_nodes:
        return {n: 0.5 for n in G.nodes}

    xy = {n: (d.get("x", 0.0), d.get("y", 0.0)) for n, d in G.nodes(data=True)}
    poi_pts = [(xy[n], w) for n, w in poi_nodes if n in xy]

    raw = {}
    r2 = radius_m ** 2
    for n, (x, y) in xy.items():
        s = 0.0
        for (px, py), w in poi_pts:
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 <= r2:
                s += w * (1 - math.sqrt(d2) / radius_m)   # 线性距离衰减
        raw[n] = s

    hi = max(raw.values()) or 1.0
    return {n: min(v / hi, 1.0) for n, v in raw.items()}


# ══════════════════════════════════════════════════════════════
# activity_density：住宅/办公/机构密度 —— 与 land_use_mix（餐饮密度）
# 是两个独立的变量，前者代表"人在的地方"，用于骑手终点与行人起讫点加权；
# 后者代表"餐饮商业密度"，专用于取餐点加权与 land_use_threshold 触发。
# 混用会导致行人/骑手终点分布不现实（详见方法论讨论），故分开维护。
# ══════════════════════════════════════════════════════════════

ACTIVITY_LANDUSE_CATEGORIES = {
    "residential", "commercial", "retail", "governmental", "education", "religious",
}


def _load_activity_landuse(G, landuse_path, categories, log):
    """
    读取 OSM landuse geojson，筛出住宅/办公/机构等"人活动相关"类别，
    转成 [((x, y), area_m2), ...]（已投影到路网 CRS），供距离衰减计算使用。
    权重用地块投影后的面积（m²），而不是简单计数——一大片住宅区
    应该比一个小地块贡献更多"活动密度"。
    """
    if not os.path.exists(landuse_path):
        log(f"未找到 activity landuse 文件 {landuse_path}，activity_density 将退化为均匀分布")
        return []
    try:
        import geopandas as gpd
    except ImportError:
        log("未安装 geopandas，跳过 activity_density")
        return []

    gdf = gpd.read_file(landuse_path)
    if "landuse" not in gdf.columns:
        log(f"{landuse_path} 缺少 'landuse' 列，跳过 activity_density")
        return []

    gdf = gdf[gdf["landuse"].isin(categories)].copy()
    if gdf.empty:
        log(f"{landuse_path} 里没有匹配 {categories} 的地块，activity_density 将退化为均匀分布")
        return []

    graph_crs = G.graph.get("crs")
    if gdf.crs is not None and graph_crs is not None:
        gdf = gdf.to_crs(graph_crs)   # 投影到米制，area才是平方米

    areas = gdf.geometry.area
    centroids = gdf.geometry.centroid
    features = [((pt.x, pt.y), float(a)) for pt, a in zip(centroids, areas) if a > 0]
    log(f"activity landuse：{len(features)} 个地块（类别：{sorted(gdf['landuse'].unique())}）")
    return features


def _compute_activity_density(G, landuse_features, radius_m):
    """
    activity_density ∈ [0,1]：节点周边住宅/办公/机构用地的距离衰减密度。

    算法与 _compute_land_use_mix 相同（线性距离衰减+归一化），
    只是输入换成了 landuse 地块的质心+面积，而不是餐饮 POI。
    landuse_features 为空时退化为全图 0.5（不产生偏好，等同于均匀随机）。
    """
    if not landuse_features:
        return {n: 0.5 for n in G.nodes}

    xy = {n: (d.get("x", 0.0), d.get("y", 0.0)) for n, d in G.nodes(data=True)}
    raw = {}
    r2 = radius_m ** 2
    for n, (x, y) in xy.items():
        s = 0.0
        for (px, py), w in landuse_features:
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 <= r2:
                s += w * (1 - math.sqrt(d2) / radius_m)
        raw[n] = s

    hi = max(raw.values()) or 1.0
    return {n: min(v / hi, 1.0) for n, v in raw.items()}


def load_soho_center_latlon(data_dir: str, boundary_file: str = "soho_boundary.geojson"):
    """从已有的 soho_boundary.geojson 取几何中心（WGS84 经纬度）。

    多半径扫描统一以这个点为圆心，保证候选边界都锚定在原来的 Soho
    研究区域上，而不是另外找一个坐标手动写死。
    返回 (lat, lon)。
    """
    import geopandas as gpd
    path = os.path.join(data_dir, boundary_file)
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    centroid = gdf.geometry.iloc[0].centroid
    return centroid.y, centroid.x


def _filter_poi_csv_by_radius(poi_path: str, center_latlon, radius_m: float):
    """从一份覆盖较大范围的 POI CSV 里，筛出距圆心 radius_m 以内的行。

    要求 poi_path 指向的文件本身就覆盖了本次扫描里最大的候选半径
    （即需要预先用比最大候选半径更大的查询范围向 FSA FHRS API 取一次数据，
    存成宽域 CSV），这样多个半径可以复用同一份底层数据，不必每个半径
    都重新调用一次外部 API。
    """
    import pandas as pd

    df = pd.read_csv(poi_path)
    lon_col = next((c for c in df.columns if c.lower() in ("lon", "longitude")), None)
    lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude")), None)
    if lon_col is None or lat_col is None:
        return df.iloc[0:0]

    lat0, lon0 = center_latlon
    lat0r = math.radians(lat0)
    R = 6371000.0

    def _haversine(lat, lon):
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat0r) * math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(min(1.0, a)))

    dists = [
        _haversine(lat, lon)
        for lat, lon in zip(df[lat_col].values, df[lon_col].values)
    ]
    return df.loc[[d <= radius_m for d in dists]].reset_index(drop=True)


def build_environment_for_radius(
    data_dir: str,
    radius_m: float,
    center_latlon,
    poi_source_file: str = "soho_food_pois_wide.csv",
    loading_bay_file: str = "raw/soho_loading_bays.geojson",
    cycle_parking_file: str = "raw/soho_cycle_parking.geojson",
    network_type: str = "bike",
    land_use_radius_m: float = 100.0,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    """以 center_latlon 为圆心、radius_m 为半径现场抓取路网并组装环境层。

    用于边界尺寸扫描实验（配送距离校准），与 load_environment() 并列存在，
    不改动 S1/S2 正式实验用的固定边界流程。

    依赖 & 前提（需要用户自行确认/准备）：
    - 本地需要能访问 OSM Overpass API（osmnx 联网抓取路网）
    - poi_source_file 需要覆盖本次扫描里最大的候选半径（否则大半径会漏算
      边界外的真实 POI，人为拉低该半径下的骑手数估算）；如果你现有的
      soho_food_pois.csv 只覆盖了原来的小范围 Soho 边界，需要先用更大的
      查询范围重新跑一次 FSA FHRS API，存成新文件，路径通过
      poi_source_file 传入
    - loading_bay_file / cycle_parking_file 同理：目前的 geojson 大概率
      只覆盖原 Soho 边界，扩大半径后这两类数据在边界外会是空的——这会让
      "地图更大但合法停车点没变多"，从而人为提高路径层的 land_use 强制
      触发率。如果你手头没有更大范围的这两类数据，建议在扫描阶段暂时
      忽略这两个机制的影响（不作为选边界的判据），只用配送距离做校准；
      正式定下边界后再单独确认这两类数据是否需要扩展

    返回：(env, poi_count) —— poi_count 是这个半径内吸附到路网的 POI 个数，
    用来按比例回推 n_riders / n_pedestrians。
    """
    def log(msg):
        if verbose:
            print(f"  [env r={radius_m:.0f}m] {msg}")

    import osmnx as ox

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"network_r{int(radius_m)}.graphml")

    if cache_path and os.path.exists(cache_path):
        G = ox.load_graphml(cache_path)
        log(f"路网缓存命中：{cache_path}")
    else:
        G = ox.graph_from_point(center_latlon, dist=radius_m, network_type=network_type)
        if cache_path:
            ox.save_graphml(G, cache_path)
        log(f"从 OSM 抓取路网（{network_type}，半径{radius_m:.0f}m）")

    G = ox.project_graph(G)
    for _, _, d in G.edges(data=True):
        d["length"] = _as_float(_first(d.get("length")), 30.0)
    log(f"路网节点 {G.number_of_nodes()}，路段 {G.number_of_edges()}")

    n_from_tag = _assign_edge_widths(G)
    log(f"路段宽度：{n_from_tag} 条来自 OSM width 标签，其余用类型默认值")

    poi_nodes = []
    poi_path = os.path.join(data_dir, poi_source_file)
    if os.path.exists(poi_path):
        filtered = _filter_poi_csv_by_radius(poi_path, center_latlon, radius_m)
        poi_nodes = _snap_pois_from_df(G, filtered, log)
    else:
        log(f"未找到宽域 POI 文件 {poi_path}，订单起点将退化为随机节点")

    land_use_mix = _compute_land_use_mix(G, poi_nodes, land_use_radius_m)

    loading = _snap_geojson(G, os.path.join(data_dir, loading_bay_file), "装卸区", log)
    cycling = _snap_geojson(G, os.path.join(data_dir, cycle_parking_file), "自行车停放", log)

    # 用圆心+半径本身构造边界多边形（用于 dist_to_boundary 边界效应诊断），
    # 不依赖额外的 geojson 文件
    import geopandas as gpd
    center_gdf = gpd.GeoDataFrame(
        geometry=[Point(center_latlon[1], center_latlon[0])], crs="EPSG:4326"
    )
    center_gdf = center_gdf.to_crs(G.graph.get("crs"))
    soho_boundary = center_gdf.geometry.iloc[0].buffer(radius_m)

    env = SohoEnvironment(
        graph=G,
        poi_nodes=poi_nodes,
        land_use_mix=land_use_mix,
        loading_bay_nodes=loading,
        cycle_parking_nodes=cycling,
        soho_boundary=soho_boundary,
    )
    env.compute_node_boundary_distances()
    log(f"POI 节点数 {len(poi_nodes)}（用于按比例回推骑手数）")
    return env, len(poi_nodes)


def _snap_geojson(G, path, label, log):
    if not os.path.exists(path):
        log(f"未找到{label}文件，相关功能降级")
        return set()
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        if gdf.empty:
            return set()
        graph_crs = G.graph.get("crs")
        if gdf.crs is not None and graph_crs is not None:
            gdf = gdf.to_crs(graph_crs)   # 先转到路网的投影坐标系
        pts = gdf.geometry.centroid       # 再算centroid，单位就是米了，warning也会消失
        import osmnx as ox
        nodes = ox.nearest_nodes(G, pts.x.values, pts.y.values)
        log(f"{label} {len(gdf)} 个吸附到 {len(set(nodes))} 个节点")
        return set(nodes)
    except Exception as e:
        log(f"{label}读取失败（{type(e).__name__}: {e}），相关功能降级")
        return set()
