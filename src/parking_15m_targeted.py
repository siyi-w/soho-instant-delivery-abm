import sys
sys.path.insert(0, "/Users/wusiyi/Documents/CASA/dis/DATA/files")

import environment
from environment import load_environment
from model import SohoDeliveryModel
from agents import RiderAgent
from interventions import add_legal_parking_targeted

_original_coverage = environment.SohoEnvironment.legal_parking_coverage
def _coverage_15m(self, node, radius_m=100.0):
    return _original_coverage(self, node, radius_m=15.0)
environment.SohoEnvironment.legal_parking_coverage = _coverage_15m

_original_nearby = environment.SohoEnvironment.has_nearby_cycle_parking
def _nearby_15m(self, node, radius_m=50.0):
    return _original_nearby(self, node, radius_m=15.0)
environment.SohoEnvironment.has_nearby_cycle_parking = _nearby_15m

DATA_DIR = "/Users/wusiyi/Documents/CASA/dis/DATA"
base_env = load_environment(DATA_DIR)
original_parking_nodes = frozenset(base_env.cycle_parking_nodes)

n_steps = int(180 * 60 / 5)
seeds = [42, 1, 7, 99, 123]
doses = [0, 1, 3, 5, 8, 12, 15, 20, 25, 30, 36, 50]  # 36是真缺口总数，50做个余量确认平台期

results = {}

for dose in doses:
    base_env.cycle_parking_nodes = set(original_parking_nodes)
    if dose > 0:
        env = add_legal_parking_targeted(base_env, n=dose, land_use_threshold=0.65)
    else:
        env = base_env

    trigger_rates = []
    for seed in seeds:
        model = SohoDeliveryModel(
            env=env, n_riders=300, n_pedestrians=600, seed=seed,
            scenario="S2_adverse", congestion_threshold=0.3, land_use_threshold=0.65,
        )
        n_checked, n_triggered = 0, 0
        for i in range(n_steps):
            model.step()
            for a in model.agents_by_type[RiderAgent]:
                n_checked += 1
                if getattr(a, "route_trigger", None) == "land_use":
                    n_triggered += 1
        trigger_rates.append(n_triggered / n_checked if n_checked else float("nan"))
        print(f"  dose={dose} seed={seed}: trigger_rate={trigger_rates[-1]:.4%}")

    results[dose] = trigger_rates

print("\n========== 15米停车干预精细剂量扫描（靶向选址修复版, S2_adverse, 5-seed）==========")
for dose, t in results.items():
    t_mean = sum(t) / len(t)
    print(f"dose={dose:>4}: land_use触发率={t_mean:.4%}, 范围=[{min(t):.4%},{max(t):.4%}]")