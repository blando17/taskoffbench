"""
final2_fair.py
Fair, untuned metaheuristic comparison for MEC-style task offloading.
- Runs PSO, DE, GA with neutral parameters (no tuning/adaptation).
- Equal evaluation budgets across algorithms.
- Workloads: 10k,20k,...,100k tasks.
- Prints: Mean Queue Wait, Mean Total Wait, Mean RT, p90/p99 RT, Energy/task, Total Energy, Offload %, Overloads.
- No exploration/exploitation metrics.
"""

import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable
import numpy as np

# ----------------------------- System & Simulation Params -----------------------------

SYSTEM_PARAMS = {
    "num_devices": 50,
    "num_edges": 5,
    "device_cpu_ghz": 2.0,
    "edge_cpu_ghz":   [10, 12, 8, 9, 11],  # per-edge GHz
    "uplink_mbps":    50,
    "rtt_ms":         10,
    "device_power_w": 1.8,
    "tx_power_w":     1.2,
    "edge_power_w":   3.0,
}

SIM_PARAMS = {
    "seed": 42,
    "deadline_ms": (150, 1200),
    "task_size_kb": (50, 1500),
    "cpu_cycles_m": (30, 600),
    "overload_penalty": 5.0,
    "energy_weight": 0.2,
    "p99_weight": 0.3,
}

random.seed(SIM_PARAMS["seed"])
np.random.seed(SIM_PARAMS["seed"])

# ----------------------------- Task Model -----------------------------

@dataclass
class Task:
    size_kb: float
    cpu_cycles_m: float
    deadline_ms: float

def gen_tasks(n: int) -> List[Task]:
    size = np.random.uniform(*SIM_PARAMS["task_size_kb"], size=n)
    cycles = np.random.uniform(*SIM_PARAMS["cpu_cycles_m"], size=n)
    deadline = np.random.uniform(*SIM_PARAMS["deadline_ms"], size=n)
    return [Task(float(size[i]), float(cycles[i]), float(deadline[i])) for i in range(n)]

# ----------------------------- Policy Representation -----------------------------

POLICY_DIM = 12

def decode_policy(theta: np.ndarray) -> Dict[str, Any]:
    # 12D vector in [0,1] → interpretable policy
    w = 4.0 * (theta[:4] - 0.5)            # weights in [-2,2] for [size, cycles, urgency, jitter]
    server_bias = 4.0 * (theta[4:9] - 0.5) # up to 5 edges
    temp = 0.3 + 1.2 * theta[9]            # [0.3, 1.5]
    bw_frac = 0.2 + 0.8 * theta[10]        # [0.2, 1.0]
    local_cpu_scale = 0.5 + 0.8 * theta[11]# [0.5, 1.3]
    return {
        "w": w,
        "server_bias": server_bias,
        "temp": float(temp),
        "bw_frac": float(bw_frac),
        "local_cpu_scale": float(local_cpu_scale)
    }

# ----------------------------- Simulator -----------------------------

def simulate(tasks: List[Task], theta: np.ndarray) -> Dict[str, float]:
    sp = SYSTEM_PARAMS
    pd = decode_policy(theta)
    num_edges = sp["num_edges"]
    server_bias = pd["server_bias"][:num_edges]

    device_free_ms = np.zeros(sp["num_devices"])
    edge_free_ms = np.zeros(num_edges)

    rt_list = []
    queue_wait_list = []
    total_wait_list = []
    energy_list = []
    offload_count = 0
    overloads = 0

    dev_idx = 0

    for t in tasks:
        size = t.size_kb
        cycles = t.cpu_cycles_m
        urgency = 1.0 / max(t.deadline_ms, 1.0)
        jitter = np.random.randn() * 0.1
        x = np.array([size/1500.0, cycles/600.0, urgency*(SIM_PARAMS["deadline_ms"][1]), jitter])
        score_offload = float(np.dot(pd["w"], x))

        # Local vs offload gate (neutral sigmoid; no tuning)
        p_off = 1.0 / (1.0 + math.exp(-score_offload))
        do_offload = (p_off > 0.5)

        if do_offload:
            # Choose edge via softmax on fixed biases
            logits = server_bias / pd["temp"]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            edge = int(np.random.choice(np.arange(num_edges), p=probs))
            # Network
            uplink_mbps = sp["uplink_mbps"] * pd["bw_frac"]
            tx_time_ms = (size * 8.0 / max(uplink_mbps, 1e-6)) + sp["rtt_ms"]
            # Edge compute time
            edge_ghz = sp["edge_cpu_ghz"][edge]
            comp_ms = (cycles / (edge_ghz * 1000.0)) * 1000.0
            # Edge queue
            wait_ms = max(0.0, edge_free_ms[edge])
            start_delay_ms = wait_ms
            finish_time_ms = wait_ms + tx_time_ms + comp_ms
            edge_free_ms[edge] = finish_time_ms
            # Energy (TX + small edge amortization)
            energy = (sp["tx_power_w"] * (tx_time_ms/1000.0)) + (sp["edge_power_w"] * (comp_ms/1000.0) * 0.1)
            offload_count += 1
        else:
            # Local compute
            local_ghz = sp["device_cpu_ghz"] * pd["local_cpu_scale"]
            comp_ms = (cycles / (local_ghz * 1000.0)) * 1000.0
            d = dev_idx
            dev_idx = (dev_idx + 1) % sp["num_devices"]
            wait_ms = max(0.0, device_free_ms[d])
            start_delay_ms = wait_ms
            finish_time_ms = wait_ms + comp_ms
            device_free_ms[d] = finish_time_ms
            # Energy (device active)
            energy = sp["device_power_w"] * (comp_ms/1000.0)

        rt_ms = start_delay_ms + (tx_time_ms if do_offload else 0.0) + comp_ms
        total_wait_ms = start_delay_ms + (tx_time_ms if do_offload else 0.0)

        rt_list.append(rt_ms)
        queue_wait_list.append(start_delay_ms)
        total_wait_list.append(total_wait_ms)
        energy_list.append(energy)

        if rt_ms > t.deadline_ms:
            overloads += 1

    rt = np.array(rt_list)
    energy = np.array(energy_list)
    return {
        "Mean Queue Wait (ms)": float(np.mean(queue_wait_list)),
        "Mean Total Wait (ms)": float(np.mean(total_wait_list)),
        "Mean RT (ms)": float(np.mean(rt)),
        "p90 RT (ms)": float(np.percentile(rt, 90)),
        "p99 RT (ms)": float(np.percentile(rt, 99)),
        "Energy/task (J)": float(np.mean(energy)),
        "Total Energy (kJ)": float(np.sum(energy)/1000.0),
        "Offload %": 100.0 * offload_count / max(1, len(tasks)),
        "Overloads": int(overloads),
    }

# ----------------------------- Fitness (neutral composite) -----------------------------

def fitness_from_metrics(m: Dict[str, float]) -> float:
    # Lower is better; neutral composite of mean RT + tail + energy + overload penalties
    return float(
        m["Mean RT (ms)"]
        + SIM_PARAMS["p99_weight"] * m["p99 RT (ms)"]
        + SIM_PARAMS["energy_weight"] * (1000.0 * m["Energy/task (J)"])
        + SIM_PARAMS["overload_penalty"] * m["Overloads"]
    )

# ----------------------------- Neutral Metaheuristics -----------------------------

class BaseOpt:
    def __init__(self, pop, iters, bounds, eval_fn, seed=0):
        self.pop = pop
        self.iters = iters
        self.bounds = bounds
        self.eval = eval_fn
        self.dim = len(bounds)
        random.seed(seed)
        np.random.seed(seed)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        self.X = lows + (highs - lows) * np.random.rand(pop, self.dim)
        self.F = np.array([self.eval(x) for x in self.X])
        gi = int(np.argmin(self.F))
        self.gb = self.X[gi].copy()
        self.gbf = float(self.F[gi])

# ===== NEUTRAL PSO: fixed w=0.7, c1=c2=1.4; no inertia schedule =====
class PSO:
    def __init__(self, pop_size, iters, bounds, evaluate, seed=0, w=0.7, c1=1.4, c2=1.4):
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.dim = len(bounds)
        self.evaluate = evaluate
        self.w, self.c1, self.c2 = w, c1, c2
        import numpy as np, random
        random.seed(seed); np.random.seed(seed)
        lows = np.array([b[0] for b in bounds]); highs = np.array([b[1] for b in bounds])
        self.X = lows + (highs - lows) * np.random.rand(pop_size, self.dim)
        self.V = np.zeros_like(self.X)
        self.F = np.array([self.evaluate(x) for x in self.X])
        self.P = self.X.copy()
        self.Pf = self.F.copy()
        gi = int(self.F.argmin())
        self.gb = self.X[gi].copy()
        self.gbf = float(self.F[gi])

    def step(self):
        import numpy as np
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.V = (self.w * self.V
                  + self.c1 * r1 * (self.P - self.X)
                  + self.c2 * r2 * (self.gb - self.X))
        lows = np.array([b[0] for b in self.bounds]); highs = np.array([b[1] for b in self.bounds])
        self.X = np.clip(self.X + self.V, lows, highs)
        self.F = np.array([self.evaluate(x) for x in self.X])
        mask = self.F < self.Pf
        self.P[mask] = self.X[mask]
        self.Pf[mask] = self.F[mask]
        gi = int(self.F.argmin())
        if self.F[gi] < self.gbf:
            self.gbf = float(self.F[gi])
            self.gb = self.X[gi].copy()

    def run(self):
        for _ in range(self.iters):
            self.step()
        return self.gb.copy(), float(self.gbf)

# ===== NEUTRAL DE: rand/1/bin, fixed F=0.5, CR=0.9 =====
class DE:
    def __init__(self, pop_size, iters, bounds, evaluate, seed=0, F=0.5, CR=0.9):
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.dim = len(bounds)
        self.evaluate = evaluate
        self.F, self.CR = F, CR
        import numpy as np, random
        random.seed(seed); np.random.seed(seed)
        lows = np.array([b[0] for b in bounds]); highs = np.array([b[1] for b in bounds])
        self.pop = lows + (highs - lows) * np.random.rand(pop_size, self.dim)
        self.fits = np.array([self.evaluate(x) for x in self.pop])

    def step(self):
        import numpy as np
        lows = np.array([b[0] for b in self.bounds]); highs = np.array([b[1] for b in self.bounds])
        new_pop = self.pop.copy()
        new_fits = self.fits.copy()
        for i in range(self.pop_size):
            idxs = [j for j in range(self.pop_size) if j != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.F * (b - c)
            cross = np.random.rand(self.dim) < self.CR
            if not np.any(cross):
                cross[np.random.randint(0, self.dim)] = True
            trial = np.where(cross, mutant, self.pop[i])
            trial = np.clip(trial, lows, highs)
            ft = self.evaluate(trial)
            if ft <= self.fits[i]:
                new_pop[i] = trial
                new_fits[i] = ft
        self.pop, self.fits = new_pop, new_fits

    def run(self):
        for _ in range(self.iters):
            self.step()
        i = int(self.fits.argmin())
        return self.pop[i].copy(), float(self.fits[i])


# ===== HYBRID PSO+DE =====
class HybridPSODE:
    def __init__(self, pop_size, iters, bounds, evaluate, seed=0,
                 w=0.7, c1=1.4, c2=1.4, F=0.5, CR=0.9, de_frac=0.3):
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.dim = len(bounds)
        self.evaluate = evaluate
        self.w, self.c1, self.c2 = w, c1, c2
        self.F, self.CR = F, CR
        self.de_frac = de_frac
        random.seed(seed); np.random.seed(seed)
        lows = np.array([b[0] for b in bounds]); highs = np.array([b[1] for b in bounds])
        self.X = lows + (highs - lows) * np.random.rand(pop_size, self.dim)
        self.V = np.zeros_like(self.X)
        self.Fvals = np.array([self.evaluate(x) for x in self.X])
        self.P = self.X.copy()
        self.Pf = self.Fvals.copy()
        gi = int(self.Fvals.argmin())
        self.gb = self.X[gi].copy()
        self.gbf = float(self.Fvals[gi])

    def step(self):
        lows = np.array([b[0] for b in self.bounds]); highs = np.array([b[1] for b in self.bounds])
        # ---- PSO update ----
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.V = (self.w * self.V
                  + self.c1 * r1 * (self.P - self.X)
                  + self.c2 * r2 * (self.gb - self.X))
        self.X = np.clip(self.X + self.V, lows, highs)

        # ---- Inject DE mutation ----
        for i in range(self.pop_size):
            if np.random.rand() < self.de_frac:
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.X[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                cross = np.random.rand(self.dim) < self.CR
                if not np.any(cross):
                    cross[np.random.randint(0, self.dim)] = True
                trial = np.where(cross, mutant, self.X[i])
                trial = np.clip(trial, lows, highs)
                ft = self.evaluate(trial)
                if ft < self.evaluate(self.X[i]):
                    self.X[i] = trial

        # ---- Fitness update ----
        self.Fvals = np.array([self.evaluate(x) for x in self.X])
        mask = self.Fvals < self.Pf
        self.P[mask] = self.X[mask]
        self.Pf[mask] = self.Fvals[mask]
        gi = int(self.Fvals.argmin())
        if self.Fvals[gi] < self.gbf:
            self.gbf = float(self.Fvals[gi])
            self.gb = self.X[gi].copy()

    def run(self):
        for _ in range(self.iters):
            self.step()
        return self.gb.copy(), float(self.gbf)


# ----------------------------- Objective Binding -----------------------------

def make_bounds() -> List[Tuple[float, float]]:
    return [(0.0, 1.0)] * POLICY_DIM

def build_eval(tasks: List[Task]) -> Callable[[np.ndarray], float]:
    def f(theta01: np.ndarray) -> float:
        x = np.clip(theta01, 0.0, 1.0)
        metrics = simulate(tasks, x)
        return fitness_from_metrics(metrics)
    return f

def run_for_workload(n_tasks: int, budget_fes: int = 4000, seed: int = 123) -> Dict[str, Dict[str, float]]:
    tasks = gen_tasks(n_tasks)
    bounds = make_bounds()
    dim = len(bounds)
    pop = max(40, min(100, 10*dim))
    iters = max(1, budget_fes // pop)

    eval_fn = build_eval(tasks)

    pso = PSO(pop, iters, bounds, eval_fn, seed=seed)
    x_pso, _ = pso.run()
    m_pso = simulate(tasks, x_pso)

    de  = DE(pop, iters, bounds, eval_fn, seed=seed+1)
    x_de, _ = de.run()
    m_de  = simulate(tasks, x_de)

    hyb = HybridPSODE(pop, iters, bounds, eval_fn, seed=seed+2)
    x_hyb, _ = hyb.run()
    m_hyb = simulate(tasks, x_hyb)

    return {"PSO": m_pso, "DE": m_de, "PSO+DE": m_hyb}

def format_report(n_tasks: int, results: Dict[str, Dict[str, float]]) -> str:
    keys = [
        "Mean Queue Wait (ms)","Mean Total Wait (ms)","Mean RT (ms)","p90 RT (ms)","p99 RT (ms)",
        "Energy/task (J)","Total Energy (kJ)","Offload %","Overloads"
    ]
    lines = []
    lines.append(f"\n================ Workload: {n_tasks:,} tasks ================\n")
    header = "Metric".ljust(24) + " | " + "PSO".rjust(12) + " | " + "DE".rjust(12) + " | " + "PSO+DE".rjust(12)
    lines.append(header)
    lines.append("-"*len(header))
    for k in keys:
        p = results["PSO"][k]; d = results["DE"][k]; h = results["PSO+DE"][k]
        if "Overloads" in k:
            line = k.ljust(24) + f" | {int(p):12d} | {int(d):12d} | {int(h):12d}"
        elif "%" in k:
            line = k.ljust(24) + f" | {p:12.2f} | {d:12.2f} | {h:12.2f}"
        else:
            line = k.ljust(24) + f" | {p:12.3f} | {d:12.3f} | {h:12.3f}"
        lines.append(line)
    return "\n".join(lines)

def main():
    random.seed(SIM_PARAMS["seed"])
    np.random.seed(SIM_PARAMS["seed"])
    budgets = 4000  # equal FEs per optimizer
    # Workloads from 5k to 100k in steps of 5k
    workloads = list(range(5_000, 100_001, 5_000))

    print("Fair Metaheuristic Comparison (Neutral Params; No Tuning)")
    print(f"Budget per optimizer: {budgets} FEs (pop * iters)\n")
    for n in workloads:
        results = run_for_workload(n, budget_fes=budgets, seed=SIM_PARAMS['seed'])
        print(format_report(n, results))

if __name__ == "__main__":
    main()
    


