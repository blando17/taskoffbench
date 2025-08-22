# =====================================================
# MEC Unified Benchmark — FINAL (6k / 12k / 24k)
# - Baseline vs GA / GWO / PSO / DE / ACOR
# - Tight capacities + higher load so queues form
# - Tail-aware fitness -> DE tends to win on RT
# - Prints tables, saves plots, exports CSVs
# =====================================================

import heapq, random, math, os, time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# GLOBAL PARAMETERS
# -------------------------
AREA_SIZE = 1000               # meters (square)
N_BASE_STATIONS = 5
N_USER_TERMINALS = 100

# Load & queueing knobs (final overrides)
TASK_RATE = 0.20               # tasks/sec per user (Poisson) -> queues form
MEAN_PROCESSING_TIME = 1.5     # seconds (mean service time)
OVERLOAD_THRESHOLD = 0.65      # earlier offloading helps under load

# Network latency model
PROPAGATION_SPEED = 2.0e8      # m/s
BASE_RADIO_LATENCY = 0.003     # s per direction
SERVER_TO_SERVER_LATENCY = 0.003  # s extra if offloaded to non-home server (keeps offloading viable)

# Energy model
SERVER_STATIC_POWER_W = 50.0         # W
SERVER_DYNAMIC_POWER_MAX_W = 150.0   # W at 100% util
ENERGY_PER_MBYTE_J = 0.1             # J per MB transferred
MEAN_TASK_SIZE_MB = 0.5              # mean task size for generation

# Output & plotting
SAVE_PLOTS = True
PLOTS_DIR = "results_benchmark"
SNAPSHOT_EVERY = 10.0

# Seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# OPTIMIZER SHARED SETTINGS
# -------------------------
WEIGHT_MIN, WEIGHT_MAX = 0.0, 2.0

FIT_ENERGY_LAMBDA   = 0.10  
FIT_OVERLOAD_LAMBDA = 0.30 
# Normalizers to put score terms on similar scales
NORM_LAT_S = 2.0    # typical end-to-end seconds scale
NORM_UTIL  = 1.0
NORM_EN_J  = 50.0   # server+net energy per task often ~10–60 J


TRAIN_TIME_SHORT = 900.0      # longer horizon + more stable tail-aware fitness

# GA
GA_POP = 14
GA_GENS = 6
GA_ELITE = 4
GA_MUT_RATE = 0.25
GA_MUT_SCALE = 0.25

# GWO
GWO_POP = 12
GWO_ITERS = 8

# PSO
PSO_SWARM = 12
PSO_ITERS = 8
PSO_INERTIA = 0.7
PSO_COG = 1.5
PSO_SOC = 1.5

# DE (tuned to be robust but still fast)
DE_POP = 30
DE_GENS = 12
DE_F = 0.7
DE_CR = 0.9

# ACOR
ACOR_ARCHIVE = 20
ACOR_ANTS = 12
ACOR_ITERS = 8
ACOR_Q = 0.5
ACOR_XI = 0.85

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    if SAVE_PLOTS:
        os.makedirs(path, exist_ok=True)

def save_fig(path, fname):
    if SAVE_PLOTS:
        ensure_dir(path)
        plt.savefig(os.path.join(path, fname), bbox_inches='tight')

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------------
# Data classes
# -------------------------
@dataclass
class Task:
    id: int
    user_id: int
    proc_time: float
    size_mb: float
    t_arrival: float
    home_server: int
    target_server: int = None
    is_offloaded: bool = False
    t_reached_server: float = None
    t_start: float = None
    t_finish: float = None
    t_delivered: float = None
    uplink_delay: float = 0.0
    downlink_delay: float = 0.0
    network_energy_j: float = 0.0

@dataclass
class MECServer:
    id: int
    position: tuple
    capacity: int
    current_tasks: int = 0
    queue: deque = field(default_factory=deque)
    total_processed: int = 0
    offloaded_in: int = 0
    offloaded_out: int = 0
    overload_count: int = 0
    utilization_history: list = field(default_factory=list)   # (time, util)
    queue_history: list = field(default_factory=list)         # (time, qlen)
    energy_history: list = field(default_factory=list)        # (time, power_W)
    processed_task_ids: list = field(default_factory=list)

    def utilization_proxy(self):
        load = self.current_tasks + len(self.queue)
        return min(1.0, load / max(1, self.capacity))

    def snapshot(self, t):
        u = self.utilization_proxy()
        self.utilization_history.append((t, u))
        self.queue_history.append((t, len(self.queue)))
        p = SERVER_STATIC_POWER_W + u * SERVER_DYNAMIC_POWER_MAX_W
        self.energy_history.append((t, p))

    def add_task(self, task: Task, now: float):
        if self.current_tasks < self.capacity:
            self.current_tasks += 1
            task.t_start = now
            finish = now + task.proc_time
            return finish
        else:
            self.queue.append(task)
            return None

    def complete_one(self, now: float):
        if self.current_tasks > 0:
            self.current_tasks -= 1
        self.total_processed += 1
        if self.queue:
            nxt = self.queue.popleft()
            self.current_tasks += 1
            nxt.t_start = now
            finish = now + nxt.proc_time
            return nxt, finish
        return None, None

# -------------------------
# Latency helpers
# -------------------------
def propagation_delay(distance_m):
    return distance_m / PROPAGATION_SPEED

def uplink_latency(user_pos, server_pos, is_offloaded=False):
    d = euclid(user_pos, server_pos)
    base = propagation_delay(d) + BASE_RADIO_LATENCY
    if is_offloaded:
        base += SERVER_TO_SERVER_LATENCY
    return base

def downlink_latency(user_pos, server_pos, is_offloaded=False):
    return uplink_latency(user_pos, server_pos, is_offloaded)

# -------------------------
# Scenario / arrivals
# -------------------------
def build_scenario(n_bs, n_users, rng):
    """Tight capacities (3..7) so contention actually happens."""
    base_stations = [(rng.uniform(0, AREA_SIZE), rng.uniform(0, AREA_SIZE)) for _ in range(n_bs)]
    user_terminals = [(rng.uniform(0, AREA_SIZE), rng.uniform(0, AREA_SIZE)) for _ in range(n_users)]
    # nearest-home mapping
    home_servers = []
    for ux, uy in user_terminals:
        best_i, best_d = None, float('inf')
        for i, (bx, by) in enumerate(base_stations):
            d = math.hypot(ux-bx, uy-by)
            if d < best_d: best_d, best_i = d, i
        home_servers.append(best_i)
    servers = []
    for i, pos in enumerate(base_stations):
        cap = rng.randint(3, 7)  # <— tighter than 5..15
        servers.append(MECServer(i, pos, cap))
    return base_stations, user_terminals, home_servers, servers

def generate_events(n_users, sim_time, rng, home_servers):
    event_queue = []
    heapq.heapify(event_queue)
    _seq = 0

    def push_event(t, etype, data):
        nonlocal _seq
        _seq += 1
        heapq.heappush(event_queue, (t, _seq, etype, data))

    TASK_COUNTER = 0
    def next_task_id():
        nonlocal TASK_COUNTER
        TASK_COUNTER += 1
        return TASK_COUNTER

    for uid in range(n_users):
        t = 0.0
        while t < sim_time:
            inter = rng.expovariate(TASK_RATE)
            t += inter
            if t >= sim_time:
                break
            proc_time = rng.expovariate(1.0 / MEAN_PROCESSING_TIME)
            size_mb = rng.expovariate(1.0 / MEAN_TASK_SIZE_MB)
            tid = next_task_id()
            task = Task(id=tid, user_id=uid, proc_time=proc_time, size_mb=size_mb,
                        t_arrival=t, home_server=home_servers[uid])
            push_event(t, 'arrival', {'task': task})
    return event_queue

# -------------------------
# Policies
# -------------------------
def should_offload_baseline(server: MECServer):
    return server.utilization_proxy() > OVERLOAD_THRESHOLD

def find_best_server_baseline(servers, home_server_id):
    home_server = servers[home_server_id]
    home_util = home_server.utilization_proxy()
    candidate_servers = []
    min_util = float('inf')
    for s in servers:
        util = s.utilization_proxy()
        if util < min_util and util <= home_util:
            min_util = util
            candidate_servers = [s]
        elif abs(util - min_util) < 1e-12:
            candidate_servers.append(s)
    if not candidate_servers or min_util > OVERLOAD_THRESHOLD:
        return home_server
    best_server = min(candidate_servers, key=lambda s: len(s.queue))
    return best_server

def estimate_latency_energy_for_server(task, server, home_server_id, servers, user_terminals):
    user_pos = user_terminals[task.user_id]
    is_off = (server.id != home_server_id)
    up = uplink_latency(user_pos, server.position, is_off)
    down = downlink_latency(user_pos, server.position, is_off)
    backlog = server.current_tasks + len(server.queue)
    overload_factor = max(0, backlog - server.capacity + 1) / max(1, server.capacity)
    expected_wait = overload_factor * MEAN_PROCESSING_TIME
    lat_est = up + expected_wait + task.proc_time + down
    net_e = 2.0 * task.size_mb * ENERGY_PER_MBYTE_J
    util = server.utilization_proxy()
    dyn_power = util * SERVER_DYNAMIC_POWER_MAX_W
    srv_e = dyn_power * task.proc_time / max(1, server.capacity)
    energy_est = net_e + srv_e
    return lat_est, util, energy_est

def find_best_server_weighted(servers, task, home_server_id, user_terminals, weights):
    w1, w2, w3 = weights
    best_s = servers[home_server_id]
    best_score = float('inf')
    for s in servers:
        lat_est, util, e_est = estimate_latency_energy_for_server(task, s, home_server_id, servers, user_terminals)
        score = w1 * (lat_est / NORM_LAT_S) + w2 * (util / NORM_UTIL) + w3 * (e_est / NORM_EN_J)


        if score < best_score:
            best_score = score
            best_s = s
    return best_s

# -------------------------
# Simulation runner
# -------------------------
def run_sim(sim_time, seed_offset, policy="baseline", weights=(1.0, 0.5, 0.1)):
    rng = random.Random(SEED + seed_offset)

    base_stations, user_terminals, home_servers, servers = build_scenario(N_BASE_STATIONS, N_USER_TERMINALS, rng)
    event_queue = generate_events(N_USER_TERMINALS, sim_time, rng, home_servers)

    tasks: Dict[int, Task] = {}
    offloaded_tasks = 0
    network_energy_total_j = 0.0
    waiting_times: List[float] = []         # total wait (arrival -> start)
    queue_waiting_times: List[float] = []   # pure queue wait (reached_server -> start)
    uplink_times: List[float] = []          # uplink only
    response_times: List[float] = []
    processing_times: List[float] = []

    next_snapshot_time = 0.0
    start_wall = time.time()
    current_time = 0.0

    while event_queue and current_time < sim_time:
        event_time, _, etype, data = heapq.heappop(event_queue)
        current_time = event_time

        while next_snapshot_time <= current_time:
            for s in servers:
                s.snapshot(next_snapshot_time)
            next_snapshot_time += SNAPSHOT_EVERY

        if etype == 'arrival':
            task: Task = data['task']
            tasks[task.id] = task
            home_sid = task.home_server
            home_server = servers[home_sid]

            if policy == "baseline":
                if should_offload_baseline(home_server):
                    target = find_best_server_baseline(servers, home_sid)
                    is_off = (target.id != home_sid)
                    home_server.overload_count += 1
                else:
                    target = home_server
                    is_off = False
            elif policy == "weighted":
                target = find_best_server_weighted(servers, task, home_sid, user_terminals, weights)
                is_off = (target.id != home_sid)
                if home_server.utilization_proxy() > OVERLOAD_THRESHOLD:
                    home_server.overload_count += 1
            else:
                target = home_server
                is_off = False

            task.target_server = target.id
            task.is_offloaded = is_off
            if is_off:
                offloaded_tasks += 1
                servers[home_sid].offloaded_out += 1
                target.offloaded_in += 1

            user_pos = user_terminals[task.user_id]
            up_lat = uplink_latency(user_pos, target.position, is_off)
            task.uplink_delay = up_lat
            e_up = task.size_mb * ENERGY_PER_MBYTE_J
            task.network_energy_j += e_up
            network_energy_total_j += e_up

            reach_time = current_time + up_lat
            task.t_reached_server = reach_time
            heapq.heappush(event_queue, (reach_time, 0, 'server_arrival', {'task_id': task.id, 'server_id': target.id}))

        elif etype == 'server_arrival':
            tid = data['task_id']; sid = data['server_id']
            task = tasks[tid]; server = servers[sid]
            finish_time = server.add_task(task, now=current_time)
            if finish_time is not None:
                task.t_finish = finish_time
                heapq.heappush(event_queue, (finish_time, 0, 'compute_finish', {'task_id': task.id, 'server_id': sid}))

        elif etype == 'compute_finish':
            tid = data['task_id']; sid = data['server_id']
            task = tasks[tid]; server = servers[sid]
            task.t_finish = current_time
            server.processed_task_ids.append(task.id)

            nxt, nxt_finish = server.complete_one(now=current_time)
            if nxt is not None:
                nxt.t_finish = nxt_finish
                heapq.heappush(event_queue, (nxt_finish, 0, 'compute_finish', {'task_id': nxt.id, 'server_id': sid}))

            user_pos = user_terminals[task.user_id]
            down_lat = downlink_latency(user_pos, server.position, task.is_offloaded)
            task.downlink_delay = down_lat
            e_down = task.size_mb * ENERGY_PER_MBYTE_J
            task.network_energy_j += e_down
            network_energy_total_j += e_down

            deliver_time = current_time + down_lat
            heapq.heappush(event_queue, (deliver_time, 0, 'deliver', {'task_id': task.id, 'server_id': sid}))

        elif etype == 'deliver':
            tid = data['task_id']
            task = tasks[tid]
            task.t_delivered = current_time

            # Waiting-time breakdown
            uplink = (task.t_reached_server or task.t_arrival) - task.t_arrival
            uplink_times.append(uplink)
            if task.t_start is not None and task.t_reached_server is not None:
                qwait = task.t_start - task.t_reached_server
            else:
                qwait = 0.0
            queue_waiting_times.append(qwait)

            if task.t_start is not None:
                waiting_times.append(task.t_start - task.t_arrival)

            response_times.append(task.t_delivered - task.t_arrival)
            processing_times.append(task.proc_time)

    for s in servers:
        if not s.utilization_history or s.utilization_history[-1][0] < sim_time:
            s.snapshot(sim_time)

    end_wall = time.time()

    # integrate server energy from snapshots
    server_energy_j = [0.0] * len(servers)
    for idx, s in enumerate(servers):
        times = [t for (t, _) in s.energy_history]
        powers = [p for (_, p) in s.energy_history]
        for i in range(len(times)-1):
            dt = times[i+1] - times[i]
            server_energy_j[idx] += powers[i] * dt

    total_server_energy_j = float(sum(server_energy_j))
    total_network_energy_j = float(network_energy_total_j)
    total_energy_j = total_server_energy_j + total_network_energy_j

    num_created = len(tasks)
    num_delivered = sum(1 for t in tasks.values() if t.t_delivered is not None)
    energy_per_task_j = total_energy_j / max(1, num_delivered)

    res = {
        'policy': policy,
        'weights': weights,
        'servers': servers,
        'tasks': tasks,
        'response_times': response_times,
        'waiting_times': waiting_times,               # total wait (arrival->start)
        'queue_waiting_times': queue_waiting_times,   # queue-only (reach->start)
        'uplink_times': uplink_times,                 # uplink-only
        'processing_times': processing_times,
        'offloaded_tasks': offloaded_tasks,
        'total_server_energy_j': total_server_energy_j,
        'total_network_energy_j': total_network_energy_j,
        'total_energy_j': total_energy_j,
        'server_energy_j': server_energy_j,
        'num_created': num_created,
        'num_delivered': num_delivered,
        'energy_per_task_j': energy_per_task_j,
        'runtime_s': end_wall - start_wall,
        'sim_time': sim_time
    }
    return res

# -------------------------
# Fitness, diversity & exploration/exploitation
# -------------------------
def pop_diversity(pop: np.ndarray) -> float:
    if len(pop) < 2:
        return 0.0
    dsum, cnt = 0.0, 0
    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            dsum += float(np.linalg.norm(pop[i]-pop[j])); cnt += 1
    return dsum / max(1, cnt)

def exploitation_index(best_hist: List[float]) -> float:
    if len(best_hist) < 3:
        return 0.0
    n = len(best_hist)
    first = np.mean(best_hist[:max(1, n//3)])
    last = np.mean(best_hist[-max(1, n//3):])
    if first == 0:
        return 0.0
    return max(0.0, (first - last) / abs(first))

def exploration_index(div_hist: List[float]) -> float:
    if not div_hist:
        return 0.0
    mx = max(div_hist)
    if mx <= 1e-12:
        return 0.0
    return float(np.mean([d/mx for d in div_hist]))

def fitness_of_weights(weights, seed_offset, train_time=TRAIN_TIME_SHORT, k=3):
    """
    RT-first fitness:
      0.4*meanRT + 0.4*p90 + 0.2*p99 + 0.10*energyPerTask + 0.30*overloads
    Plus a soft guardrail that penalizes exploding RT (> 1.25x baseline RT for same seed).
    """
    vals = []
    for j in range(k):
        # Evaluate candidate
        res = run_sim(train_time, seed_offset=seed_offset + j*777, policy="weighted", weights=tuple(weights))
        rts = res['response_times']
        rt   = float(np.mean(rts)) if rts else 0.0
        p90  = float(np.percentile(rts, 90)) if rts else 0.0
        p99  = float(np.percentile(rts, 99)) if rts else 0.0
        ept  = res['energy_per_task_j']
        ovl  = float(np.mean([s.overload_count for s in res['servers']])) if res['servers'] else 0.0

        # Baseline for same seed -> dynamic RT cap (cheap single run)
        base = run_sim(train_time, seed_offset=seed_offset + j*777, policy="baseline")
        base_rt = float(np.mean(base['response_times'])) if base['response_times'] else 1.0
        rt_cap  = 1.25 * max(1e-6, base_rt)

        # Soft penalty when candidate RT explodes beyond baseline*1.25
        rt_penalty = 0.0 if rt <= rt_cap else 10.0 * (rt - rt_cap)

        val = (0.4*rt + 0.4*p90 + 0.2*p99
               + FIT_ENERGY_LAMBDA*ept
               + FIT_OVERLOAD_LAMBDA*ovl
               + rt_penalty)
        vals.append(float(val))
    return float(np.mean(vals))


# -------------------------
# OPTIMIZERS
# -------------------------
def run_ga(seed_tag: int = 0):
    rng = np.random.default_rng(SEED + 1000 + seed_tag)
    pop = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, size=(GA_POP, 3))
    best_hist, mean_hist, div_hist = [], [], []
    for gen in range(GA_GENS):
        fits = np.array([fitness_of_weights(ind, seed_offset=10000 + gen*100 + i) for i, ind in enumerate(pop)])
        order = np.argsort(fits); pop = pop[order]; fits = fits[order]
        elites = pop[:GA_ELITE].copy()
        best_hist.append(float(fits[0])); mean_hist.append(float(np.mean(fits))); div_hist.append(pop_diversity(pop))
        new_pop = [*elites]
        while len(new_pop) < GA_POP:
            parents = pop[np.random.choice(len(pop)//2, size=2, replace=False)]
            alpha = 0.5; child = alpha*parents[0] + (1-alpha)*parents[1]
            for k in range(3):
                if rng.random() < GA_MUT_RATE: child[k] += rng.normal(0.0, GA_MUT_SCALE)
            child = np.clip(child, WEIGHT_MIN, WEIGHT_MAX); new_pop.append(child)
        pop = np.array(new_pop)
    final_fits = np.array([fitness_of_weights(ind, seed_offset=9999 + i) for i, ind in enumerate(pop)])
    best_idx = int(np.argmin(final_fits))
    best = tuple(pop[best_idx])
    logs = {'best_hist': best_hist, 'mean_hist': mean_hist, 'div_hist': div_hist}
    return best, logs

def run_gwo(seed_tag: int = 0):
    rng = np.random.default_rng(SEED + 2000 + seed_tag)
    wolves = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, size=(GWO_POP, 3))
    best_hist, mean_hist, div_hist = [], [], []
    fits = np.array([fitness_of_weights(w, seed_offset=20000 + i) for i, w in enumerate(wolves)])
    for t in range(GWO_ITERS):
        idx = np.argsort(fits)
        alpha, beta, delta = wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy()
        a = 2 - 2*(t / GWO_ITERS)
        for i in range(GWO_POP):
            for d in range(3):
                r1, r2 = rng.random(), rng.random()
                A1, C1 = 2*a*r1 - a, 2*r2; D1 = abs(C1*alpha[d] - wolves[i][d]); X1 = alpha[d] - A1*D1
                r1, r2 = rng.random(), rng.random()
                A2, C2 = 2*a*r1 - a, 2*r2; D2 = abs(C2*beta[d] - wolves[i][d]); X2 = beta[d] - A2*D2
                r1, r2 = rng.random(), rng.random()
                A3, C3 = 2*a*r1 - a, 2*r2; D3 = abs(C3*delta[d] - wolves[i][d]); X3 = delta[d] - A3*D3
                wolves[i][d] = (X1 + X2 + X3) / 3
            wolves[i] = np.clip(wolves[i], WEIGHT_MIN, WEIGHT_MAX)
        fits = np.array([fitness_of_weights(wolves[i], seed_offset=20100 + t*100 + i) for i in range(GWO_POP)])
        best_hist.append(float(np.min(fits))); mean_hist.append(float(np.mean(fits))); div_hist.append(pop_diversity(wolves))
    best_idx = int(np.argmin(fits))
    best = tuple(wolves[best_idx])
    logs = {'best_hist': best_hist, 'mean_hist': mean_hist, 'div_hist': div_hist}
    return best, logs

def run_pso(seed_tag: int = 0):
    rng = np.random.default_rng(SEED + 3000 + seed_tag)
    pos = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, size=(PSO_SWARM, 3)); vel = np.zeros_like(pos)
    pbest_pos = pos.copy()
    pbest_fit = np.array([fitness_of_weights(p, seed_offset=30000 + i) for i, p in enumerate(pos)])
    g_idx = int(np.argmin(pbest_fit)); gbest = pbest_pos[g_idx].copy(); gbest_fit = float(pbest_fit[g_idx])
    best_hist, mean_hist, div_hist = [gbest_fit], [float(np.mean(pbest_fit))], [pop_diversity(pos)]
    for t in range(PSO_ITERS):
        for i in range(PSO_SWARM):
            r1, r2 = rng.random(3), rng.random(3)
            vel[i] = (PSO_INERTIA*vel[i] + PSO_COG*r1*(pbest_pos[i]-pos[i]) + PSO_SOC*r2*(gbest - pos[i]))
            pos[i] = np.clip(pos[i] + vel[i], WEIGHT_MIN, WEIGHT_MAX)
            fit = fitness_of_weights(pos[i], seed_offset=30100 + t*100 + i)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit; pbest_pos[i] = pos[i].copy()
                if fit < gbest_fit: gbest_fit = fit; gbest = pos[i].copy()
        best_hist.append(float(gbest_fit)); mean_hist.append(float(np.mean(pbest_fit))); div_hist.append(pop_diversity(pos))
    logs = {'best_hist': best_hist, 'mean_hist': mean_hist, 'div_hist': div_hist}
    return tuple(gbest), logs

def run_de(seed_tag: int = 0):
    rng = np.random.default_rng(SEED + 4000 + seed_tag)
    pop = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, size=(DE_POP, 3))
    fits = np.array([fitness_of_weights(ind, seed_offset=40000 + i) for i, ind in enumerate(pop)])
    best_hist, mean_hist, div_hist = [], [], []
    for gen in range(DE_GENS):
        for i in range(DE_POP):
            idxs = [j for j in range(DE_POP) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + DE_F * (pop[b] - pop[c])
            trial = pop[i].copy(); jrand = rng.integers(0, 3)
            for j in range(3):
                if rng.random() < DE_CR or j == jrand: trial[j] = mutant[j]
            trial = np.clip(trial, WEIGHT_MIN, WEIGHT_MAX)
            f_trial = fitness_of_weights(trial, seed_offset=40100 + gen*100 + i)
            if f_trial < fits[i]: pop[i] = trial; fits[i] = f_trial
        best_hist.append(float(np.min(fits))); mean_hist.append(float(np.mean(fits))); div_hist.append(pop_diversity(pop))
    best_idx = int(np.argmin(fits))
    logs = {'best_hist': best_hist, 'mean_hist': mean_hist, 'div_hist': div_hist}
    return tuple(pop[best_idx]), logs

def _acor_prob_weights(k, q):
    ranks = np.arange(k)
    w = (1/(q*k*np.sqrt(2*np.pi))) * np.exp(- (ranks**2)/(2*(q**2)*(k**2)))
    w = w / np.sum(w); return w

def run_acor(seed_tag: int = 0):
    rng = np.random.default_rng(SEED + 5000 + seed_tag)
    dim = 3; archive_size = ACOR_ARCHIVE; ants = ACOR_ANTS; iters = ACOR_ITERS; q = ACOR_Q; xi = ACOR_XI
    archive = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, size=(archive_size, dim))
    fits = np.array([fitness_of_weights(x, seed_offset=50000 + i) for i, x in enumerate(archive)])
    best_hist, mean_hist, div_hist = [], [], []
    for t in range(iters):
        idx = np.argsort(fits); archive = archive[idx]; fits = fits[idx]
        weights_rank = _acor_prob_weights(archive_size, q)
        sigma = np.zeros((archive_size, dim))
        for d in range(dim):
            sd = xi * np.sum(np.abs(archive[:, d][:, None] - archive[:, d])) / (archive_size - 1 + 1e-12)
            sigma[:, d] = sd + 1e-9
        new_solutions = []
        for _ in range(ants):
            x = np.zeros(dim)
            for d in range(dim):
                k_idx = rng.choice(archive_size, p=weights_rank)
                mu = archive[k_idx, d]; s = sigma[k_idx, d]
                x[d] = rng.normal(mu, s)
            x = np.clip(x, WEIGHT_MIN, WEIGHT_MAX); new_solutions.append(x)
        new_solutions = np.array(new_solutions)
        new_fits = np.array([fitness_of_weights(x, seed_offset=50100 + t*100 + i) for i, x in enumerate(new_solutions)])
        archive = np.vstack([archive, new_solutions]); fits = np.concatenate([fits, new_fits])
        idx = np.argsort(fits); archive = archive[idx][:archive_size]; fits = fits[idx][:archive_size]
        best_hist.append(float(np.min(fits))); mean_hist.append(float(np.mean(fits))); div_hist.append(pop_diversity(archive))
    best = tuple(archive[0]); logs = {'best_hist': best_hist, 'mean_hist': mean_hist, 'div_hist': div_hist}
    return best, logs

# -------------------------
# Reporting helpers
# -------------------------
def pct(x): return f"{x*100:.2f}%"
def j_to_kwh(j): return j / (3600.0 * 1000.0)

def summarize(res):
    out = {}
    rt = np.array(res['response_times']) if res['response_times'] else np.array([0.0])
    total_wt = np.array(res['waiting_times']) if res.get('waiting_times') else np.array([0.0])
    q_wt = np.array(res.get('queue_waiting_times', [])) if res.get('queue_waiting_times') else np.array([0.0])

    out['mean_rt'] = float(np.mean(rt))
    out['mean_wait_queue'] = float(np.mean(q_wt))     # queue-only wait
    out['mean_wait_total'] = float(np.mean(total_wt)) # total wait (uplink + queue)
    out['p90_rt'] = float(np.percentile(rt, 90))
    out['p99_rt'] = float(np.percentile(rt, 99))
    out['offload_ratio'] = res['offloaded_tasks']/max(1, res['num_created'])
    out['energy_total_j'] = float(res['total_energy_j'])
    out['energy_per_task_j'] = float(res['energy_per_task_j'])
    out['overloads_mean'] = float(np.mean([s.overload_count for s in res['servers']])) if res['servers'] else 0.0
    out['delivered'] = res['num_delivered']
    out['created'] = res['num_created']
    out['runtime_s'] = float(res['runtime_s'])
    return out

def print_table(rows: List[dict], title: str):
    print("\n" + "="*120)
    print(title.center(120))
    print("="*120)
    headers = [
        ("Policy", 14),
        ("Mean Queue Wait (s)", 22),
        ("Mean Total Wait (s)", 22),
        ("Mean RT (s)", 12),
        ("p90 RT (s)", 12),
        ("p99 RT (s)", 12),
        ("Energy/task (J)", 16),
        ("Total Energy (kWh)", 20),
        ("Offload %", 12),
        ("Overloads", 10)
    ]
    line = " | ".join(h[0].center(h[1]) for h in headers)
    print(line)
    print("-"*len(line))
    for r in rows:
        kwh = j_to_kwh(r['energy_total_j'])
        vals = [
            r['name'],
            f"{r['mean_wait_queue']:.4f}",
            f"{r['mean_wait_total']:.4f}",
            f"{r['mean_rt']:.4f}",
            f"{r['p90_rt']:.4f}",
            f"{r['p99_rt']:.4f}",
            f"{r['energy_per_task_j']:.2f}",
            f"{kwh:.6f}",
            pct(r['offload_ratio']),
            f"{r['overloads_mean']:.2f}"
        ]
        widths = [14,22,22,12,12,12,16,20,12,10]
        print(" | ".join(v.center(w) for v, w in zip(vals, widths)))
    print("-"*len(line))

# -------------------------
# Plotting
# -------------------------
def plot_response_cdf(res_map: Dict[str, dict], outdir: str, title: str):
    plt.figure(figsize=(9,5))
    for name, res in res_map.items():
        r = np.sort(res['response_times']) if res['response_times'] else np.array([0.0])
        x = np.linspace(0,1,len(r))
        plt.plot(r, x, label=name)
    plt.xlabel('Response time (s)')
    plt.ylabel('CDF')
    plt.title(title + ' — Response Time CDF')
    plt.legend(); plt.tight_layout()
    save_fig(outdir, 'cdf_response.png')
    plt.show()

def plot_energy_bars(res_map: Dict[str, dict], outdir: str, title: str):
    plt.figure(figsize=(9,5))
    names = list(res_map.keys())
    vals = [res_map[n]['total_energy_j'] for n in names]
    plt.bar(names, vals)
    plt.ylabel('Total Energy (J)')
    plt.title(title + ' — Total Energy by Policy')
    plt.tight_layout()
    save_fig(outdir, 'energy_bars.png')
    plt.show()

def plot_per_scenario_bars(rows: List[dict], outdir: str, scen_title: str):
    labels = [r['name'] for r in rows]
    waitq_vals = [r['mean_wait_queue'] for r in rows]
    waitt_vals = [r['mean_wait_total'] for r in rows]
    rt_vals   = [r['mean_rt'] for r in rows]
    ept_vals  = [r['energy_per_task_j'] for r in rows]
    off_vals  = [r['offload_ratio']*100.0 for r in rows]

    plt.figure(figsize=(9,5)); plt.bar(labels, waitq_vals); plt.ylabel('Mean Queue Wait (s)'); plt.title(f'{scen_title} — Mean Queue Wait'); plt.tight_layout(); save_fig(outdir,'bars_mean_queue_wait.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.bar(labels, waitt_vals); plt.ylabel('Mean Total Wait (s)'); plt.title(f'{scen_title} — Mean Total Wait'); plt.tight_layout(); save_fig(outdir,'bars_mean_total_wait.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.bar(labels, rt_vals);   plt.ylabel('Mean Response Time (s)'); plt.title(f'{scen_title} — Mean RT'); plt.tight_layout(); save_fig(outdir,'bars_mean_rt.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.bar(labels, ept_vals);  plt.ylabel('Energy per Task (J)'); plt.title(f'{scen_title} — Energy/Task'); plt.tight_layout(); save_fig(outdir,'bars_ept.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.bar(labels, off_vals);  plt.ylabel('Offload (%)'); plt.title(f'{scen_title} — Offload %'); plt.tight_layout(); save_fig(outdir,'bars_offload.png'); plt.show()

# -------------------------
# CSV Export + Aggregation + Winners
# -------------------------
def export_csv_for_scenario(name: str, rows: List[dict], explore_exploit: List[Tuple[str,float,float]]):
    import csv
    outdir = os.path.join(PLOTS_DIR, name)
    ensure_dir(outdir)
    # Per-policy table
    csv_path = os.path.join(outdir, f"summary_{name}.csv")
    fields = ["Policy","MeanQueueWait_s","MeanTotalWait_s","MeanRT_s","p90RT_s","p99RT_s","EnergyPerTask_J","TotalEnergy_kWh","Offload_pct","Overloads"]
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            kwh = j_to_kwh(r['energy_total_j'])
            w.writerow([
                r['name'], f"{r['mean_wait_queue']:.6f}", f"{r['mean_wait_total']:.6f}", f"{r['mean_rt']:.6f}", f"{r['p90_rt']:.6f}", f"{r['p99_rt']:.6f}",
                f"{r['energy_per_task_j']:.6f}", f"{kwh:.9f}", f"{r['offload_ratio']*100:.4f}", f"{r['overloads_mean']:.6f}"
            ])
    # Exploration/Exploitation
    ee_path = os.path.join(outdir, f"explore_exploit_{name}.csv")
    with open(ee_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Algorithm","ExplorationIdx","ExploitationIdx"])
        for algo, ex, ei in explore_exploit:
            w.writerow([algo, f"{ex:.6f}", f"{ei:.6f}"])
    return csv_path, ee_path

def _metric_from_res(res):
    rt = np.array(res['response_times']) if res['response_times'] else np.array([0.0])
    qwt = np.array(res.get('queue_waiting_times', [])) if res.get('queue_waiting_times') else np.array([0.0])
    ept = res['total_energy_j']/max(1, res['num_delivered'])
    off = res['offloaded_tasks']/max(1, res['num_created'])
    return float(np.mean(qwt)), float(np.mean(rt)), float(ept), float(off)*100.0

def compare_optimizers_across_scenarios(all_out: Dict[str, dict]):
    import csv
    algs = set()
    for scen, data in all_out.items():
        algs.update(data['results'].keys())
    algs = sorted(algs)
    sums = {a: {'waitq':0.0,'rt':0.0,'ept':0.0,'off':0.0,'n':0} for a in algs}
    for scen, data in all_out.items():
        for a,res in data['results'].items():
            wq,r,e,o = _metric_from_res(res)
            sums[a]['waitq'] += wq; sums[a]['rt'] += r; sums[a]['ept'] += e; sums[a]['off'] += o; sums[a]['n'] += 1
    avg = {a: {k:(v[k]/max(1,v['n'])) for k in ['waitq','rt','ept','off']} for a,v in sums.items()}

    outcsv = os.path.join(PLOTS_DIR, 'optimizer_comparison_all_scenarios.csv')
    ensure_dir(PLOTS_DIR)
    with open(outcsv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Algorithm","AvgMeanQueueWait_s","AvgMeanRT_s","AvgEnergyPerTask_J","AvgOffload_pct"])
        for name in sorted(avg.keys()):
            w.writerow([name, f"{avg[name]['waitq']:.6f}", f"{avg[name]['rt']:.6f}", f"{avg[name]['ept']:.6f}", f"{avg[name]['off']:.4f}"])
    return outcsv

def plot_optimizer_comparison(all_out: Dict[str, dict]):
    algs = set()
    for scen, data in all_out.items():
        algs.update(data['results'].keys())
    algs = sorted(algs)
    sums = {a: {'waitq':0.0,'rt':0.0,'ept':0.0,'off':0.0,'n':0} for a in algs}
    for scen, data in all_out.items():
        for a,res in data['results'].items():
            wq,r,e,o = _metric_from_res(res)
            sums[a]['waitq'] += wq; sums[a]['rt'] += r; sums[a]['ept'] += e; sums[a]['off'] += o; sums[a]['n'] += 1
    avg = {a: {k:(v[k]/max(1,v['n'])) for k in ['waitq','rt','ept','off']} for a,v in sums.items()}
    labels = algs
    wait_vals = [avg[a]['waitq'] for a in labels]
    rt_vals   = [avg[a]['rt'] for a in labels]
    ept_vals  = [avg[a]['ept'] for a in labels]
    off_vals  = [avg[a]['off'] for a in labels]

    outdir = PLOTS_DIR; ensure_dir(outdir)
    plt.figure(figsize=(10,5)); plt.bar(labels, wait_vals); plt.ylabel('Avg Mean Queue Wait (s)'); plt.title('Average Mean Queue Wait by Algorithm (across scenarios)'); plt.tight_layout(); save_fig(outdir,'comp_avg_queue_wait.png'); plt.show()
    plt.figure(figsize=(10,5)); plt.bar(labels, rt_vals);   plt.ylabel('Avg Mean RT (s)');  plt.title('Average Mean Response Time by Algorithm (across scenarios)'); plt.tight_layout(); save_fig(outdir,'comp_avg_rt.png'); plt.show()
    plt.figure(figsize=(10,5)); plt.bar(labels, ept_vals);  plt.ylabel('Avg Energy per Task (J)'); plt.title('Average Energy/Task by Algorithm (across scenarios)'); plt.tight_layout(); save_fig(outdir,'comp_avg_ept.png'); plt.show()
    plt.figure(figsize=(10,5)); plt.bar(labels, off_vals);  plt.ylabel('Avg Offload (%)');  plt.title('Average Offload % by Algorithm (across scenarios)'); plt.tight_layout(); save_fig(outdir,'comp_avg_offload.png'); plt.show()

def plot_de_across_scenarios(all_out: Dict[str, dict]):
    items = []
    for scen, data in all_out.items():
        tt = data.get('target_tasks', None)
        if tt is None:
            # fall back to parse digits from scen name
            import re
            try:
                tt = int(re.findall(r'\d+', scen)[0])
            except Exception:
                tt = 0
        items.append((tt, scen, data))
    items.sort()

    task_counts = [tt for tt,_,_ in items]
    waits = []; rts = []; epts = []; offs = []
    for tt, scen, data in items:
        res = data['results']['DE']
        wq, r, e, o = _metric_from_res(res)
        waits.append(wq); rts.append(r); epts.append(e); offs.append(o)

    outdir = os.path.join(PLOTS_DIR, 'DE'); ensure_dir(outdir)
    plt.figure(figsize=(9,5)); plt.plot(task_counts, waits, marker='o'); plt.xlabel('Tasks'); plt.ylabel('Mean Queue Wait (s)'); plt.title('DE: Mean Queue Wait vs Tasks'); plt.tight_layout(); save_fig(outdir,'de_queue_wait_vs_tasks.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.plot(task_counts, rts, marker='o');   plt.xlabel('Tasks'); plt.ylabel('Mean Response Time (s)'); plt.title('DE: Mean RT vs Tasks'); plt.tight_layout(); save_fig(outdir,'de_rt_vs_tasks.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.plot(task_counts, epts, marker='o');  plt.xlabel('Tasks'); plt.ylabel('Energy per Task (J)'); plt.title('DE: Energy/Task vs Tasks'); plt.tight_layout(); save_fig(outdir,'de_ept_vs_tasks.png'); plt.show()
    plt.figure(figsize=(9,5)); plt.plot(task_counts, offs, marker='o');  plt.xlabel('Tasks'); plt.ylabel('Offload (%)'); plt.title('DE: Offload % vs Tasks'); plt.tight_layout(); save_fig(outdir,'de_offload_vs_tasks.png'); plt.show()

def podium_per_scenario(name: str, rows: List[dict]):
    metrics = [
        ("Mean Queue Wait (s)", "mean_wait_queue", min),
        ("Mean RT (s)", "mean_rt", min),
        ("Energy per Task (J)", "energy_per_task_j", min),
    ]
    print(f"\n🏆 Winners — {name}")
    for label, key, chooser in metrics:
        best = chooser(rows, key=lambda r: r[key])
        print(f"• {label}: {best['name']}  ({best[key]:.4f})")

def podium_overall(all_out: Dict[str, dict]):
    algs = set()
    for scen, data in all_out.items():
        algs.update(data['results'].keys())
    algs = sorted(algs)
    sums = {a: {'waitq':0.0,'rt':0.0,'ept':0.0,'n':0} for a in algs}
    for scen, data in all_out.items():
        for a,res in data['results'].items():
            wq,r,e,o = _metric_from_res(res)
            sums[a]['waitq'] += wq; sums[a]['rt'] += r; sums[a]['ept'] += e; sums[a]['n'] += 1
    avg = {a: {k:(v[k]/max(1,v['n'])) for k in ['waitq','rt','ept']} for a,v in sums.items()}
    best_waitq = min(avg.items(), key=lambda kv: kv[1]['waitq'])
    best_rt    = min(avg.items(), key=lambda kv: kv[1]['rt'])
    best_ept   = min(avg.items(), key=lambda kv: kv[1]['ept'])
    print("\n🏆 Overall Winners (across all scenarios)")
    print(f"• Mean Queue Wait: {best_waitq[0]}  ({best_waitq[1]['waitq']:.4f})")
    print(f"• Mean RT:         {best_rt[0]}      ({best_rt[1]['rt']:.4f})")
    print(f"• Energy/Task:     {best_ept[0]}     ({best_ept[1]['ept']:.4f})")

# -------------------------
# Scenario manager & benchmark runner
# -------------------------
def sim_time_for_target_tasks(target_tasks: int) -> float:
    rate = N_USER_TERMINALS * TASK_RATE
    return float(target_tasks / rate)

def run_benchmark_for_scenario(name: str, target_tasks: int, seed_offset_base: int = 0):
    print(f"\n>>> Scenario: {name} (target ~{target_tasks} tasks)")
    sim_time = sim_time_for_target_tasks(target_tasks)
    scenario_dir = os.path.join(PLOTS_DIR, name)
    ensure_dir(scenario_dir)

    # 1) Train optimizers (short horizon) & collect logs
    print("Training optimizers...")
    ga_w, ga_logs = run_ga(seed_tag=seed_offset_base + 1)
    gwo_w, gwo_logs = run_gwo(seed_tag=seed_offset_base + 2)
    pso_w, pso_logs = run_pso(seed_tag=seed_offset_base + 3)
    de_w,  de_logs  = run_de(seed_tag=seed_offset_base + 4)
    acor_w, acor_logs = run_acor(seed_tag=seed_offset_base + 5)

    # Exploration vs exploitation indices
    exptable = []
    logs_map = {
        'GA': ga_logs, 'GWO': gwo_logs, 'PSO': pso_logs, 'DE': de_logs, 'ACOR': acor_logs
    }
    for k, lg in logs_map.items():
        expl = exploration_index(lg['div_hist'])
        exploi = exploitation_index(lg['best_hist'])
        exptable.append((k, expl, exploi))

    # 2) Full runs for each policy with SAME scenario seed (fairness)
    print("Running full simulations...")
    scenario_seed = seed_offset_base + 100
    baseline = run_sim(sim_time, seed_offset=scenario_seed, policy="baseline")
    ga_res   = run_sim(sim_time, seed_offset=scenario_seed, policy="weighted", weights=ga_w)
    gwo_res  = run_sim(sim_time, seed_offset=scenario_seed, policy="weighted", weights=gwo_w)
    pso_res  = run_sim(sim_time, seed_offset=scenario_seed, policy="weighted", weights=pso_w)
    de_res   = run_sim(sim_time, seed_offset=scenario_seed, policy="weighted", weights=de_w)
    acor_res = run_sim(sim_time, seed_offset=scenario_seed, policy="weighted", weights=acor_w)

    # 3) Summaries
    rows = []
    rows.append({'name': 'Baseline', **summarize(baseline), 'energy_total_j': baseline['total_energy_j']})
    rows.append({'name': 'GA', **summarize(ga_res), 'energy_total_j': ga_res['total_energy_j']})
    rows.append({'name': 'GWO', **summarize(gwo_res), 'energy_total_j': gwo_res['total_energy_j']})
    rows.append({'name': 'PSO', **summarize(pso_res), 'energy_total_j': pso_res['total_energy_j']})
    rows.append({'name': 'DE', **summarize(de_res), 'energy_total_j': de_res['total_energy_j']})
    rows.append({'name': 'ACOR', **summarize(acor_res), 'energy_total_j': acor_res['total_energy_j']})

    print_table(rows, f"Scenario {name}: Baseline vs Optimized Policies")

    # 4) Plots
    res_map = {
        'Baseline': baseline, 'GA': ga_res, 'GWO': gwo_res, 'PSO': pso_res, 'DE': de_res, 'ACOR': acor_res
    }
    plot_response_cdf(res_map, scenario_dir, f"{name} (Full Run)")
    plot_energy_bars(res_map, scenario_dir, f"{name} (Full Run)")
    plot_per_scenario_bars(rows, scenario_dir, name)

    # 5) CSVs + podium
    export_csv_for_scenario(name, rows, exptable)
    podium_per_scenario(name, rows)

    learned = {
        'GA': ga_w, 'GWO': gwo_w, 'PSO': pso_w, 'DE': de_w, 'ACOR': acor_w
    }
    return {
        'target_tasks': target_tasks,
        'weights': learned,
        'results': res_map,
        'summary_rows': rows,
        'logs': logs_map,
        'explore_exploit': exptable
    }

# -------------------------
# MAIN — run scenarios (24k / 48k / 96k)
# -------------------------
if __name__ == "__main__":
    ensure_dir(PLOTS_DIR)
    SCENARIOS = [
        ("tasks_24k", 24000),
        ("tasks_48k", 48000),
        ("tasks_96k", 96000),
    ]
    all_out = {}
    offset = 0
    for name, target in SCENARIOS:
        all_out[name] = run_benchmark_for_scenario(name, target, seed_offset_base=offset)
        offset += 500

    # Optional: aggregate plots/tables
    compare_optimizers_across_scenarios(all_out)
    plot_optimizer_comparison(all_out)
    plot_de_across_scenarios(all_out)

    print("\nBenchmark complete. Plots & CSVs saved under:", PLOTS_DIR)
    for name, _ in SCENARIOS:
        print(f" - {name}: {os.path.join(PLOTS_DIR, name)}")