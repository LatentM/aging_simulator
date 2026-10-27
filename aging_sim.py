"""
Simulation of preemptive priority CPU scheduling with aging.
Sweeps the aging interval A and measures waiting-time statistics.

Model
- N processes, Poisson arrivals with rate lambda = load / mean_burst
- burst times: exponential, mean 20 ms, rounded up to whole ms (min 1)
- base priority: uniform integer 1..10 (1 = highest)
- effective priority: max(1, base - floor(wait / A)), wait = total ready-queue time so far
- scheduler re-evaluates every 1 ms tick; best effective priority runs,
  ties broken by arrival time; a better waiting process preempts the running one
- A = 0 means aging switched off
- adaptive variant (after Becker and Schule): the promotion step is not fixed but
  equals K times the current mean waiting time of the ready processes,
  A(t) = max(1, K * mean ready wait); levels gained are never taken back,
  so p_eff = max(1, base - g), with g = largest floor(wait / A(t)) reached so far
- load gate (optional): the adaptive rule is active only while at least G processes
  are in the ready queue; below that, scheduling uses base priorities only
"""
import numpy as np, json, sys
from scipy.stats import wilcoxon

MEAN_BURST = 20
LEVELS = 10
N = 150
SEEDS = 20
INTERVALS = [0, 400, 200, 100, 50, 25, 10]
FACTORS = [0.25, 0.5, 1.0, 1.7, 3.0]  # adaptive policy, K values; 1.7 is the value used by Becker and Schule
GATED = [(k, g) for g in (4, 8) for k in (0.25, 0.5, 1.0, 1.7)]  # (K, gate G) pairs
LOADS = [0.7, 0.9, 1.1]
STARVE_MS = 500  # absolute starvation threshold used for the count


def gen_workload(seed, load):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(MEAN_BURST / load, N)
    arrival = np.floor(np.cumsum(gaps)).astype(int)
    burst = np.maximum(1, np.ceil(rng.exponential(MEAN_BURST, N))).astype(int)
    prio = rng.integers(1, LEVELS + 1, N)
    return arrival, burst, prio


def run(arrival, burst, prio, A, K=None, G=0):
    remaining = burst.copy()
    wait = np.zeros(N, dtype=int)
    done = np.zeros(N, dtype=bool)
    finish = np.zeros(N, dtype=int)
    t = 0
    running = -1
    preemptions = 0
    promo = np.zeros(N, dtype=int)  # levels gained so far (adaptive policy)
    while not done.all():
        ready = (arrival <= t) & ~done
        if not ready.any():
            t += 1
            continue
        if K is not None and ready.sum() >= G:
            step = max(1.0, K * wait[ready].mean())  # adaptive aging step
            # promotion is never reversed, as for the fixed interval
            promo = np.maximum(promo, np.floor(wait / step).astype(int))
            eff = np.maximum(1, prio - promo)
        elif K is not None:
            eff = prio  # load gate closed: base priorities only
        elif A > 0:
            eff = np.maximum(1, prio - wait // A)
        else:
            eff = prio
        idx = np.flatnonzero(ready)
        # pick min eff, then min arrival
        order = np.lexsort((arrival[idx], eff[idx]))
        chosen = idx[order[0]]
        if running >= 0 and running != chosen and not done[running]:
            preemptions += 1
        running = chosen
        remaining[chosen] -= 1
        others = ready.copy()
        others[chosen] = False
        wait[others] += 1
        t += 1
        if remaining[chosen] == 0:
            done[chosen] = True
            finish[chosen] = t
            running = -1
    return wait, preemptions


def stats(wait, prio, burst, preemptions):
    hi = wait[prio == 1].mean() if (prio == 1).any() else np.nan
    lo = wait[prio == LEVELS].mean() if (prio == LEVELS).any() else np.nan
    return dict(
        avg=wait.mean(), max=wait.max(), p95=np.percentile(wait, 95), p99=np.percentile(wait, 99),
        starved=int((wait > STARVE_MS).sum()),
        hi=hi, lo=lo,
        preempt=preemptions / N,
    )


def main():
    results = {}
    T95 = 2.093  # t value, 95%, 19 degrees of freedom
    policies = INTERVALS + ["k%g" % k for k in FACTORS] + ["k%gg%d" % kg for kg in GATED]
    for load in LOADS:
        per_A = {}
        for A in policies:
            acc = []
            for s in range(SEEDS):
                arrival, burst, prio = gen_workload(1000 * s + 7, load)
                if isinstance(A, str) and "g" in A:
                    k, g = A[1:].split("g")
                    wait, pre = run(arrival, burst, prio, 0, K=float(k), G=int(g))
                elif isinstance(A, str):
                    wait, pre = run(arrival, burst, prio, 0, K=float(A[1:]))
                else:
                    wait, pre = run(arrival, burst, prio, A)
                acc.append(stats(wait, prio, burst, pre))
            per_A[A] = acc
        base = per_A[0]  # same seeds are used for every policy, so results are paired
        # adaptive K = 1.7 against the fixed intervals of 100 ms and 50 ms, paired by seed
        for A in (100, 50):
            for k in ("max", "p99", "hi", "lo"):
                results[f"{load}|cmp_k1.7_vs_{A}|{k}"] = float(wilcoxon(
                    [a[k] for a in per_A["k1.7"]], [b[k] for b in per_A[A]]).pvalue)
        # gated against ungated adaptive rule with the same K, paired by seed
        for k_, g_ in GATED:
            for k in ("max", "hi", "lo"):
                results[f"{load}|cmp_k{k_:g}g{g_}_vs_k{k_:g}|{k}"] = float(wilcoxon(
                    [a[k] for a in per_A["k%gg%d" % (k_, g_)]], [b[k] for b in per_A["k%g" % k_]]).pvalue)
        for A in policies:
            acc = per_A[A]
            agg = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
            # 95% confidence interval of the mean over the 20 seeds
            for k in ("avg", "max", "p95", "p99"):
                agg[k + "_ci"] = float(T95 * np.std([a[k] for a in acc], ddof=1) / np.sqrt(SEEDS))
            # paired change from aging off, seed by seed
            d_avg = np.array([a["avg"] - b["avg"] for a, b in zip(acc, base)])
            r_max = np.array([a["max"] / b["max"] for a, b in zip(acc, base)])
            agg["d_avg"] = float(d_avg.mean())
            agg["d_avg_ci"] = float(T95 * d_avg.std(ddof=1) / np.sqrt(SEEDS))
            agg["max_ratio"] = float(r_max.mean())
            agg["max_ratio_ci"] = float(T95 * r_max.std(ddof=1) / np.sqrt(SEEDS))
            # paired Wilcoxon signed-rank test against aging off, seed by seed
            if A != 0:
                for k in ("max", "p95", "avg"):
                    agg["p_" + k] = float(wilcoxon([a[k] for a in acc], [b[k] for b in base]).pvalue)
            # differentiation ratio pooled across seeds, with a bootstrap 95% CI
            agg["ratio"] = agg["lo"] / agg["hi"]
            rng = np.random.default_rng(1)
            lo_s = np.array([a["lo"] for a in acc]); hi_s = np.array([a["hi"] for a in acc])
            boots = []
            for _ in range(2000):
                i = rng.integers(0, SEEDS, SEEDS)
                boots.append(lo_s[i].mean() / hi_s[i].mean())
            agg["ratio_lo"], agg["ratio_hi"] = map(float, np.percentile(boots, [2.5, 97.5]))
            results[f"{load}|{A}"] = agg
            print(load, A, {k: round(v, 2) for k, v in agg.items()}, flush=True)
    json.dump(results, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

