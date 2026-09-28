import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Kubernetes Economics", layout="wide")
st.title("Sub-300 ms ML Inference: EKS vs ECS vs Cloudflare Workers Economics")
st.caption(
    "One CPU model serving an app. EKS and ECS run on EC2 nodes or on Fargate, behind an ALB; Cloudflare "
    "Workers runs the model inside the isolate (WASM, CPU only). The SLO is end to end: "
    "user → (backend →) model → back. Each option picks its cheapest setup that meets it. "
    "Every fixed number is listed in the appendix."
)

HOURS_PER_MONTH = 730
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600

# --- Presets & fixed assumptions (documented in the appendix) -----------------

# On-demand Linux from the AWS Price List API, 2026-09. Node = c6i.xlarge (4 vCPU, 8 GiB).
# Fargate = Linux/x86 per vCPU-hour and per GB-hour (EKS on Fargate uses the same prices).
# ALB is billed per LCU-hour; with keep-alive clients the binding LCU dimension
# is processed bytes (1 GB = 1 LCU-hour).
REGIONS = {
    "us-east-1 (N. Virginia)":   dict(node_hr=0.170, fargate_vcpu_hr=0.04048, fargate_gb_hr=0.004445, alb_hr=0.0225, lcu_hr=0.008, egress_gb=0.090),
    "eu-west-1 (Ireland)":       dict(node_hr=0.1824, fargate_vcpu_hr=0.04048, fargate_gb_hr=0.004445, alb_hr=0.0252, lcu_hr=0.008, egress_gb=0.090),
    "eu-central-1 (Frankfurt)":  dict(node_hr=0.194, fargate_vcpu_hr=0.04656, fargate_gb_hr=0.00511, alb_hr=0.0270, lcu_hr=0.008, egress_gb=0.090),
    "ap-southeast-1 (Singapore)": dict(node_hr=0.196, fargate_vcpu_hr=0.05056, fargate_gb_hr=0.00553, alb_hr=0.0252, lcu_hr=0.008, egress_gb=0.120),
    "ap-northeast-1 (Tokyo)":    dict(node_hr=0.214, fargate_vcpu_hr=0.05056, fargate_gb_hr=0.00553, alb_hr=0.0243, lcu_hr=0.008, egress_gb=0.114),
}
EKS_CONTROL_PLANE_HR = 0.10  # same in every region (standard support)

# Cloudflare Workers Paid, Standard usage model — one global price list.
CF_BASE = 5.0
CF_INCLUDED_REQ = 10e6
CF_PER_M_REQ = 0.30
CF_INCLUDED_CPU_MS = 30e6
CF_PER_M_CPU_MS = 0.02

# p99 round-trip time from users for one request on an open connection (handshakes
# are added separately). `pops` is how many edge locations the traffic spreads
# across — it drives how often Workers isolates and CloudFront origin connections
# go cold. A global audience can't reach one AWS region under ~250 ms p99 (fiber
# alone is 160–200 ms for antipodal users), so AWS — and an app backend — may run
# in 3 regions instead (`rtt_aws_multi`).
AUDIENCES = {
    "Same metro as the AWS region": dict(rtt_aws=30, rtt_cf=20, pops=3),
    "Same continent": dict(rtt_aws=90, rtt_cf=30, pops=40),
    "Global": dict(rtt_aws=250, rtt_aws_multi=90, rtt_cf=50, pops=150),
}
MULTI_REGIONS = ["us-east-1 (N. Virginia)", "eu-west-1 (Ireland)", "ap-southeast-1 (Singapore)"]

# How the app reaches the model.
#  Directly: phones/browsers open connections often enough that well over 1% of
#  requests pay a handshake, so it's inside p99. TCP + TLS 1.3 costs 2 extra round
#  trips to an ALB (no HTTP/3); HTTP/3 (QUIC) costs 1 at Cloudflare's or
#  CloudFront's edge.
#  Through the app's backend: the backend runs on the same platform as the model,
#  and the SLO covers app → backend → model → backend → app. On AWS the backend runs
#  in the region (or 3 regions for a global audience) behind an HTTP/3 edge and calls
#  the model in-region over pooled connections. On Workers the backend is itself a
#  Worker at the PoP nearest the user and calls the model Worker through a service
#  binding: same location, no network hop.
CALLERS = {
    "Directly from the app": dict(direct=True),
    "Through the app's backend (on the same platform)": dict(direct=False),
}
HANDSHAKE_RTTS_TCP_TLS = 2
HANDSHAKE_RTTS_QUIC = 1
BACKEND_TO_AWS_MS = 2  # in-region call over an internal ALB
BACKEND_TO_CF_MS = 0  # service binding: the model Worker runs in the same location

# CloudFront in front of the ALB (direct calls only). HTTP/3 handshake at a nearby
# edge; the request rides a keep-alive connection to the origin — if one is open.
# Origin keep-alive defaults to 5 s, configurable to 60 s; at low volume per edge
# it idles out and the request pays a TCP + TLS handshake across the backbone.
CLOUDFRONT_PER_M_REQ = 1.00  # $0.0100 per 10k HTTPS requests (US); first 10M/month free
CLOUDFRONT_FREE_REQ = 10e6
CLOUDFRONT_EGRESS_GB = 0.085  # first 1 TB/month free; replaces EC2 egress
CLOUDFRONT_FREE_GB = 1000
CLOUDFRONT_TO_ORIGIN_GB = 0.02  # request bodies forwarded to the origin
CLOUDFRONT_KEEPALIVE_S = 60

# Keeping Workers warm: synthetic HTTP probes from a monitoring service ping the
# Worker from each probe location often enough that its PoP's isolate never idles
# out. Probes can only reach the PoPs their locations map to — the long tail of
# small PoPs stays on demand.
PROBE_LOCATIONS = 22  # Grafana Cloud and Checkly both offer 22 public probe locations
PROBE_INTERVAL_MIN = 4  # must be shorter than the idle window
PROBE_PRICE_PER_10K = 5.0  # $ per 10k runs: Grafana overage, Datadog annual (range $2.50–7.20)
PING_CPU_MS = 1  # the ping handler touches the loaded model but skips inference

# Minimum viable capacity (no inputs): enough for expected traffic at every
# moment. Autoscaling follows the daily curve, so the daily peak doesn't cost
# extra; what costs extra is the headroom held for growth that arrives before
# new capacity does (see SURGES and SCALE_OUT_MIN).
NODE_VCPU = 4
MIN_NODES = 2  # one per AZ
FLOOR_NODE_VCPU = 2  # the floor runs the smallest node that holds the model: c6i.large, half a c6i.xlarge
MAX_UTIL = 0.75  # above this, queueing starts to show in p99
EKS_ALLOCATABLE = 0.85  # kubelet, CNI, kube-proxy, log/metrics daemonsets
ECS_ALLOCATABLE = 0.95  # ECS agent only

# Fargate: billed per task for its vCPU and memory — no nodes to manage, no node
# overhead. Serving capacity runs 1 vCPU / 2 GB tasks (one inference at full speed
# per vCPU). The floor runs 2 × 0.5 vCPU / 1 GB tasks: CPU quota is enforced per
# 100 ms period, so a lone 20–40 ms request still finishes at full speed.
FARGATE_GB_PER_VCPU = 2
FARGATE_FLOOR_TASK = dict(vcpu=0.5, gb=1)
# EKS on Fargate adds 256 MB to every pod for kubelet / kube-proxy / containerd and
# rounds up to a valid size; pods request 256 MB less so they stay on the same
# billed size. A Fargate-only cluster also runs its system pods on Fargate:
# 2 × CoreDNS + 2 × AWS Load Balancer Controller, each at the smallest billed size.
EKS_FARGATE_SYSTEM_PODS = 4
EKS_FARGATE_SYSTEM_POD = dict(vcpu=0.25, gb=1)
REQUEST_KB, RESPONSE_KB = 1.0, 2.0

# Expected traffic surges: how fast load can double (minutes). Capacity must cover
# the growth that arrives before scale-out completes.
SURGES = {
    "Smooth: load doubles over 1 h": 60,
    "Surges: doubles within 10 min": 10,
    "Sharp: doubles within 3 min": 3,
    "Spikes: doubles within 30 s": 0.5,
}
SURGE_MIN, SURGE_MAX = 0.25, 120  # winner-map range (minutes, log scale)

# Ties go to the simpler platform: each step up in complexity (nodes to manage,
# Kubernetes to run) must be at least this much cheaper to win.
TIE_MARGIN = 0.05

# Surge begins → new capacity serving (image pull excluded; it's the same for
# both). ECS is modeled tuned (step scaling on a fast metric), because tuning is
# a free configuration choice; the default target-tracking path is in the appendix.
# EKS: HPA 15–30 s + Karpenter node Ready 45–70 s + pod start ≈ 1.5–2 min.
# ECS tuned: step-scaling alarm 20–90 s + capacity provider → ASG → EC2 boot and
# agent registration 2–4 min ≈ 3–4.5 min. ECS default: the target-tracking alarm
# alone takes ~6 min (AWS: 363 s), ≈ 7–10 min end to end.
# Fargate skips the node stage but starts a fresh single-use micro-VM per task/pod
# (~30–60 s, AWS Batch cites ~30 s) and registers it with the ALB (20–30 s).
# ECS on Fargate tuned: step-scaling alarm 60–120 s + launch + health check ≈ 2–3.5 min;
# default target tracking (AWS: 386 s to provision) ≈ 6.5–7.5 min. EKS on Fargate:
# metrics-server + HPA 15–45 s + pod start + ALB IP-target registration ≈ 1.5–2.5 min.
SCALE_OUT_MIN = dict(eks=1.75, ecs=3.5, ecs_default=8.0,
                     eks_fargate=2.0, ecs_fargate=2.5, ecs_fargate_default=7.0)

# Model size drives compute: a dense model answering one request touches every
# weight at least once. Averaged across model types on one native core; ranges
# from ~0.1 ms/MB (tabular nets, tree ensembles) to ~2 ms/MB (a transformer over a
# sequence, where each weight is reused per token).
MS_PER_MB = 1.0

# Latency constants.
ALB_MS = 5
TAIL_FACTOR = 1.5  # inference p99 / p50
WASM_SLOWDOWN = 2.0  # single-thread WASM SIMD vs single-thread native, averaged across model types

# Cloudflare isolate limits and cold-start estimates. Eviction timing is not
# published; since 2025 a low-traffic Worker is routed to one server per PoP
# ("shard and conquer"), so warm isolates are shared across that PoP's traffic.
CF_MEMORY_MB = 128
CF_USABLE_MODEL_MB = 90  # after JS heap + WASM runtime
CF_IDLE_WINDOW_S = 300  # an idle isolate is assumed warm for ~5 minutes
CF_ISOLATES_PER_POP = 1  # sharding: one server per PoP until traffic needs more
CF_COLD_BASE_MS = 50  # isolate start; the model ships in the 64 MiB script, no R2 fetch
CF_COLD_MS_PER_MB = 4  # parse weights + ONNX session init per MB of model
POP_ZIPF = 1.0  # traffic concentrates in big locations; the long tail goes cold first
P99_THRESHOLD = 0.01  # at ≥ 1% of requests, a slow path is the p99

SEGMENTS = {
    "App ↔ backend": "#c9c7bf",
    "Network": "#9ec5f4",
    "Handshake": "#6da7ec",
    "Inference": "#2a78d6",
    "Cold start": "#104281",
    "Overhead": "#a8a69e",
}

# --- Controls -----------------------------------------------------------------

st.subheader("Assumptions")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    region_name = st.selectbox("AWS region", list(REGIONS), index=list(REGIONS).index("ap-southeast-1 (Singapore)"))
with c2:
    caller_name = st.selectbox(
        "How the app calls the model", list(CALLERS), index=1,
        help="Directly: phones/browsers connect to the model, so handshakes count. Through a backend: "
             "the user reaches the backend, which calls the model over pooled connections.",
    )
    direct = CALLERS[caller_name]["direct"]
with c3:
    audience_name = st.selectbox("Where users are", list(AUDIENCES), index=1)
with c4:
    model_mb = st.number_input(
        "Model size (MB)", min_value=1, max_value=1000, value=20, step=5,
        help=f"Weights in memory. Compute ≈ {MS_PER_MB:g} ms per MB on one native core; "
             f"a Workers isolate holds ≈ {CF_USABLE_MODEL_MB} MB of its {CF_MEMORY_MB} MB.",
    )
with c5:
    surge_name = st.selectbox(
        "Expected traffic surges", list(SURGES), index=2,
        help="How fast load can grow. AWS must hold headroom for the growth that arrives before scale-out completes; Workers scales per request.",
    )
with c6:
    slo_ms = st.number_input("p99 latency SLO (ms)", min_value=50, max_value=1000, value=300, step=10)

region = REGIONS[region_name]
audience = AUDIENCES[audience_name]
doubling_min = SURGES[surge_name]
multi = "rtt_aws_multi" in audience
# Where an AWS-hosted backend runs (backend mode): next to its users.
backend_regions = MULTI_REGIONS if multi else [region_name]
backend_rtt = audience["rtt_aws_multi"] if multi else audience["rtt_aws"]
# Edge locations the Workers traffic spreads across: every PoP near users, whether
# the app calls the model Worker directly or a backend Worker at that PoP does.
cf_pops = audience["pops"]
warm_pops = min(cf_pops, PROBE_LOCATIONS)


def infer_ms(mb):
    return MS_PER_MB * mb


def cold_load_ms(mb):
    return CF_COLD_BASE_MS + CF_COLD_MS_PER_MB * mb


# --- Model --------------------------------------------------------------------


def headroom(doubling, scale_out_min):
    """Capacity multiple needed so a surge never outruns capacity while scale-out
    catches up: load grows linearly to 2× over `doubling` minutes."""
    return 1 + np.minimum(1.0, scale_out_min / np.asarray(doubling, dtype=float))


def _required_vcpu(requests, surge, scale_out_min):
    load_vcpu = requests / SECONDS_PER_MONTH * (infer_ms(model_mb) / 1000)
    return load_vcpu * headroom(surge, scale_out_min) / MAX_UTIL


def _ec2_compute(allocatable, scale_out_min):
    """$/hour of EC2 nodes in one region. Billed per vCPU needed above the floor:
    Karpenter and ECS capacity providers mix instance sizes, so at scale you pay
    close to what you use. Rounding up to whole 4-vCPU nodes would add step noise
    larger than the differences being compared."""
    def compute_hr(requests, surge, rg):
        floor = MIN_NODES * FLOOR_NODE_VCPU / NODE_VCPU  # in c6i.xlarge equivalents (same price per vCPU)
        nodes = np.maximum(floor, _required_vcpu(requests, surge, scale_out_min) / (NODE_VCPU * allocatable))
        return nodes * rg["node_hr"]
    return compute_hr


def _fargate_task_hr(rg, vcpu, gb):
    return vcpu * rg["fargate_vcpu_hr"] + gb * rg["fargate_gb_hr"]


def _fargate_compute(scale_out_min, system_pods=0):
    """$/hour of Fargate tasks in one region: serving capacity per vCPU needed,
    never below the floor tasks, plus any cluster system pods."""
    def compute_hr(requests, surge, rg):
        floor = MIN_NODES * _fargate_task_hr(rg, **FARGATE_FLOOR_TASK)
        serving = _required_vcpu(requests, surge, scale_out_min) * _fargate_task_hr(rg, 1, FARGATE_GB_PER_VCPU)
        system = system_pods * _fargate_task_hr(rg, **EKS_FARGATE_SYSTEM_POD)
        return np.maximum(floor, serving) + system
    return compute_hr


def cold_share(requests, pops, window_s, warm=0):
    """Share of requests arriving at an edge location that saw no other request
    within `window_s` (Poisson arrivals). Locations get a Zipf share of traffic;
    the `warm` biggest are kept warm and never go cold."""
    weights = 1 / np.arange(1, pops + 1) ** POP_ZIPF
    weights /= weights.sum()
    rps = np.asarray(requests, dtype=float)[..., None] / SECONDS_PER_MONTH
    cold = weights * np.exp(-rps * weights / CF_ISOLATES_PER_POP * window_s)
    cold[..., :warm] = 0.0
    return cold.sum(axis=-1)


def _aws_cost(regions, compute_hr, control_plane_hr, delivery):
    """Traffic splits evenly across the regions; each region pays its own floor.
    `delivery`: "internet" (EC2 egress), "cloudfront", or "internal" (to the backend)."""
    def cost(requests, surge):
        share = requests / len(regions)
        total = 0.0
        for r in regions:
            rg = REGIONS[r]
            total = total + (
                (compute_hr(share, surge, rg) + rg["alb_hr"] + control_plane_hr) * HOURS_PER_MONTH
                + share * (REQUEST_KB + RESPONSE_KB) / 1e6 * rg["lcu_hr"]
            )
            if delivery == "internet":
                total = total + share * RESPONSE_KB / 1e6 * rg["egress_gb"]
        if delivery == "cloudfront":
            total = total + (
                np.maximum(0, requests - CLOUDFRONT_FREE_REQ) / 1e6 * CLOUDFRONT_PER_M_REQ
                + np.maximum(0, requests * RESPONSE_KB / 1e6 - CLOUDFRONT_FREE_GB) * CLOUDFRONT_EGRESS_GB
                + requests * REQUEST_KB / 1e6 * CLOUDFRONT_TO_ORIGIN_GB
            )
        return total
    return cost


RUNS_PER_PROBE = HOURS_PER_MONTH * 60 / PROBE_INTERVAL_MIN  # per location per month


def _cf_cost(kept_warm):
    def cost(requests, surge):
        requests = requests + 0 * np.asarray(surge)  # scales per request: surges cost nothing extra
        pings = RUNS_PER_PROBE * warm_pops * kept_warm
        # Through a backend Worker, the model call is a service binding: no request fee of
        # its own (the app's backend request carries it), only the model's CPU time.
        model_requests = requests if direct else 0.0 * requests
        billable_req = np.maximum(0, model_requests + pings - CF_INCLUDED_REQ)
        cpu_ms = requests * infer_ms(model_mb) * WASM_SLOWDOWN + pings * PING_CPU_MS
        billable_cpu = np.maximum(0, cpu_ms - CF_INCLUDED_CPU_MS)
        probes = pings / 1e4 * PROBE_PRICE_PER_10K
        return CF_BASE + billable_req / 1e6 * CF_PER_M_REQ + billable_cpu / 1e6 * CF_PER_M_CPU_MS + probes
    return cost


def _parts(requests, surge, **segments):
    """p99 components, broadcast over the (requests, surge) grid. In backend mode the
    app ↔ backend leg depends on where the platform hosts the backend; it's included
    so the SLO stays end to end."""
    flat = np.ones(np.broadcast(np.asarray(requests), np.asarray(surge)).shape)
    return {seg: segments.get(seg, 0.0) * flat for seg in SEGMENTS}


# App ↔ backend in backend mode: HTTP/3 handshake at a nearby edge, then the round
# trip to wherever the backend runs — the AWS region(s), or the PoP itself.
APP_TO_AWS_BACKEND = HANDSHAKE_RTTS_QUIC * audience["rtt_cf"] + backend_rtt
APP_TO_CF_BACKEND = HANDSHAKE_RTTS_QUIC * audience["rtt_cf"] + audience["rtt_cf"]


def _latency_aws_alb(rtt):
    def latency(requests, surge):
        return _parts(requests, surge, Network=rtt, Handshake=HANDSHAKE_RTTS_TCP_TLS * rtt,
                      Inference=infer_ms(model_mb) * TAIL_FACTOR, Overhead=ALB_MS)
    return latency


def _latency_aws_cloudfront(rtt):
    def latency(requests, surge):
        origin_cold = cold_share(requests, audience["pops"], CLOUDFRONT_KEEPALIVE_S) >= P99_THRESHOLD
        handshake = HANDSHAKE_RTTS_QUIC * audience["rtt_cf"] + np.where(
            origin_cold, HANDSHAKE_RTTS_TCP_TLS * (rtt - audience["rtt_cf"]), 0.0,
        )
        return _parts(requests, surge, Network=rtt, Handshake=handshake,
                      Inference=infer_ms(model_mb) * TAIL_FACTOR, Overhead=ALB_MS)
    return latency


def _latency_aws_backend(requests, surge):
    return _parts(requests, surge, Network=BACKEND_TO_AWS_MS,
                  Inference=infer_ms(model_mb) * TAIL_FACTOR, Overhead=ALB_MS,
                  **{"App ↔ backend": APP_TO_AWS_BACKEND})


def _latency_cf(warm):
    def latency(requests, surge):
        cold = cold_share(requests, cf_pops, CF_IDLE_WINDOW_S, warm) >= P99_THRESHOLD
        network = dict(Network=audience["rtt_cf"], Handshake=HANDSHAKE_RTTS_QUIC * audience["rtt_cf"]) \
            if direct else {"Network": BACKEND_TO_CF_MS, "App ↔ backend": APP_TO_CF_BACKEND}
        return _parts(requests, surge, **network,
                      Inference=infer_ms(model_mb) * WASM_SLOWDOWN * TAIL_FACTOR,
                      **{"Cold start": np.where(cold, cold_load_ms(model_mb), 0.0)})
    return latency


def _p99(parts):
    return sum(parts.values())


def _aws_setups(compute_hr, control_plane_hr):
    if not direct:
        return [dict(
            label="next to the backend",
            cost=_aws_cost(backend_regions, compute_hr, control_plane_hr, "internal"),
            latency=_latency_aws_backend,
        )]
    footprints = [("1 region", [region_name], audience["rtt_aws"])]
    if multi:
        footprints.append(("3 regions", MULTI_REGIONS, audience["rtt_aws_multi"]))
    setups = []
    for label, regions, rtt in footprints:
        setups.append(dict(
            label=label + ", ALB only",
            cost=_aws_cost(regions, compute_hr, control_plane_hr, "internet"),
            latency=_latency_aws_alb(rtt),
        ))
        setups.append(dict(
            label=label + " + CloudFront",
            cost=_aws_cost(regions, compute_hr, control_plane_hr, "cloudfront"),
            latency=_latency_aws_cloudfront(rtt),
        ))
    return setups


def _fits_everywhere(surge):
    return np.ones(np.shape(surge), dtype=bool)


def _fits_cf(surge):
    return np.full(np.shape(surge), model_mb <= CF_USABLE_MODEL_MB)


# Every option picks its cheapest setup that meets the SLO — footprint, edge and
# keep-warm are choices, and each carries its full cost.
# Color = orchestrator (validated palette, first three slots — all-pairs safe).
# Running on EC2 nodes is the second encoding (dashed line, dotted map texture):
# it's the structural difference from serverless (Workers, Fargate). No five hues
# stay distinguishable when any two map regions can touch.
# `complexity` counts steps up from the simplest: nodes to manage, Kubernetes to run.
FAMILIES = {"CF Workers": "#2a78d6", "EKS": "#eb6834", "ECS": "#1baf7a"}
PLATFORMS = [
    dict(name="CF Workers", family="CF Workers", fargate=False, complexity=0, fits=_fits_cf, setups=[
        dict(label="on demand", cost=_cf_cost(False), latency=_latency_cf(0)),
        dict(label="kept warm", cost=_cf_cost(True), latency=_latency_cf(warm_pops)),
    ]),
    dict(name="EKS on EC2", family="EKS", fargate=False, complexity=2, fits=_fits_everywhere,
         setups=_aws_setups(_ec2_compute(EKS_ALLOCATABLE, SCALE_OUT_MIN["eks"]), EKS_CONTROL_PLANE_HR)),
    dict(name="ECS on EC2", family="ECS", fargate=False, complexity=1, fits=_fits_everywhere,
         setups=_aws_setups(_ec2_compute(ECS_ALLOCATABLE, SCALE_OUT_MIN["ecs"]), 0.0)),
    dict(name="EKS on Fargate", family="EKS", fargate=True, complexity=1, fits=_fits_everywhere,
         setups=_aws_setups(_fargate_compute(SCALE_OUT_MIN["eks_fargate"], EKS_FARGATE_SYSTEM_PODS),
                            EKS_CONTROL_PLANE_HR)),
    dict(name="ECS on Fargate", family="ECS", fargate=True, complexity=0, fits=_fits_everywhere,
         setups=_aws_setups(_fargate_compute(SCALE_OUT_MIN["ecs_fargate"]), 0.0)),
]
for _p in PLATFORMS:
    _p["color"] = FAMILIES[_p["family"]]
    _p["ec2"] = _p["family"] != "CF Workers" and not _p["fargate"]
NONE_COLOR = "#c9c7bf"
NONE_LABEL = "none meets SLO"


def evaluate(p, requests, surge):
    """Cost, p99, feasibility and chosen setup index for one option. Picks the
    cheapest setup that meets the SLO; if none does, shows the lowest-latency one."""
    costs = np.array([s["cost"](requests, surge) for s in p["setups"]])
    p99s = np.array([_p99(s["latency"](requests, surge)) for s in p["setups"]])
    costs, p99s = np.broadcast_arrays(costs, p99s)
    ok = p["fits"](surge) & (p99s <= slo_ms)
    any_ok = ok.any(axis=0)
    idx = np.where(any_ok, np.where(ok, costs, np.inf).argmin(axis=0), p99s.argmin(axis=0))

    def pick(a):
        return np.take_along_axis(a, idx[None], axis=0)[0]

    return pick(costs), pick(p99s), any_ok, idx


def rank_costs(costs, oks):
    """Costs used to pick a winner: infeasible = ∞, and each step of complexity
    carries the tie margin, so within 5% the simpler option wins."""
    penalty = np.array([(1 / (1 - TIE_MARGIN)) ** p["complexity"] for p in PLATFORMS])
    penalty = penalty.reshape((-1,) + (1,) * (costs.ndim - 1))
    return np.where(oks, costs * penalty, np.inf)


def _fmt(v):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.1f}{suf}".replace(".0" + suf, suf)
    return f"{v:,.0f}"


def _fmt_min(minutes):
    return f"{minutes * 60:.0f} s" if minutes < 1 else (f"{minutes:g} min" if minutes < 60 else f"{minutes / 60:g} h")


SURGE_TICKS = [0.25, 0.5, 1, 2, 5, 10, 30, 60, 120]


def _name(key):
    """Winner keys encode platform * 10 + setup index; -1 = nothing meets the SLO."""
    if key < 0:
        return NONE_LABEL
    p = PLATFORMS[key // 10]
    setup = p["setups"][key % 10]["label"]
    return p["name"] + (f" ({setup})" if len(p["setups"]) > 1 else "")


CHART_HEIGHT = 400
CHART_FONT = dict(size=13)
REQ_MIN, REQ_MAX = 1e2, 1e10  # 100 req/mo ≈ 3 a day: effectively idle; exact zero is in the headline
LOG_RANGE = [np.log10(REQ_MIN), np.log10(REQ_MAX)]

st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--text-color) !important;
        opacity: 0.92 !important;
        font-size: 0.9rem;
    }
    .st-key-chart_columns [data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:not(:first-child) {
        border-left: 1px solid rgba(128, 128, 128, 0.4);
        padding-left: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sweep request volume once; the headline and the first two charts read from it.
req_grid = np.logspace(*LOG_RANGE, 400)
sweep = [evaluate(p, req_grid, doubling_min) for p in PLATFORMS]
sweep_cost = np.array([r[0] for r in sweep])
sweep_p99 = np.array([r[1] for r in sweep])
sweep_ok = np.array([r[2] for r in sweep])
sweep_setup = np.array([r[3] for r in sweep])
best = rank_costs(sweep_cost, sweep_ok).argmin(axis=0)
winner = np.where(
    sweep_ok.any(axis=0), best * 10 + np.take_along_axis(sweep_setup, best[None], axis=0)[0], -1,
)


# Options within the tie margin of the cheapest raw cost are "roughly equal": the
# winner is picked by the simplicity rule, and the headline names the others too.
raw_costs = np.where(sweep_ok, sweep_cost, np.inf)
close = raw_costs <= raw_costs.min(axis=0) * (1 + TIE_MARGIN)


def _label_at(i):
    if winner[i] < 0:
        return NONE_LABEL
    others = [PLATFORMS[k]["name"] for k in np.argsort(raw_costs[:, i]) if close[k, i] and k != winner[i] // 10]
    return " ≈ ".join([_name(winner[i])] + others)


def _winner_zones():
    """Runs of the same headline label (winner plus anything within the tie margin).
    Runs narrower than ~1.4× in volume fold into the zone before them, so the reader
    sees the crossovers, not a flicker."""
    labels = [_label_at(i) for i in range(len(req_grid))]
    zones = []
    for i, label in enumerate(labels):
        if zones and zones[-1]["label"] == label:
            zones[-1]["e"] = i + 1
        else:
            zones.append(dict(label=label, s=i, e=i + 1))
    merged = []
    for z in zones:
        short = np.log10(req_grid[z["e"] - 1] / req_grid[z["s"]]) < 0.15
        if merged and (short or z["label"] == merged[-1]["label"]):
            merged[-1]["e"] = z["e"]
        else:
            merged.append(z)
    for z in merged:
        z["tie"] = " ≈ " in z["label"]
    return merged


zones = _winner_zones()
st.subheader(f"Cheapest option that meets the {slo_ms} ms p99 SLO")
spans = [f"**{z['label']}** {_fmt(req_grid[z['s']])}–{_fmt(req_grid[z['e'] - 1])} req/mo" for z in zones]
(st.success if (winner >= 0).any() else st.error)(" → ".join(spans))
st.caption(
    f"≈ = within {TIE_MARGIN:.0%} of the cheapest; the first name wins the tie as the simpler "
    "option (no nodes beats nodes, ECS beats EKS, each step needing a 5% edge)."
)

# Exact zero traffic: what each option costs just to exist (a log axis can't show 0).
idle = []
for p in PLATFORMS:
    cost, _, ok, setup = evaluate(p, np.array([0.0]), doubling_min)
    idle.append(f"{p['name']} \\${cost[0]:,.0f}" + (f" ({p['setups'][setup[0]]['label']})" if len(p["setups"]) > 1 else ""))
st.caption(
    "**At zero traffic:** " + " · ".join(idle) + " per month. Only Workers scales to zero and "
    "still meets the SLO; EKS and ECS can scale to zero, but the first request after idle waits "
    "for an EC2 instance to boot (minutes) or a Fargate task to start (tens of seconds), so "
    "their floor is the smallest always-on setup."
)

chart_columns = st.container(key="chart_columns")
left, middle, right = chart_columns.columns(3)


def _setup_labels(p, setup):
    return np.array([s["label"] for s in p["setups"]])[setup]


def _feasibility_traces(fig, x, y, ok, p, setup, hovertemplate):
    """Solid (serverless: Workers, Fargate) or dashed (EC2 nodes) where the option
    meets the SLO, dotted where it doesn't. Hover names the setup."""
    labels = _setup_labels(p, setup)
    fig.add_trace(go.Scatter(
        x=x, y=np.where(ok, y, np.nan), mode="lines", name=p["name"], legendgroup=p["name"],
        line=dict(color=p["color"], width=2, dash="dash" if p["ec2"] else "solid"),
        customdata=labels, hovertemplate=hovertemplate,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.where(ok, np.nan, y), mode="lines", name=p["name"] + " (misses SLO)",
        legendgroup=p["name"], showlegend=bool(not ok.all()),
        line=dict(color=p["color"], width=2, dash="dot"), customdata=labels, hovertemplate=hovertemplate,
    ))


def _mark_winner_changes(fig):
    for z in zones:
        if z["tie"]:
            fig.add_vrect(
                x0=np.log10(req_grid[z["s"]]), x1=np.log10(req_grid[z["e"] - 1]),
                fillcolor="gray", opacity=0.15, line_width=0,
            )
        elif z["s"] > 0:
            fig.add_vline(x=np.log10(req_grid[z["s"]]), line=dict(color="gray", dash="dot", width=1))


with left:
    st.markdown(f"**Monthly cost vs request volume** ({model_mb} MB model ≈ {infer_ms(model_mb):g} ms)")
    fig_cost = go.Figure()
    for p, y, ok, setup in zip(PLATFORMS, sweep_cost, sweep_ok, sweep_setup):
        _feasibility_traces(fig_cost, req_grid, y, ok, p, setup, "%{y:$,.0f} · %{customdata}")
    _mark_winner_changes(fig_cost)
    fig_cost.update_layout(
        xaxis=dict(title="Requests per month", type="log", range=LOG_RANGE),
        yaxis=dict(title="Cost per month ($)", type="log", dtick=1, tickformat="$~s"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.3),
        margin=dict(t=10, l=10, r=10, b=40),
        height=CHART_HEIGHT,
        font=CHART_FONT,
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption(
        "Color = orchestrator; dashed = on EC2 nodes, solid = serverless. AWS has a fixed floor per region (2 small "
        "nodes or tasks, ALB, EKS control plane), then pays per vCPU for average load plus the "
        "surge headroom its scale-out speed requires; Workers is pay-per-request from $5. Each line is that option's cheapest setup meeting the SLO "
        "(hover for which); the app's own backend compute isn't counted. "
        "Dotted = misses the SLO. Vertical lines mark where the winner changes; shading "
        "marks a zone where two options stay within 5% of each other."
    )

with middle:
    st.markdown(f"**p99 latency vs request volume** (SLO {slo_ms} ms, end to end)")
    fig_p99 = go.Figure()
    # Every AWS option (EKS/ECS, EC2/Fargate) has the same latency profile and makes
    # the same setup choice, so one line stands for all four.
    for k, name, color in ((0, PLATFORMS[0]["name"], PLATFORMS[0]["color"]),
                           (1, "EKS / ECS, EC2 or Fargate", "#8a8880")):
        fig_p99.add_trace(go.Scatter(
            x=req_grid, y=sweep_p99[k], mode="lines", name=name,
            line=dict(color=color, width=2),
            customdata=_setup_labels(PLATFORMS[k], sweep_setup[k]),
            hovertemplate="%{y:,.0f} ms · %{customdata}",
        ))
        # Label each setup switch: a step in the line is a change of setup (cost-driven),
        # not traffic changing a setup's latency.
        labels = _setup_labels(PLATFORMS[k], sweep_setup[k])
        for i in np.nonzero(labels[1:] != labels[:-1])[0] + 1:
            fig_p99.add_annotation(
                x=np.log10(req_grid[i]), y=sweep_p99[k][i], text=f"→ {labels[i]}",
                showarrow=True, arrowhead=2, arrowsize=0.8, ax=50, ay=40 if k else -30,
                font=dict(size=11, color="black"), bgcolor="rgba(255,255,255,0.85)",
            )
    fig_p99.add_hline(
        y=slo_ms, line=dict(color="gray", dash="dash", width=1.5),
        annotation_text=f"SLO {slo_ms} ms", annotation_position="top left",
    )
    _mark_winner_changes(fig_p99)
    fig_p99.update_layout(
        xaxis=dict(title="Requests per month", type="log", range=LOG_RANGE),
        yaxis=dict(title="p99 latency (ms)", rangemode="tozero"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.3),
        margin=dict(t=10, l=10, r=10, b=40),
        height=CHART_HEIGHT,
        font=CHART_FONT,
    )
    st.plotly_chart(fig_p99, use_container_width=True)

    def _edge(share):
        below = share < P99_THRESHOLD
        return _fmt(req_grid[np.argmax(below)]) if below.any() else f"> {_fmt(REQ_MAX)}"

    cf_edge = _edge(cold_share(req_grid, cf_pops, CF_IDLE_WINDOW_S))
    notes = [
        f"Workers pays a {cold_load_ms(model_mb):,.0f} ms model load inside p99 until fewer than 1% "
        f"of requests hit a cold isolate (≈ {cf_edge} req/mo across {cf_pops} PoP{'s' if cf_pops > 1 else ''})."
    ]
    if direct:
        cloudfront_edge = _edge(cold_share(req_grid, audience["pops"], CLOUDFRONT_KEEPALIVE_S))
        notes.append(
            f"CloudFront pays a backbone TCP + TLS handshake to the ALB until its origin connections "
            f"stay open (≈ {cloudfront_edge} req/mo). A plain ALB pays the full handshake to the region every time."
        )
    else:
        notes.append(
            f"Each platform hosts the app's backend: the user ↔ backend leg is {APP_TO_AWS_BACKEND} ms "
            f"to an AWS region, {APP_TO_CF_BACKEND} ms to a backend Worker at the nearest PoP, "
            "which calls the model Worker through a service binding."
        )
    notes.append(
        "Each line is the cheapest setup that meets the SLO, so a step up marks a switch to a "
        "cheaper setup that is slower but still within the SLO (e.g. dropping CloudFront once its "
        "free tier runs out), not traffic slowing a setup down. Capacity keeps utilization ≤ 75%, "
        "so no setup gets slower with load."
    )
    st.caption(" ".join(notes))

with right:
    st.markdown("**Cheapest option that meets the SLO** (winner map)")
    nx, ny = 120, 100
    log_x = np.linspace(*LOG_RANGE, nx)
    log_y = np.linspace(np.log10(SURGE_MIN), np.log10(SURGE_MAX), ny)
    LX, LY = np.meshgrid(log_x, log_y)
    REQ, SURGE = 10 ** LX, 10 ** LY

    grid = [evaluate(p, REQ, SURGE) for p in PLATFORMS]
    grid_costs = np.array([np.where(ok, c, np.inf) for c, _, ok, _ in grid])
    grid_setup = np.array([idx for *_, idx in grid])
    ok_any = np.isfinite(grid_costs).any(axis=0)
    WIN = np.where(ok_any, rank_costs(grid_costs, np.isfinite(grid_costs)).argmin(axis=0), -1)

    # Discrete colorscale by orchestrator: -1 = none feasible, 0..2 = family index.
    n = len(PLATFORMS)
    family_names = list(FAMILIES)
    family_of = np.array([family_names.index(p["family"]) for p in PLATFORMS])
    FAM = np.where(WIN >= 0, family_of[np.maximum(WIN, 0)], -1)
    colors = [NONE_COLOR] + list(FAMILIES.values())
    scale = []
    for i, col in enumerate(colors):
        scale += [[i / len(colors), col], [(i + 1) / len(colors), col]]

    hover = np.empty(WIN.shape, dtype=object)
    for j in range(ny):
        for i in range(nx):
            rows = [
                f"{_name(k * 10 + grid_setup[k, j, i])}: "
                + (f"${grid_costs[k, j, i]:,.0f}" if np.isfinite(grid_costs[k, j, i]) else "misses SLO")
                for k in range(n)
            ]
            hover[j, i] = f"{_fmt(REQ[j, i])} req/mo · load doubles in {_fmt_min(SURGE[j, i])}<br>" + "<br>".join(rows)

    fig_map = go.Figure()
    fig_map.add_trace(go.Heatmap(
        x=log_x, y=log_y, z=FAM, zmin=-1.5, zmax=len(FAMILIES) - 0.5,
        colorscale=scale, showscale=False,
        text=hover, hovertemplate="%{text}<extra></extra>",
    ))
    # Texture: a dot grid over cells where an option on EC2 nodes wins.
    on_ec2 = np.isin(WIN, [k for k, p in enumerate(PLATFORMS) if p["ec2"]])
    dots = np.zeros_like(on_ec2)
    dots[::3, ::3] = True
    jj, ii = np.nonzero(on_ec2 & dots)
    fig_map.add_trace(go.Scatter(
        x=log_x[ii], y=log_y[jj], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=3, color="rgba(255,255,255,0.8)"),
    ))
    # Legend entries for what's present — small regions are too small to label directly.
    for f, name in enumerate([NONE_LABEL] + family_names, start=-1):
        if (FAM == f).any():
            fig_map.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker=dict(symbol="square", size=12, color=colors[f + 1]),
            ))
    if on_ec2.any():
        fig_map.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="dotted = on EC2 nodes",
            marker=dict(symbol="square-dot", size=12, color="#8a8880"),
        ))
    # The two charts on the left are this horizontal slice.
    fig_map.add_hline(
        y=np.log10(doubling_min), line=dict(color="rgba(255,255,255,0.7)", dash="dot", width=1),
        annotation_text=f"doubles in {_fmt_min(doubling_min)} (slice)", annotation_position="bottom left",
        annotation_font=dict(color="black", size=11), annotation_bgcolor="rgba(255,255,255,0.8)",
    )
    tick_exps = range(int(LOG_RANGE[0]), int(LOG_RANGE[1]) + 1)
    fig_map.update_layout(
        xaxis=dict(
            title="Requests per month",
            tickvals=list(tick_exps), ticktext=[_fmt(10 ** e) for e in tick_exps],
            range=[log_x[0], log_x[-1]],
        ),
        yaxis=dict(
            title="Load can double within",
            tickvals=np.log10(SURGE_TICKS), ticktext=[_fmt_min(t) for t in SURGE_TICKS],
            range=[log_y[0], log_y[-1]],
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.38),
        margin=dict(t=10, l=10, r=10, b=40),
        height=CHART_HEIGHT,
        font=CHART_FONT,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    if model_mb > CF_USABLE_MODEL_MB:
        st.info(
            f"A {model_mb} MB model doesn't fit a Workers isolate (≈ {CF_USABLE_MODEL_MB} MB of "
            f"{CF_MEMORY_MB} MB), so CF Workers is excluded."
        )
    st.caption(
        "Both uncertain axes together. Color = the orchestrator of the cheapest option that "
        "meets the SLO, dots = it runs on EC2 nodes (plain = serverless); grey = nothing does. Faster surges make AWS "
        "hold more idle headroom, so faster scalers (EKS on EC2 ~1.75 min, EKS on Fargate ~2 min, ECS on Fargate ~2.5 min) gain "
        "on ECS on EC2 (~3.5 min). Ties go to the simpler option: each step up (nodes to manage, "
        f"Kubernetes to run) must be at least {TIE_MARGIN:.0%} cheaper. The dotted line is the "
        "current surge profile."
    )

# --- Appendix -----------------------------------------------------------------

st.divider()
st.header("Appendix: fixed assumptions")
a_left, a_right = st.columns([1, 1])

with a_left:
    st.markdown(
        f"**p99 latency budget** ({model_mb} MB model, {audience_name.lower()}, "
        f"{'called directly' if direct else 'called from the backend'})"
    )
    cf_warm = _latency_cf(cf_pops)(REQ_MAX, doubling_min)
    rows = {
        "CF Workers (warm)": cf_warm,
        "CF Workers (cold isolate)": {**cf_warm, "Cold start": cold_load_ms(model_mb)},
    }
    for setup in PLATFORMS[-1]["setups"]:
        rows[f"AWS ({setup['label']})"] = setup["latency"](REQ_MAX, doubling_min)
        if "CloudFront" in setup["label"]:
            rows[f"AWS ({setup['label']}, cold origin)"] = setup["latency"](REQ_MIN, doubling_min)
    fig_lat = go.Figure()
    for seg, color in SEGMENTS.items():
        values = [float(np.asarray(r[seg]).item()) for r in rows.values()]
        if not any(values):
            continue
        fig_lat.add_trace(go.Bar(
            y=list(rows), x=values, name=seg, orientation="h",
            marker=dict(color=color, line=dict(color="white", width=2)),
            hovertemplate="%{y} · " + seg + ": %{x:,.0f} ms<extra></extra>",
        ))
    totals = [float(np.asarray(_p99(r)).item()) for r in rows.values()]
    for name, t in zip(rows, totals):
        fig_lat.add_annotation(
            x=t, y=name, text=f"{t:,.0f} ms {'✓' if t <= slo_ms else '✗'}",
            showarrow=False, xanchor="left", xshift=6, font=dict(size=12),
        )
    fig_lat.add_vline(
        x=slo_ms, line=dict(color="gray", dash="dash", width=1.5),
        annotation_text=f"SLO {slo_ms} ms", annotation_position="top",
    )
    fig_lat.update_layout(
        barmode="stack",
        xaxis=dict(title="p99 latency (ms)", range=[0, max(max(totals), slo_ms) * 1.25]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.2),
        margin=dict(t=30, l=10, r=10, b=40),
        height=160 + 45 * len(rows),
        font=CHART_FONT,
    )
    st.plotly_chart(fig_lat, use_container_width=True)
    st.caption(
        f"Workers wins on network (nearest PoP) and loses on compute (WASM {WASM_SLOWDOWN:g}× "
        f"slower). Cold load = {CF_COLD_BASE_MS} ms + {CF_COLD_MS_PER_MB} ms/MB × {model_mb} MB. "
        "Percentiles are added per component, which overstates the true p99 of the total by "
        "roughly 10–20% — conservative for every option."
    )

    st.markdown("**Network**")
    st.dataframe(pd.DataFrame(
        [
            {"Audience": name, "p99 RTT → 1 AWS region (ms)": a["rtt_aws"],
             "→ nearest of 3 regions (ms)": a.get("rtt_aws_multi", "—"),
             "→ nearest edge (ms)": a["rtt_cf"], "Edge locations used": a["pops"]}
            for name, a in AUDIENCES.items()
        ]
    ), hide_index=True, use_container_width=True)
    st.caption(
        "RTTs are p99 for one request on an open connection, including last-mile variance. "
        "Called directly from the app, p99 also pays a handshake: +2 RTT for TCP + TLS 1.3 to "
        "an ALB (no HTTP/3), +1 RTT for HTTP/3 at Cloudflare's or CloudFront's edge. Through a "
        "backend, the backend runs on the same platform as the model and the user reaches it via "
        "an HTTP/3 edge: on AWS the backend is in the region and calls the model in-region "
        f"({BACKEND_TO_AWS_MS} ms, pooled connections); on Workers it's a Worker at the nearest PoP "
        "calling the model Worker through a service binding (no network hop)."
    )

    st.markdown("**Model size → compute**")
    st.dataframe(pd.DataFrame([
        {"Model type": "Tree ensemble, tabular neural net", "ms per MB (1 native core)": "~0.1",
         "WASM slowdown": "~1.2–1.5× (trees), ~2–3× (nets)"},
        {"Model type": "CNN", "ms per MB (1 native core)": "~0.7–1", "WASM slowdown": "~2.5–4×"},
        {"Model type": "Transformer over a sequence", "ms per MB (1 native core)": "~1.5–2.5",
         "WASM slowdown": "~2.5–4×"},
        {"Model type": "Model (averaged)", "ms per MB (1 native core)": f"{MS_PER_MB:g}",
         "WASM slowdown": f"{WASM_SLOWDOWN:g}×"},
    ]), hide_index=True, use_container_width=True)
    st.caption(
        "Answering one request touches every weight at least once, so compute grows with size; "
        "models that reuse weights per token or per pixel (transformers, CNNs) cost more per MB. "
        f"Inference tail p99/p50 = {TAIL_FACTOR:g}×; ALB overhead {ALB_MS} ms. WASM is single-thread "
        "SIMD vs single-thread native; native AVX-512 pulls ahead most on matrix-heavy models."
    )

with a_right:
    st.markdown("**Setups each option can choose** (the cheapest one meeting the SLO wins)")
    probe_cost = RUNS_PER_PROBE * warm_pops / 1e4 * PROBE_PRICE_PER_10K
    st.dataframe(pd.DataFrame([
        {"Setup": "EKS / ECS, 1 region (direct)",
         "What it takes": f"{region_name}: floor of 2 small nodes or tasks + ALB (+ EKS control plane)"},
        {"Setup": "EKS / ECS + CloudFront (direct)",
         "What it takes": f"HTTP/3 handshake at a nearby edge, keep-alive (≤ {CLOUDFRONT_KEEPALIVE_S} s) "
                          f"to the ALB; ${CLOUDFRONT_PER_M_REQ:.2f}/M requests + ${CLOUDFRONT_EGRESS_GB}/GB "
                          "after 10M requests / 1 TB free, instead of EC2 egress"},
        {"Setup": "EKS / ECS, 3 regions (direct, global)",
         "What it takes": "us-east-1 + eu-west-1 + ap-southeast-1, each with its own floor; traffic "
                          "split evenly; latency-based DNS (~$0.60/M queries, negligible)"},
        {"Setup": "EKS / ECS next to the backend (backend mode)",
         "What it takes": "The backend runs on AWS too: one deployment per backend region (1, or 3 for a global "
                          "audience), internal ALB, no internet egress"},
        {"Setup": "CF Workers behind a backend Worker (backend mode)",
         "What it takes": "The backend runs on Workers too: a Worker at the nearest PoP calls the model Worker "
                          "through a service binding (no network hop, no extra request charge)"},
        {"Setup": "CF Workers, on demand",
         "What it takes": "Isolates load the model on the first request after going idle"},
        {"Setup": "CF Workers, kept warm",
         "What it takes": f"{warm_pops} probe location{'s' if warm_pops > 1 else ''} × a ping every "
                          f"{PROBE_INTERVAL_MIN} min = {_fmt(RUNS_PER_PROBE * warm_pops)} runs/mo ≈ "
                          f"${probe_cost:,.0f}/mo at ${PROBE_PRICE_PER_10K:g}/10k; warms the {warm_pops} "
                          f"biggest of {cf_pops} PoP{'s' if cf_pops > 1 else ''}"},
    ]), hide_index=True, use_container_width=True)
    st.caption(
        f"Probe vendors offer ~{PROBE_LOCATIONS} public locations (Grafana Cloud, Checkly), far "
        "fewer than Cloudflare's 300+ PoPs, so the long tail stays on demand. A probe in a city "
        "usually lands on a nearby PoP (anycast follows BGP, not geography) — not guaranteed."
    )

    st.markdown("**AWS pricing by region** (on-demand, verify before quoting)")
    st.dataframe(pd.DataFrame(
        [
            {"Region": name, "c6i.xlarge $/hr": r["node_hr"],
             "Fargate $/vCPU-hr": r["fargate_vcpu_hr"], "Fargate $/GB-hr": r["fargate_gb_hr"],
             "ALB $/hr": r["alb_hr"], "LCU $/hr": r["lcu_hr"], "Egress $/GB": r["egress_gb"]}
            for name, r in REGIONS.items()
        ]
    ), hide_index=True, use_container_width=True)
    st.caption(f"EKS control plane ${EKS_CONTROL_PLANE_HR:.2f}/hr in every region (standard support).")

    st.markdown("**Cloudflare Workers** (Paid, Standard — one global price)")
    st.dataframe(pd.DataFrame([
        {"Item": "Base", "Value": f"${CF_BASE:g}/month"},
        {"Item": "Requests", "Value": f"{_fmt(CF_INCLUDED_REQ)} included, then ${CF_PER_M_REQ:.2f}/M"},
        {"Item": "CPU time", "Value": f"{_fmt(CF_INCLUDED_CPU_MS)} ms included, then ${CF_PER_M_CPU_MS:.2f}/M ms"},
        {"Item": "Egress", "Value": "free"},
        {"Item": "Isolate memory", "Value": f"{CF_MEMORY_MB} MB (≈ {CF_USABLE_MODEL_MB} MB usable for the model)"},
        {"Item": "Idle isolate stays warm (estimate)", "Value": f"≈ {CF_IDLE_WINDOW_S} s"},
        {"Item": "Warm isolates per PoP (estimate)", "Value": f"≈ {CF_ISOLATES_PER_POP} (low-traffic sharding)"},
        {"Item": "Script size", "Value": "64 MiB uncompressed; global-scope startup ≤ 1 s"},
        {"Item": "Cold model load (estimate)", "Value": f"{CF_COLD_BASE_MS} ms + {CF_COLD_MS_PER_MB} ms/MB"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("**Capacity** (enough for expected traffic at every moment)")
    st.dataframe(pd.DataFrame([
        {"Item": "Node pricing", "Value": f"c6i family, billed per vCPU (c6i.xlarge = {NODE_VCPU} vCPU)"},
        {"Item": "Minimum capacity", "Value": f"{MIN_NODES} × c6i.large ({FLOOR_NODE_VCPU} vCPU) per region, one per AZ; above that, billed per vCPU needed (mixed instance sizes)"},
        {"Item": "Max CPU utilization", "Value": f"{MAX_UTIL:.0%} (above this, queueing shows in p99)"},
        {"Item": "Allocatable CPU (EC2)", "Value": f"EKS {EKS_ALLOCATABLE:.0%} (daemonsets) · ECS {ECS_ALLOCATABLE:.0%}"},
        {"Item": "Fargate serving tasks", "Value": f"1 vCPU / {FARGATE_GB_PER_VCPU} GB each, billed per vCPU needed; no node overhead"},
        {"Item": "Fargate floor", "Value": f"{MIN_NODES} × {FARGATE_FLOOR_TASK['vcpu']:g} vCPU / {FARGATE_FLOOR_TASK['gb']:g} GB tasks per region, one per AZ"},
        {"Item": "EKS on Fargate extras", "Value": f"pods request 256 MB less to stay on the same billed size; {EKS_FARGATE_SYSTEM_PODS} system pods (CoreDNS, LB controller) on Fargate at {EKS_FARGATE_SYSTEM_POD['vcpu']:g} vCPU / {EKS_FARGATE_SYSTEM_POD['gb']:g} GB"},
        {"Item": "Request / response size", "Value": f"{REQUEST_KB:g} KB / {RESPONSE_KB:g} KB"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("**Surge headroom** (capacity held above current load)")
    st.dataframe(pd.DataFrame([
        {"Surge profile": name,
         f"EKS on EC2 ({SCALE_OUT_MIN['eks']:g} min)": f"{headroom(t, SCALE_OUT_MIN['eks']):.2f}×",
         f"EKS on Fargate ({SCALE_OUT_MIN['eks_fargate']:g} min)": f"{headroom(t, SCALE_OUT_MIN['eks_fargate']):.2f}×",
         f"ECS on Fargate ({SCALE_OUT_MIN['ecs_fargate']:g} min)": f"{headroom(t, SCALE_OUT_MIN['ecs_fargate']):.2f}×",
         f"ECS on EC2 ({SCALE_OUT_MIN['ecs']:g} min)": f"{headroom(t, SCALE_OUT_MIN['ecs']):.2f}×",
         f"ECS default ({SCALE_OUT_MIN['ecs_fargate_default']:g}–{SCALE_OUT_MIN['ecs_default']:g} min)": f"{headroom(t, SCALE_OUT_MIN['ecs_default']):.2f}×"}
        for name, t in SURGES.items()
    ]), hide_index=True, use_container_width=True)
    st.caption(
        "Autoscaling follows the daily curve, so the daily peak costs nothing extra. Headroom = "
        "the growth a surge adds before new capacity is serving (image pull excluded — same "
        "for both). EKS: HPA every 15 s, Karpenter node Ready in 45–70 s. ECS default: target "
        "tracking needs 3 of 3 one-minute datapoints (~6 min, per AWS), then the capacity "
        "provider scales the Auto Scaling group and EC2 boots. ECS tuned: step scaling on a "
        "fast metric (1 of 1 datapoints), but the capacity-provider stage still costs 2–4 min. "
        "Fargate skips the node stage: ECS on Fargate is the alarm plus a task start and health check (~2–3.5 min); "
        "EKS on Fargate is the HPA plus a pod on a fresh Fargate micro-VM (~1.5–2.5 min). "
        "The model uses tuned ECS. Workers needs no headroom: it scales per request."
    )

    st.markdown(f"**Cold-path sensitivity** ({cf_pops} Workers PoP{'s' if cf_pops > 1 else ''})")

    def _cold_edge(w, warm):
        share = cold_share(req_grid, cf_pops, w, warm)
        below = share < P99_THRESHOLD
        return f"{_fmt(req_grid[np.argmax(below)])} req/mo" if below.any() else f"> {_fmt(REQ_MAX)}"

    st.dataframe(pd.DataFrame([
        {"Idle isolate stays warm": f"{w // 60:g} min" + (" (model)" if w == CF_IDLE_WINDOW_S else ""),
         "Cold share < 1% above (on demand)": _cold_edge(w, 0),
         f"… (kept warm, {warm_pops} PoP{'s' if warm_pops > 1 else ''})": _cold_edge(w, warm_pops)}
        for w in (60, CF_IDLE_WINDOW_S, 1800)
    ]), hide_index=True, use_container_width=True)
    st.caption(
        "Cloudflare doesn't publish the idle window, and an isolate holding a large model is "
        "a likely early eviction under memory pressure. Traffic per location follows a Zipf "
        "distribution: big locations stay warm, the long tail goes cold first. CloudFront's "
        f"origin connections follow the same math with a {CLOUDFRONT_KEEPALIVE_S} s keep-alive."
    )

st.caption(
    "Not modeled: the app's own backend compute (it moves with the platform; its cost depends on the app), ops labor, NAT gateway, "
    "observability, CI/CD, Savings Plans / Spot (incl. Fargate Spot — interruptions break the "
    "capacity guarantee), GPU models (Workers and Fargate have no GPU)."
)
