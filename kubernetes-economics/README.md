# Sub-300 ms ML Inference: EKS vs ECS vs Cloudflare Workers

An app needs an ML model to answer in **under 300 ms**. Which platform serves it at the lowest cost: EKS or ECS (each on EC2 nodes or on Fargate), or Cloudflare Workers?

**"A sub-300 ms ML model" is a reference point, not a specific product.** It stands for the **higher end of practical compute inside a request path**: a function heavy enough that where and how it runs matters, yet light enough to answer synchronously within a user-facing latency budget. The default is a 20 MB model, about 20 ms of native CPU per request and 40 ms on Workers. That's roughly a small quantized text encoder, far more than typical serverless code (routing, auth, formatting) spends.
- **Lighter functions** (tree-model scoring, small tabular nets, business logic) fit comfortably on any option, and the fixed costs decide.
- **Heavier ones** (large models, GPUs) leave the synchronous request path, or at least this comparison.

```docker-compose up``` then open http://localhost:8502 to try the dashboard. Docker is the only requirement.

## The goal

**Meeting latency under load is the hard constraint. Cost is what gets optimized.**

| | |
|---|---|
| **Hard constraints** (pass or fail) | 1. **Latency:** end-to-end p99 ≤ SLO at every volume: user → (app backend →) model → back, including handshakes and cold paths.<br>2. **Capacity:** enough for expected traffic at every moment, including surges, and never below 2 small nodes or tasks per region.<br>3. **Fit:** the model fits the platform (≈ 90 MB in a Workers isolate). |
| **Objective** | Minimize the monthly cloud bill. |
| **Choices** | Platform × setup: AWS picks 1 or 3 regions, and ALB or CloudFront in front, on EC2 nodes or Fargate; Workers picks on demand or kept warm. Each option uses its cheapest setup that passes the constraints. |
| **Tie-break** | The simpler option wins within 5% per step of complexity: no nodes beats nodes, ECS beats EKS. |
| **Given** (you know these) | AWS region, how the app calls the model, where users are, model size, SLO. |
| **Swept** (uncertain, and they change the answer) | Request volume and surge speed: the axes of the charts. |

Every other number is fixed at a realistic, sourced value and listed in the dashboard's appendix.

## The options

| | On EC2 nodes | Serverless |
|---|---|---|
| **Kubernetes** | **EKS on EC2**: pods on nodes you run (Karpenter) | **EKS on Fargate**: each pod gets its own Fargate micro-VM |
| **ECS** | **ECS on EC2**: tasks on nodes you run (capacity provider) | **ECS on Fargate**: each task gets its own Fargate micro-VM |
| **Edge** | | **Cloudflare Workers**: the model runs inside the isolate (WASM, CPU only, 128 MB memory) at the PoP nearest the caller |

All AWS options sit behind an ALB and run the same container, so their latency is identical. They differ in cost structure and in how fast they add capacity.

## Constraint 1: latency, end to end

How the app calls the model changes which latency costs apply at all. In backend mode, **the backend runs on the same platform as the model**: an AWS backend for the AWS options, a backend Worker for Workers. The SLO covers the whole round trip: app → backend → model → backend → app.

| | App calls the model directly | App → its backend → model |
|---|---|---|
| Handshake inside p99 | Yes. TCP + TLS 1.3 to an ALB costs +2 RTT; HTTP/3 at an edge (Cloudflare, CloudFront) costs +1 | Only to the backend, via an HTTP/3 edge; the backend → model call is pooled or in-process |
| User distance | Paid to the model | Paid to the backend: the AWS region, or the nearest PoP |
| Workers | Edge advantage; traffic spread thin across PoPs, so cold starts | Backend Worker and model Worker both at the nearest PoP, joined by a service binding: no network hop, no extra request fee (only CPU is billed). Traffic still spreads across PoPs, so cold starts remain |
| AWS | ALB, or CloudFront in front (edge handshake, ~$1/M requests) | Backend and model in the same region: internal call (~2 ms), no egress |
| Global audience | AWS in 1 region (if the model is fast enough) or 3 | Backend and model in 3 regions, so AWS pays 3 floors; Workers is already everywhere at no fixed cost |

**Volume changes latency, not just cost.** A Workers isolate is evicted when it sits idle, and a CloudFront origin connection closes. At low volume spread across many locations, many requests land cold and pay a model load or a handshake. Once fewer than 1% of requests are cold, that cost drops out of p99. AWS nodes behind a plain ALB are always warm, but direct callers pay the full handshake to the region every time.

**Model size sets compute.** Answering one request touches every weight at least once, so compute time grows with size: ~1 ms per MB on one native core, averaged across model types. Workers runs it ~2× slower in WASM.

## Constraint 2: capacity under load

AWS capacity follows traffic through the day (autoscaling), never below the smallest always-on setup per region: 2 × c6i.large nodes on EC2, or 2 × 0.5 vCPU tasks on Fargate (one per AZ). Above that it's billed per vCPU needed. On EC2 that's because Karpenter and ECS capacity providers mix instance sizes; on Fargate, tasks are 1 vCPU / 2 GB each. It also holds headroom for the growth a surge adds before new capacity is serving. The daily peak costs nothing extra. What costs extra is surges that outrun the scaler.

| Surge begins → new capacity serving | Time | Main cost |
|---|---|---|
| EKS on EC2 (HPA + Karpenter) | ~1.5–2 min | HPA reacts in 15 s; a node is Ready in 45–70 s |
| EKS on Fargate (HPA) | ~1.5–2.5 min | Each pod starts on a fresh micro-VM (~30–60 s), then registers with the ALB |
| ECS on Fargate, tuned (step scaling) | ~2–3.5 min | The alarm (60–120 s), then task start and ALB health check; no node stage |
| ECS on EC2, tuned (step scaling) | ~3–4.5 min | The capacity provider → Auto Scaling group → EC2 boot stage takes 2–4 min |
| ECS, default (target tracking) | ~6.5–10 min | The alarm alone takes ~6 min (AWS: 363 s) |
| Workers | per request | No headroom; new isolates start cold |

The model uses tuned ECS, because tuning is a free configuration choice.

## Nodes or serverless: EC2 vs Fargate

Fargate removes nodes: no instances to manage, no node overhead, no node-launch stage when scaling out. Its price is a higher rate per vCPU.

| Singapore, per usable vCPU-hour | EC2 (c6i) | Fargate (1 vCPU / 2 GB) |
|---|---|---|
| ECS | $0.052 (95% of the node usable) | $0.062 (+19%) |
| EKS | $0.058 (85% usable) | $0.062 (+7%), plus 4 system pods (CoreDNS, LB controller) on Fargate |

What that does to the winner (backend mode, one continent, Singapore):
- **Among AWS options, ECS on Fargate is the cheapest at lower volume.** Its floor costs **$63/mo**, against $161 for ECS on EC2. It stays cheaper up to ~170M–310M req/mo, depending on surge speed. Above that, EC2's lower per-vCPU price wins. Fargate's faster scale-out (~2.5 vs ~3.5 min) trims its surge headroom but doesn't close the gap.
- **Overall, Fargate wins outright only where Workers can't play.** With a 20 MB model, Workers is cheaper at those volumes. With a 95 MB model (too big for an isolate): **ECS on Fargate** up to ~40M req/mo → ECS on EC2 → EKS on EC2 from ~330M.
- **EKS on Fargate never wins.** It pays the control plane, its system pods and the Fargate premium. Its scale-out (~2 min) is only slightly faster than ECS on Fargate.

Techniques each Fargate option uses to meet the constraints at the lowest cost:
- **The same setup choices as EC2:** 1 or 3 regions, ALB or CloudFront, next to the backend.
- **Right-sized tasks.** 1 vCPU / 2 GB serving tasks. The floor runs 0.5 vCPU tasks, which still finish a lone request at full speed.
- **EKS pods request 256 MB less**, so Fargate's added 256 MB for Kubernetes components doesn't bump them to a larger billed size.
- **Excluded:** Fargate Spot (ECS only, up to 70% off). A 2-minute interruption notice breaks the capacity guarantee, the same reason EC2 Spot is excluded. Fargate also never caches images, so every new task pulls the full image; SOCI lazy loading helps on ECS. Image pull is left out for all options.

## The choices: being close, being warm and fronting with an edge

- **EKS / ECS: 1 region, or 3 regions for a global audience.** One region can't serve the whole world under 300 ms: fiber alone costs 160–200 ms round trip to antipodal users, and the p99 RTT is ~250 ms. Three regions (us-east-1, eu-west-1, ap-southeast-1) bring it down to ~90 ms, but each region pays its own floor.
- **EKS / ECS + CloudFront (direct calls).** The HTTP/3 handshake happens at a nearby edge, and the request rides a keep-alive connection to the ALB. That connection stays open only up to 60 s, so at low volume per edge it goes cold too, the same math as a Workers isolate.
- **CF Workers: on demand, or kept warm.** Synthetic probes from a monitoring service (~22 locations, ~$5 per 10k runs) ping every 4 minutes so isolates never idle out. Probes only reach the biggest PoPs, so the long tail stays on demand.

## From zero to 10B requests a month

Scale-to-zero is a real case for a startup, so the range starts at effectively nothing: the volume axis runs from 100 req/mo (~3 a day) to 10B (~3,800 req/s on average), and the headline states the exact cost at zero traffic.

- **At zero:** Workers costs its $5 base, plus probes if it needs keeping warm. **It's the only option that scales to zero and still meets the SLO.** EKS and ECS can scale to zero, but the first request after idle waits for an EC2 instance to boot (minutes) or a Fargate task to start (tens of seconds). Their real floor is the smallest always-on setup: 2 small nodes or tasks and an ALB per region, plus the control plane for EKS. In Singapore, one region: **ECS on Fargate $63**, ECS on EC2 $161, EKS on Fargate $189, EKS on EC2 $234 a month.
- **Above 10B:** every cost scales linearly and the winner stops changing. At that scale, list prices also stop being realistic (negotiated pricing, Savings Plans, tiered egress, GPUs), so the model stops there.

## When EKS wins, and when it doesn't

Against ECS on EC2, EKS on EC2 wins only in a pocket: surges that double in ~3–5 minutes, at ~1B+ requests a month. On the winner map the pocket starts later (~4B at the defaults), because Workers stays within 5% of both until then and wins the tie as the simpler option. EKS on Fargate never wins (see above), so this section compares EKS and ECS on EC2. A natural expectation is "burstier traffic → more EKS", so this needs explaining.

**EKS has a fixed penalty and one lever.**
- **Penalty:** only ~85% of each 4-vCPU node is usable, because Kubernetes system agents (kubelet, CNI, kube-proxy, log/metrics daemonsets) take ~0.5 vCPU. On ECS it's ~95%. EKS also pays $73/mo for its control plane.
- **Lever:** scale-out speed. EKS adds capacity in ~1.75 min, tuned ECS in ~3.5 min. Each platform holds spare capacity for the growth that arrives before new capacity is serving, and a surge tops out at ×2, so neither ever holds more than 2×.

**EKS's advantage is the gap between the two platforms' spare capacity, not the amount of burst.** The gap closes at both ends:

| Load doubles within | EKS spare capacity | ECS spare capacity | Nodes EKS needs vs ECS | Winner |
|---|---|---|---|---|
| 60 min | 1.03× | 1.06× | +9% | ECS: both barely need spare capacity; EKS's overhead dominates |
| 10 min | 1.18× | 1.35× | −3% | ECS (within the 5% tie margin) |
| **3.5 min** | **1.5×** | **2.0×** (already at the cap) | **−16%** | **EKS: the widest gap** |
| 1.75 min or faster | 2.0× (at the cap) | 2.0× | +12% | ECS: both hold the whole surge; only the overhead differs |

- **Slow surges:** both scalers keep up. EKS's speed is worth almost nothing, and its overhead still applies.
- **Fast surges:** both scalers are too slow and must keep the whole surge ready at all times. Speed is worth nothing again.
- **In between:** ECS has hit the cap but EKS can still catch part of the surge. EKS's advantage peaks right at ECS's reaction time (~3.5 min).

**Why the pocket also narrows to the left:** at lower volume both sit at their 2-node minimum, so spare capacity doesn't matter and EKS's control plane decides. EKS needs enough load for its node savings to outgrow that fixed cost:

| Total cost, EKS ÷ ECS | Doubles in 2.5 min | 3.5 min | 5 min | 10 min |
|---|---|---|---|---|
| 500M req/mo | 1.13 | 1.03 | 1.10 | 1.23 |
| 1B req/mo | 1.04 | **0.94** | 1.00 | 1.11 |
| 3B req/mo | 0.98 | **0.88** | **0.93** | 1.02 |
| 10B req/mo | 0.96 | **0.85** | **0.90** | 0.99 |

Bold = EKS wins (at least 5% cheaper). The table is from the dashboard's model: backend mode, one continent, Singapore, 20 MB model.

**Two assumptions set the pocket's size:**
- **Surge size (×2).** If load keeps climbing past ×2, a faster scaler catches a growing share of it, and the pocket reaches down toward faster bursts. At ×5 surges, EKS wins from ~1 min doubling and is up to ~30% cheaper. This is where "bursty favors EKS" holds.
- **EKS node overhead (~15% of a 4-vCPU node).** The system agents cost about the same absolute CPU on any node, so on 16-vCPU nodes they're only ~3–5%. With that penalty mostly gone, EKS would win from much smaller bursts.

"We already run an EKS cluster" is not a reason. Capacity that is already paid for or already running isn't free. It could be scaled down or used for something else, so keeping it is a choice with a cost like any other. If ECS is sufficient, ECS is the choice.

## The dashboard

The headline names every option within 5% of the cheapest ("A ≈ B"). In backend mode, Workers, ECS on EC2 and EKS on EC2 stay within ~10% of each other from ~300M to ~4B req/mo, so what decides that range is the tie rule, not a clear price gap.

1. **Monthly cost vs request volume.** Dotted where an option misses the SLO; hover names the setup chosen.
2. **p99 latency vs request volume.** End to end, against the SLO line. All four AWS options share one line, because their latency is identical.
3. **Winner map.** Volume × surge speed, with each cell showing the cheapest option that meets the constraints.

**Encoding:** color = orchestrator (Workers blue, EKS orange, ECS aqua). Running on EC2 nodes is marked on top: dashed lines on the cost chart, dotted cells on the map. Serverless (Workers, Fargate) is plain. Five separate hues can't all stay distinguishable where any two map regions touch, and "nodes or not" is the structural difference anyway.
4. **Appendix.** The latency budget broken down per setup, and every fixed number with its source.

## Limits of the framing

- **Constraints are all-or-nothing.** 301 ms fails the same as 900 ms. Borderline cases can flip on small assumption changes, and p99 is estimated by adding each component's p99, which overstates the total by roughly 10–20%.
- **Cost is the cloud bill only.** Not included: ops labor, the app's own backend compute (it runs on the same platform as the model, but its cost depends on the app, not the model), NAT gateway, observability, CI/CD, Savings Plans / Spot (including Fargate Spot).
- **"Under load" covers traffic, not failures.** Surviving the loss of an Availability Zone without breaking the SLO would need extra idle AWS capacity (~1.5× with 3 AZs). That isn't modeled.
- **Out of scope:** GPU models (Workers and Fargate have no GPU) and Workers AI.

## What the numbers say

<!-- screenshot + takeaway -->
