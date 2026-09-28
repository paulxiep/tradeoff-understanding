# Problem Understanding

Small studies of what a technical decision actually trades off, done before building anything.

Each folder takes one real-world question, reduces it to an explicit model and turns that into an interactive dashboard. The point isn't a prediction. It's to make the tradeoff visible: which assumptions drive the answer, where it flips, and what the requested solution misses.

## How each study is framed

- **Separate constraints from the objective.** Some things must hold (no counterfeit reaches inventory; p99 stays under the SLO). Everything else is optimized, usually cost.
- **Separate what you know from what you don't.** Known facts (prices, region, model size) are inputs. The uncertain factors that change the answer become the chart axes. Everything else is fixed at a sourced value and listed.
- **Count every choice at full cost.** Keeping capacity warm, running extra regions, or keeping an existing cluster are choices with costs, never free defaults.

## Studies

| Folder | Question | Constraint → objective | What the model shows |
|---|---|---|---|
| [goods-authentication](goods-authentication) | Should ML automatically reject counterfeit luxury goods to cut expert labor? | No fake reaches inventory → minimize the cost of expert time plus wrongly rejected genuine items | At single-digit fake rates, one false rejection costs more than the expert time the model saves. The lever is **augmenting** experts, not automating them. |
| [kubernetes-economics](kubernetes-economics) | EKS or ECS (on EC2 nodes or Fargate), or Cloudflare Workers, for an app's sub-300 ms ML model? The model is a reference point for the higher end of practical compute inside a request path. | End-to-end p99 under load, capacity for expected traffic → minimize the monthly cloud bill | How the app calls the model (directly or through its backend) decides which latency costs apply at all. Workers has no fixed floor but pays in cold starts wherever traffic is thin. AWS pays a floor per region: smallest on Fargate, while EC2's lower per-vCPU price wins at scale. EKS beats ECS only for surges its faster scaler can catch and ECS's can't. |

## Running a dashboard

Each folder runs with Docker alone:

```bash
docker-compose up
```

goods-authentication serves on http://localhost:8501, kubernetes-economics on http://localhost:8502. Both can run side by side.
