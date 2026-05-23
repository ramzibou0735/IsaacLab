# W&B Guide For AirGym X152b In IsaacLab

This document explains how to use Weights & Biases (W&B) with the AirGym x152b RSL-RL environments in this repository for:

- live training monitoring
- experiment comparison
- report generation
- export of plots for papers
- export of raw metrics for custom analysis

The AirGym x152b RSL-RL runner configs now default to:

- `logger = "wandb"`
- `wandb_project = "isaaclab-airgym-x152b"`

Affected task families:

- `Isaac-AirGym-X152b-Hovering-Direct-v0`
- `Isaac-AirGym-X152b-Balloon-Direct-v0`
- `Isaac-AirGym-X152b-Tracking-Direct-v0`
- `Isaac-AirGym-X152b-Avoid-Direct-v0`
- `Isaac-AirGym-X152b-Planning-Direct-v0`

## 1. Why Use W&B Here

W&B is the best fit for research monitoring in this codebase because it gives you:

- run tracking across many experiments
- grouped comparison of seeds and task variants
- interactive smoothing and filtering for scalar curves
- dashboards and reports for collaborators
- export paths for PDF / LaTeX reports and raw metric data

Official references:

- experiment tracking overview: https://docs.wandb.ai/models/track
- Python login API: https://docs.wandb.ai/models/ref/python/functions/login
- CLI login: https://docs.wandb.ai/models/ref/cli/wandb-login
- Public API overview: https://docs.wandb.ai/models/ref/python/public-api
- run history export via Public API: https://docs.wandb.ai/models/ref/python/public-api/run
- reports overview: https://docs.wandb.ai/models/reports
- report export to PDF / LaTeX: https://docs.wandb.ai/models/reports/clone-and-export-reports

Use W&B for the experiment system.
Use Matplotlib / Seaborn for final paper figures.

## 2. Authentication

### Option A: Interactive login

Run once in the same environment you use for training:

```bash
source env_isaaclab/bin/activate
wandb login --verify
```

This stores credentials locally.

### Option B: Environment variable

Useful for clusters, remote jobs, CI, or non-interactive shells:

```bash
export WANDB_API_KEY="your_api_key_here"
```

If you use a self-hosted W&B server:

```bash
export WANDB_BASE_URL="https://your-wandb-instance"
export WANDB_API_KEY="your_api_key_here"
```

## 3. Basic Training Command

Example smoke test:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Hovering-Direct-v0 \
  --num_envs 32 \
  --max_iterations 5 \
  --headless
```

Example real run:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Tracking-Direct-v0 \
  --num_envs 256 \
  --max_iterations 2000 \
  --run_name vel_seed42 \
  --headless
```

Because the AirGym runner configs now default to W&B, no extra logger flag is required.

## 4. Overriding Project Or Logger At Runtime

If needed, you can still override from CLI without editing code.

### Set a different W&B project

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Balloon-Direct-v0 \
  --log_project_name my-paper-project \
  --run_name seed01 \
  --headless
```

### Force TensorBoard for one run

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Hovering-Direct-v0 \
  --logger tensorboard \
  --headless
```

### Change the experiment folder name on disk

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Avoid-Direct-v0 \
  --experiment_name airgym_x152b_avoid_ablation \
  --run_name cnn_variant_a \
  --headless
```

## 5. Good Run Naming For Research

Use names that make cross-run comparison obvious.

Recommended pattern:

```text
<control_mode>_<task_variant>_seed<seed>_<ablation>
```

Examples:

- `vel_seed42`
- `vel_seed43`
- `vel_reward_ablation_a_seed42`
- `atti_seed42`
- `vision_cnn32_seed42`

This makes W&B grouping and filtering much easier.

## 6. What You Should Monitor

At minimum, monitor these metrics in W&B:

- training reward
- episode length
- success rate, if the task exposes one
- termination breakdowns
- per-reward-component curves
- policy loss, value loss, entropy, KL
- throughput and wall-clock time

This AirGym port already writes reward decomposition into `extras["item_reward_info"]` inside the environments. Those values are the right place to start when building report panels.

Suggested research dashboards:

- total reward over environment steps
- selected reward terms over environment steps
- episode length over environment steps
- success / failure rate by task
- training stability across seeds
- comparison by control mode: `vel`, `atti`, `rate`, `prop`

## 7. W&B Workspace Workflow

Recommended workflow in the W&B app:

1. Open the project workspace.
2. Group runs by task and seed.
3. Pin the key scalar charts you care about.
4. Apply smoothing only for presentation, not for debugging.
5. Save a report once the chart layout stabilizes.

W&B reports are the right place to assemble the figures, notes, and comparisons that later turn into paper plots.

Official references:

- reports overview: https://docs.wandb.ai/models/reports
- create reports: https://docs.wandb.ai/models/reports/create-a-report
- export reports: https://docs.wandb.ai/models/reports/clone-and-export-reports

## 8. Exporting Plots For Papers

### Option A: Export a W&B report

Inside the report UI:

1. open the report
2. click the kebab menu
3. choose `Download`
4. select `PDF` or `LaTeX`

This is useful for:

- appendices
- internal reports
- collaborator review
- quick benchmark summaries

Official reference:

- https://docs.wandb.ai/models/reports/clone-and-export-reports

### Option B: Export data and replot with Matplotlib / Seaborn

This is the preferred path for publication-quality final figures because it gives you exact control over:

- fonts
- line width
- confidence intervals
- figure size
- panel spacing
- vector output (`PDF`, `SVG`, `EPS` depending on backend)

Official references:

- Matplotlib `savefig`: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
- Seaborn docs: https://seaborn.pydata.org/

## 9. Exporting Raw Metrics From W&B

W&B Public API is the cleanest path for exporting run histories.

Official references:

- Public API overview: https://docs.wandb.ai/models/ref/python/public-api
- Run API including `history()` and `download_history_exports()`: https://docs.wandb.ai/models/ref/python/public-api/run
- Runs API: https://docs.wandb.ai/models/ref/python/public-api/runs

### Example: export sampled history to CSV

```python
import wandb
import pandas as pd

api = wandb.Api()
run = api.run("your_entity/isaaclab-airgym-x152b/your_run_id")

df = run.history(
    keys=["Train/mean_reward", "Train/mean_episode_length"],
    pandas=True,
)

df.to_csv("run_history_sampled.csv", index=False)
```

Use this when:

- you only need the main curves
- downsampling is acceptable
- you want a fast export path

### Example: export many runs from one project

```python
import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()
runs = api.runs("your_entity/isaaclab-airgym-x152b")
out_dir = Path("wandb_exports")
out_dir.mkdir(exist_ok=True)

for run in runs:
    df = run.history(
        keys=["Train/mean_reward", "Train/mean_episode_length"],
        pandas=True,
    )
    safe_name = f"{run.name or run.id}".replace("/", "_")
    df.to_csv(out_dir / f"{safe_name}.csv", index=False)
```

### Example: download parquet history exports

```python
import wandb

api = wandb.Api()
run = api.run("your_entity/isaaclab-airgym-x152b/your_run_id")
result = run.download_history_exports("history_exports")
print(result)
```

Use this when:

- you want higher-fidelity time series export
- you need reproducible post-processing for many runs
- you are building benchmark aggregation scripts

## 10. Creating Publication Figures Locally

After exporting CSV data, use Matplotlib / Seaborn.

### Example: single run curve

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

df = pd.read_csv("run_history_sampled.csv")

plt.figure(figsize=(4.0, 3.0))
plt.plot(df["_step"], df["Train/mean_reward"], linewidth=1.8)
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.tight_layout()
plt.savefig("mean_reward.pdf")
plt.savefig("mean_reward.svg")
```

### Example: multi-seed mean with uncertainty band

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper")

frames = []
for csv_path in Path("wandb_exports").glob("*.csv"):
    df = pd.read_csv(csv_path)
    df["run"] = csv_path.stem
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

plt.figure(figsize=(4.4, 3.2))
sns.lineplot(
    data=all_df,
    x="_step",
    y="Train/mean_reward",
    estimator="mean",
    errorbar=("ci", 95),
)
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.tight_layout()
plt.savefig("reward_multiseed.pdf")
```

This is usually the figure you want in a paper, not the direct screenshot from W&B.

## 11. Recommended Research Pipeline

Use this workflow:

1. train with W&B enabled
2. compare runs in the W&B workspace
3. create a W&B report for internal review
4. export raw metrics with the Public API
5. regenerate final plots locally with Matplotlib / Seaborn
6. save final figures as `PDF` or `SVG`

This gives you:

- good online tracking during training
- reproducible final figures
- cleaner paper visuals than dashboard screenshots

## 12. Common Failure Modes

### Login works locally but not on cluster

Use environment variables:

```bash
export WANDB_API_KEY="..."
export WANDB_BASE_URL="..."   # only if self-hosted
```

### Wrong project name

Override at launch:

```bash
--log_project_name my-project
```

### Want local logging only for one run

Override at launch:

```bash
--logger tensorboard
```

### Need run names to be more informative

Use:

```bash
--run_name vel_seed42_ablation1
```

## 13. Suggested Commands For This Repo

### Hovering

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Hovering-Direct-v0 \
  --num_envs 256 \
  --run_name vel_seed42 \
  --headless
```

### Balloon

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Balloon-Direct-v0 \
  --num_envs 256 \
  --run_name vel_seed42 \
  --headless
```

### Tracking

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Tracking-Direct-v0 \
  --num_envs 256 \
  --run_name vel_seed42 \
  --headless
```

### Avoid

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Avoid-Direct-v0 \
  --num_envs 128 \
  --run_name vel_seed42 \
  --headless
```

### Planning

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-AirGym-X152b-Planning-Direct-v0 \
  --num_envs 128 \
  --run_name vel_seed42 \
  --headless
```

## 14. Bottom Line

For this repository:

- default monitor: W&B
- final publication figures: Matplotlib / Seaborn
- report export: W&B PDF / LaTeX
- raw data export: W&B Public API

That gives you the best research workflow without changing the training pipeline structure.
