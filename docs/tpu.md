# TPU Setup & Workflow

## TPU Factsheet

```
  Trillium (v6e)
  ──────────────────────────────────────────
  HBM per chip        32 GB
  BF16 FLOPS          918 TFLOPS
  HBM bandwidth       1,640 GB/s
  ICI bandwidth       3,584 Gbps
  On-demand/chip/hr   $2.70 (US regions)
  ──────────────────────────────────────────

  Dataset
  ──────────────────────────────────────────
  Text                2M Tool-call pairs
                      (query, tools, answers)
                      synthesized via Gemini
                      (local: data/tool_calls_unified/)
  ──────────────────────────────────────────

  ┌────────────────────┬──────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │  v6e-32  │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │ 32       │
  │ Total HBM          │ 256 GB   │ 512 GB   │ 1024 GB  │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │ 0.7×     │
  │ Eff. TFLOPS        │ 994      │ 1,766    │ 3,091    │
  │ Est. time          │ ~88h     │ ~49h     │ ~29h     │
  │ On-demand $/hr     │ $21.60   │ $43.20   │ $86.40   │
  │ Est. total cost    │ ~$1,900  │ ~$2,120  │ ~$2,510  │
  └────────────────────┴──────────┴──────────┴──────────┘
```

## CLI Reference

```
  dana tpu

    create NAME                Create TPU (auto-finds zone)
      --type STR               Accelerator (default: v6e-8)
      --version STR            TPU OS (auto-detected from --type)
      --preemptible            Create spot/preemptible instance
    connect NAME               SSH config + connect (auto-zone)
    setup NAME                 Full setup: sync code + venv + deps
    sync NAME                  Fast sync: code (no venv rebuild)
    train NAME [-- ARGS]       Launch training on all workers
    pretrain NAME [-- ARGS]    Launch pretraining on all workers
    claude NAME                Install Claude Code on instance
    stop NAME                  Stop instance (auto-zone)
    start NAME                 Start stopped instance (auto-zone)
    delete NAME                Delete instance (auto-zone)
    list                       List all TPU instances
      --zone ZONE              Override auto-detected zone
```

## GCP Setup

- Download the `macOS ARM` gcloud SDK from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and unzip.
- Open terminal, cd to your downloads and run `./google-cloud-sdk/install.sh`
- Restart terminal and run `gcloud init`, sign in with cactus email, should prompt for project
- Else, set the project with `gcloud config set [PROJECT NAME]`
- Run `gcloud help` and read carefully

## Single-host (v6e-8) - SSH into instance

```
1. Create an instance (auto: finds zone -> installs Claude Code -> connects via SSH)
   dana tpu create my-experiment
   (exit with 'exit' or Ctrl+D)

2. Reconnect anytime
   ssh my-experiment
   or VS Code: click the '><' in the bottom left -> select my-experiment

--- run from the instance ---

3. Create a GitHub Personal Access Token (PAT)
   GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
   Generate a token with 'repo' scope

4. Clone the repo on your instance using your PAT
   git clone https://github.com/cactus-compute/needle.git
   cd dana && source ./setup

5. Use tmux for long training runs (survives SSH disconnects)
   tmux new -s train          # start a named session
   dana train --wandb       # run training inside it
   Ctrl+B, then D             # detach (keeps running)
   tmux attach -t train       # reattach later

--- back on your Mac ---

6. Stop when done (saves disk, stops billing)
   dana tpu stop my-experiment

7. (Optional) Delete instance when no longer needed
   dana tpu delete my-experiment
```

## Multi-host (v6e-16+) - run from your Mac

For multi-host TPUs (v6e-16+), you drive everything from your Mac.
The CLI syncs code and launches training across all workers automatically.

```
1. Set auth tokens (workers need this to log metrics)
   export WANDB_API_KEY=...

2. Add SSH key to agent (required for gcloud scp/ssh)
   ssh-add ~/.ssh/google_compute_engine

3. Create a multi-host TPU (auto: finds zone -> syncs code -> installs deps)
   dana tpu create my-experiment --type v6e-16

4. Launch training on all workers from your Mac
   dana tpu train my-experiment -- --wandb --epochs 1

5. After code changes, sync without rebuilding venv (fast, ~seconds)
   dana tpu sync my-experiment

6. Full re-setup if deps changed (slow, rebuilds venv)
   dana tpu setup my-experiment

7. Stop/delete when done
   dana tpu stop my-experiment
   dana tpu delete my-experiment
```
