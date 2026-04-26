# Blacklight Training Pipeline

This is the standard learning workflow.

## Rule

Blacklight has one current model. Normal training updates that current model
in-place after replaying fresh teacher examples.

Do not train by collecting a pile of checkpoints and picking a winner.

If another model already exists, treat it as a learning source. Distill a small
amount of its weights into the current model instead of replacing the current
model outright.

Vast.ai is different when it starts from the current champion. In that case the
remote output is not a competing model; it is the current champion after more
training, so `vast apply` installs it back into the single local current model
path after making a rollback backup.

## Hardware Split

The workflow is CPU plus GPU:

- `collect` uses CPU worker processes to run Armagetron games and record teacher
  examples.
- `train` uses the PyTorch learner on GPU/MPS/CUDA by default to replay those
  examples into the CNN.
- `train --cpu` is only a fallback when the GPU learner is unavailable.

## Normal Run

Collect fresh teacher data from the current best model:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh collect --resume current --duration 21600 --parallel-workers 4 --sync-seconds 300
```

Replay the latest collected data into the same current model. This uses GPU/MPS
by default on this machine:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh train --in-place --resume current
```

Force CPU replay only if the GPU learner is broken:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh train --in-place --cpu --resume current
```

## Vast.ai GPU Run

Use this when you want the Vast.ai website to do the replay/training step
instead of the Mac. The Mac still does the CPU-oriented teacher collection; the
Vast instance does the GPU-oriented learner pass. This still keeps one local
model: the bundle starts from `current_model.txt`, Vast writes one
`output/current_model.txt`, and `vast apply` installs that trained continuation
back into the local current model path.

First collect local teacher data:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh collect --resume current --duration 21600 --parallel-workers 4 --sync-seconds 300
```

Build the upload bundle:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh vast setup --epochs 4 --batch-size 4096
```

On the Vast.ai website, rent one SSH or Jupyter instance using a PyTorch/CUDA
image. For RTX 5090, use a CUDA 12.8 / PyTorch 2.7+ compatible template.

The `vast setup` command prints a `control_kit` folder. Edit
`control_kit/vast.env` with the host and port from the Vast SSH command, then
run:

```bash
./01_upload.sh
./02_start_training.sh
./03_watch.sh
./04_download.sh
./05_apply.sh
./06_benchmark.sh
```

Those helper scripts upload the bundle, start training in the Vast instance,
watch the remote log, download `output/current_model.txt`, install it into the
one current model path, and run the first sanity benchmark.

If you want to run the remote commands manually instead, upload the printed
`.tar.gz` bundle to `/workspace` and run:

```bash
cd /workspace
tar -xzf blacklight_vast_YYYYMMDD-HHMMSS.tar.gz
cd blacklight_vast_YYYYMMDD-HHMMSS
./remote_train.sh
```

After downloading the remote output, apply it locally as the trained champion:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh vast apply /path/to/current_model.txt
```

To distill any local model source directly:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh current distill --from /path/to/source_model.txt --alpha 0.10
```

The current model is backed up before the distilled model is installed. The
backup is rollback material only; it is not a second champion.

For normal Vast runs, use `vast apply` rather than `current distill`, because
the Vast output started from the champion and is the champion after more
training. Use `current distill` only when absorbing an older or separate model.

Useful Vast tuning knobs:

```bash
BLACKLIGHT_VAST_EPOCHS=4 BLACKLIGHT_VAST_BATCH_SIZE=8192 ./remote_train.sh
BLACKLIGHT_VAST_DEVICE=cuda ./remote_train.sh
BLACKLIGHT_VAST_MAX_EXAMPLES=200000 ./remote_train.sh
```

Vast.ai reference pages:

- SSH instances and SCP:
  https://docs.vast.ai/documentation/instances/connect/ssh
- Jupyter instances:
  https://docs.vast.ai/documentation/instances/connect/jupyter
- Instance overview:
  https://docs.vast.ai/documentation/instances/overview

## Benchmark Check

If you want a sanity check after a training phase, benchmark the current model:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh bench --suite classic_primary --candidate current --sessions 3 --duration 45 --rounds 30
```

Read survival/placement and average distance first. Win rate is useful, but it
is too sparse to be the only signal for this project. If training gets worse,
stop and collect better teacher examples before continuing.

## Watch

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh monitor --interval 1
```

## Clean Temporary Files

Dry run first:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh prune --dry-run
```

Then apply:

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh prune --apply
```

## Smoke Test

```bash
cd /Users/romilbijarnia/Desktop/Armagetron
./scripts/blacklight.sh smoke
```
