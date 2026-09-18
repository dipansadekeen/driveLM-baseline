# Local DriveLM startup

Upstream: https://github.com/OpenDriveLab/DriveLM

Checked out revision: `1de72a74b257e5373400fa68239e99bd5d20580a`

Local branch: `setup/local-demo`. This is a shallow upstream clone, not yet
a GitHub fork. No authenticated GitHub CLI, token, or configured credential
helper was available during setup.

## Completed

Run from this repository root:

```bash
python3 scripts/smoke_data.py
```

This runs the upstream extraction and conversion functions on the bundled
sample, with random seed 0. It checks unique question IDs, six views per
question, and decodes the referenced images. The first run passed with
66 questions and 54 unique images across 9 frames in 2 scenes.

Generated files live in `local_runs/sample/`, including `report.json`,
`test_eval.json`, `test_llama.json`, and a one-question input for first inference.
These are data preparation results, not new model predictions.

Both host GPUs passed a tiny CUDA tensor calculation with PyTorch 2.8.0+cu128.
GPU access requires execution outside the restricted shell. Each GPU is an
RTX 4090 with 24 GB VRAM; roughly 13.65 GiB and 9.81 GiB were free at the check.
Other workloads were left running.

## Requirements before model inference

1. Original LLaMA 7B backbone weights in the upstream layout:

   ```text
   llama_weights/
     tokenizer.model
     7B/
       consolidated.00.pth
       params.json
   ```

2. A compatible LLaMA-Adapter V2 checkpoint from the source linked in
   `challenge/README.md`. No model checkpoints are bundled in this checkout.
3. An isolated baseline environment. It has not been installed yet. The
   upstream recipe uses Python 3.8 and pins PyTorch 2.0.0+cu117 and
   torchvision 0.15.1+cu117 in
   `challenge/llama_adapter_v2_multimodal7b/requirements.txt`. The host Python
   is 3.9.21 with PyTorch 2.8.0; baseline imports such as CLIP, timm, OpenCV,
   fairscale, sentencepiece, and torchvision are missing. Do not install
   the legacy requirements into the shared host environment.
4. Sufficient free GPU memory. The published batch-8 inference configuration
   uses about 35 GB VRAM. The upstream demo replicates the model per GPU;
   two cards do not pool their memory. Batch size 1 is a starting experiment,
   not a verified fit; model cache allocation may also need adjustment.

Once the environment and weights are ready, the first inference command is:

```bash
cd challenge/llama_adapter_v2_multimodal7b
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --llama_dir /absolute/path/to/llama_weights \
  --checkpoint /absolute/path/to/adapter_checkpoint.pth \
  --data ../../local_runs/sample/one_question.json \
  --output ../../local_runs/sample/prediction.json \
  --batch_size 1 --num_processes 1
```

This command has not been run. The upstream demo uses worker threads and
can write an empty output even if a worker fails; verify that the output
contains one answer with the input ID before treating inference as successful.
Full evaluation also has additional language-metric and API requirements.
CARLA inference is still marked unreleased in the upstream README.

## GitHub fork

Authenticate GitHub in the execution environment and identify the destination
account or organization. Then create the fork, preserve the official repository
as `upstream`, and set `origin` to the fork. No remote changes or pushes have
been performed.
