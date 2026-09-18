# Dataset and BLIP-2 reproduction status

The selected target is the paper's **BLIP-2 with FLAN-T5-XL**, not the
LLaMA-Adapter V2 challenge baseline included in this repository.

## Available code and weights

- The base model is available as
  [Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl).
- [Hugging Face PEFT](https://huggingface.co/docs/peft/en/package_reference/lora)
  provides LoRA training components.
- The authors have not provided the original DriveLM BLIP-2 training pipeline
  in this checkout. In [issue 136](https://github.com/OpenDriveLab/DriveLM/issues/136),
  a maintainer says release of that implementation is not promised.
- Building on the available base model and LoRA tooling would be a reproduction,
  not the authors' exact implementation. Plain image/question/answer fine-tuning
  does not reproduce the paper's graph context, behavior-to-motion stage, or
  trajectory tokenizer by itself.

No BLIP-2 fine-tuning has been run yet, and no DriveLM-tuned BLIP-2 checkpoint
has been obtained. The generic pretrained BLIP-2 weights have not been
downloaded as part of dataset preparation.

## Official DriveLM-nuScenes v1.1 data

The official [challenge instructions](../challenge/README.md) link Google Drive
and Hugging Face mirrors. The Hugging Face repository currently requires dataset
access approval. The authors' Google Drive links are publicly downloadable.
The dataset is listed under CC BY-NC-SA 4.0; the underlying nuScenes assets
retain their own distribution terms. See the [official dataset page](https://huggingface.co/datasets/OpenDriveLab/DriveLM).

This workspace uses `../data/drivelm/` relative to the repository root, as
agreed with the user. It is outside Git. The four source downloads are:

| File | Bytes |
| --- | ---: |
| `v1_1_train_nus.json` | 192,961,829 |
| `v1_1_val_nus_q_only.json` | 9,860,626 |
| `drivelm_nus_imgs_train.zip` | 3,483,205,396 |
| `drivelm_nus_imgs_val.zip` | 704,864,335 |

From the repository root, using the download environment already created in
the workspace:

```bash
../.tools/download-env/bin/python scripts/download_data.py \
  --destination ../data/drivelm
python3 scripts/validate_data.py --data-root ../data/drivelm
```

For a different machine, install `gdown==5.2.2` in an isolated Python environment
for the downloader, and Pillow for the validator.

The downloader checks official file sizes, parses JSON, computes local SHA-256
hashes, and verifies ZIP CRCs during extraction. HF masks its LFS hashes when
access is not granted, so no comparison against those hidden hashes is claimed.
Provenance is recorded in `download_manifest.json`. The validator checks all
referenced image paths and train/validation scene and image disjointness, and
decodes a deterministic sample from each camera. It writes
`validation_report.json` in the data directory.

The validation archive uses `val_data/CAM_*` internally. The downloader places
those images under `nuscenes/samples/CAM_*` to match the official annotations.

Completed audit on 2026-09-18:

| Split | Scenes | Frames | Images | Questions | Nonempty answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 696 | 4,072 | 24,432 | 377,956 | 377,955 |
| Validation | 149 | 799 | 4,794 | 15,480 | 0 |

All referenced images are present. ZIP integrity checks passed, 36 sampled
images decoded successfully, and no scenes or images overlap between splits.
The raw data contains one blank training answer and one blank validation
question. Preserve the original JSON files and filter invalid pairs in the
training loader. Source archives plus extracted data occupy about 8.1 GiB.

The published validation annotations contain questions only. Use a deterministic
scene-level holdout from the training annotations for local supervised validation;
do not randomly split individual QAs from the same scene across train and validation.
