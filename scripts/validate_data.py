"""Audit downloaded DriveLM annotations and camera images before training."""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    report = {}
    scenes_by_split = {}
    images_by_split = {}
    for split, filename in (
        ("train", "v1_1_train_nus.json"),
        ("val", "v1_1_val_nus_q_only.json"),
    ):
        with (root / filename).open() as stream:
            data = json.load(stream)
        scenes_by_split[split] = set(data)
        frame_count = 0
        categories = Counter()
        answers = 0
        empty_questions = 0
        usable_pairs = 0
        missing = []
        images = set()
        for scene in data.values():
            for frame in scene["key_frames"].values():
                frame_count += 1
                if len(frame["image_paths"]) != 6:
                    raise ValueError("Expected six camera paths per frame")
                for path in frame["image_paths"].values():
                    parts = PurePosixPath(path).parts
                    if parts and parts[0] == "..":
                        parts = parts[1:]
                    resolved = (root / Path(*parts)).resolve()
                    if root not in resolved.parents:
                        raise ValueError(f"Image path escapes data root: {path}")
                    images.add(resolved)
                for category, questions in frame["QA"].items():
                    categories[category] += len(questions)
                    for qa in questions:
                        has_question = bool(isinstance(qa.get("Q"), str) and qa["Q"].strip())
                        has_answer = bool(isinstance(qa.get("A"), str) and qa["A"].strip())
                        empty_questions += not has_question
                        answers += has_answer
                        usable_pairs += has_question and has_answer
        for image in sorted(images):
            if not image.is_file():
                missing.append(str(image))
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[:3]}")
        # ZIP CRC checks cover every image; fully decode a deterministic sample
        # from every camera to additionally validate the image loader.
        cameras = sorted({path.parent.name for path in images})
        decoded = 0
        for camera in cameras:
            for path in sorted(p for p in images if p.parent.name == camera)[:3]:
                with Image.open(path) as image:
                    image.convert("RGB").load()
                decoded += 1
        images_by_split[split] = images
        report[split] = {
            "scenes": len(data),
            "frames": frame_count,
            "questions_by_category": dict(categories),
            "total_questions": sum(categories.values()),
            "nonempty_answers": answers,
            "empty_questions": empty_questions,
            "usable_supervised_pairs": usable_pairs,
            "unique_images": len(images),
            "missing_images": 0,
            "sample_images_decoded": decoded,
        }
    report["shared_scenes"] = len(scenes_by_split["train"] & scenes_by_split["val"])
    report["shared_images"] = len(images_by_split["train"] & images_by_split["val"])
    if report["shared_scenes"] or report["shared_images"]:
        raise ValueError("Published train and validation splits overlap")
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
