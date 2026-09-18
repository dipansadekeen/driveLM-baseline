"""Run the upstream sample-data pipeline without weights or network access."""
import contextlib
import json
from pathlib import Path
import random
import sys

from PIL import Image


def main():
    repo = Path(__file__).resolve().parents[1]
    challenge = repo / "challenge"
    output = repo / "local_runs" / "sample"
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(challenge))
    from extract_data import extract_data
    from convert_data import loop_test
    from convert2llama import convert2llama

    random.seed(0)
    with (output / "preparation.log").open("w") as log:
        with contextlib.redirect_stdout(log):
            extract_data(str(challenge / "data/train_sample.json"), str(output / "test.json"))
            loop_test(str(output / "test.json"), str(output / "test_eval.json"))
            convert2llama(str(output / "test_eval.json"), str(output / "test_llama.json"))

    records = json.loads((output / "test_llama.json").read_text())
    if not records:
        raise ValueError("The upstream pipeline produced no questions")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    image_paths = set()
    for record in records:
        if len(record["image"]) != 6:
            raise ValueError("Expected six camera views per question")
        if not record["conversations"][0]["value"].startswith("<image>\n"):
            raise ValueError("Missing image prompt prefix")
        for relative_path in record["image"]:
            image_paths.add(challenge / "llama_adapter_v2_multimodal7b" / relative_path)
    for path in sorted(image_paths):
        with Image.open(path) as image:
            image.verify()

    # Keep the original relative paths for the documented model working directory.
    (output / "one_question.json").write_text(json.dumps(records[:1], indent=2) + "\n")
    report = {
        "status": "passed",
        "scope": "Upstream data preparation and bundled image decoding only; no model inference",
        "questions": len(records),
        "unique_images": len(image_paths),
        "views_per_question": 6,
        "random_seed": 0,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
