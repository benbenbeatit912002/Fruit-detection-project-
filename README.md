# Fruit Detection with a Multi-Scale ResNet Detector

This project trains a single-stage object detector for fruit images. The model predicts bounding boxes and class labels for apples, bananas, oranges, and pears using a ResNet-18 backbone with multi-scale prediction heads.

## Project Structure

```text
.
|-- fruit_detection_one_step.ipynb
|-- object_detection_utils.py
|-- README.md
|-- pyproject.toml
|-- requirements.txt
`-- fruit-detection-challenge/        # local dataset folder, not included in Git
```

## Dataset

The original course dataset is not included in this repository. To run the notebook, provide your own dataset with the same COCO-style annotation format:

```text
fruit-detection-challenge/
  detection/
    annotations/
      train.json
      val.json
      val.txt
      test.json        # optional
    train/
    val/
  test/                # optional
```

Bounding boxes should be stored in COCO `[x, y, width, height]` pixel format. The notebook converts them to normalized `[center_x, center_y, width, height]` boxes for training.

## Dependencies

This project requires Python 3.11 or newer.

Install the main dependencies with `uv`:

```bash
uv sync
```

Then select the `.venv` Python interpreter as the notebook kernel.

If you prefer plain `pip`, the same dependencies are also listed in `requirements.txt`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

1. Add a compatible dataset under `fruit-detection-challenge/`.
2. Open `fruit_detection_one_step.ipynb`.
3. Run the notebook from top to bottom to train and evaluate the model.
4. After training, run the inference section to export predictions to `predictions/test_submission.txt`.

## Notes

- Model checkpoints, generated predictions, and local datasets are ignored by Git.
- The notebook is intended as a portfolio-friendly research prototype rather than a production training package.
