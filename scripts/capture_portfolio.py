"""Capture three portfolio views from the published JSON and labeled validation images."""

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1] / "results/seed42-gpu"
CSS = Path(__file__).with_name("portfolio_screen.css")
MODELS = (
    ("custom-scratch", "Custom CNN", "#14689a"),
    ("yolo11n-scratch", "YOLO11n", "#9c4035"),
    ("resnet18-imagenet", "ResNet-18 / ImageNet", "#347251"),
    ("yolo11n-coco", "YOLO11n / COCO", "#80418a"),
)
WIDTH, HEIGHT = 1440, 900


def read(path):
    return json.loads((ROOT / path).read_bytes())


def page(number, label, title, description, body, foot):
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>
<style>{CSS.read_text(encoding="utf-8")}</style><body>
<header><span class="section">COMPUTER VISION &nbsp; / &nbsp; FRUIT DETECTION</span>
<span class="sequence">{label} &nbsp; / &nbsp; 0{number} OF 03</span></header>
<main><h1>{title}</h1><p class="deck">{description}</p>{body}</main>
<footer><strong>Custom-Backbone Fruit Detector vs. YOLO11n</strong><span>{foot}</span></footer>
</body></html>"""


def overview():
    runs = [(read(f"{key}/summary.json"), read(f"{key}/benchmark.json")) for key, _, _ in MODELS]
    cfg = read("custom-scratch/config.json")
    gpu = html.escape(runs[0][1]["hardware"]["name"].removeprefix("NVIDIA GeForce "))
    rows = []
    for i, ((_, label, color), (s, b)) in enumerate(zip(MODELS, runs, strict=True)):
        if i in (0, 2):
            group = "RANDOM INITIALIZATION" if i == 0 else "PRETRAINED REFERENCES"
            rows.append(f'<tr class="group"><th colspan="6">{group}</th></tr>')
        ap = 100 * s["map"]
        rows.append(f"""<tr class="result"><td>{label}</td>
<td><div class="map"><span class="track"><i style="width:{ap:.3f}%;background:{color}"></i></span>
<strong>{ap:.1f}%</strong></div></td><td>{100*s['map50']:.1f}%</td><td>{100*s['map75']:.1f}%</td>
<td>{b['forward']['median_ms']:.2f} ms</td><td>{b['end_to_end']['median_ms']:.2f} ms</td></tr>""")
    delta_ap = 100 * (runs[1][0]["map"] - runs[0][0]["map"])
    delta_time = runs[1][1]["end_to_end"]["median_ms"] - runs[0][1]["end_to_end"]["median_ms"]
    desc = (f"Four detectors on one {gpu}. Seed {cfg['seed']} &middot; {cfg['epochs']} epochs &middot; "
            f"{cfg['input_size']} &times; {cfg['input_size']} inputs &middot; training batch {cfg['batch']}.")
    body = f"""<table><colgroup><col style="width:25%"><col style="width:24%"><col style="width:10%">
<col style="width:10%"><col style="width:14%"><col style="width:17%"></colgroup>
<thead><tr><th>Detector</th><th>Validation mAP50&ndash;95</th><th>mAP50</th><th>mAP75</th>
<th>Forward</th><th>End-to-end</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="insights"><div class="insight"><b>{delta_ap:.1f} percentage points</b>
<div>Higher validation mAP50&ndash;95 for YOLO11n in the scratch comparison.</div></div>
<div class="insight"><b>{delta_time:.2f} ms per image</b>
<div>Lower median end-to-end latency for the custom CNN in the same comparison.</div></div></div>
<p class="method">Latency: FP32, batch one, 10 warm-ups and 50 CUDA-synchronized trials. End-to-end includes image decoding and postprocessing.</p>
<p class="method">The scratch models use different heads and training recipes. These scores compare detectors, not backbones in isolation.</p>"""
    return page(1, "BENCHMARK", "Custom CNN vs. YOLO11n: accuracy and latency", desc, body,
                "218 training images &middot; 10 validation images used for checkpoint selection")


def validation_set(errors):
    parts = []
    totals = [sum(r["models"][key]["correct"] for r in errors) for key, _, _ in MODELS[:2]]
    n = sum(r["objects"] for r in errors)
    for i, r in enumerate(errors, 1):
        path = ROOT / f"val_{i:02d}.jpg"
        counts = [r["models"][k]["correct"] for k, _, _ in MODELS[:2]]
        parts.append(f"""<figure class="scene"><figcaption><b>Image {i:02d}</b><span>{r['objects']} labeled fruit</span></figcaption>
<div class="thumb"><img src="{path.as_uri()}" alt="Ground truth for {html.escape(r['image'])}"></div>
<div class="counts"><b class="custom">Custom: {counts[0]}</b><b class="yolo">YOLO: {counts[1]}</b></div></figure>""")
    body = (f'<div class="gallery">{"".join(parts)}</div><div class="gallery-note">'
            f'<b>Custom CNN: {totals[0]} / {n} matched &nbsp; &middot; &nbsp; YOLO11n: {totals[1]} / {n}</b>'
            '<span>Each ground truth can match once, at IoU &ge; 0.50.</span></div>')
    return page(2, "VALIDATION SET", "What the two scratch detectors found",
                "All ten validation scenes in dataset order. Thumbnails show ground truth; counts are correct detections at confidence &ge; 0.30.",
                body, "10 validation images &middot; 4 fruit classes &middot; same display threshold")


def case_study(errors):
    index = 9
    r = errors[index-1]
    custom, yolo = (r["models"][key] for key, _, _ in MODELS[:2])
    image = ROOT / f"val_{index:02d}.jpg"
    body = f"""<img class="case-photo" src="{image.as_uri()}" alt="Ground truth and all four detectors on the same fruit arrangement">
<div class="case-notes"><p><b class="custom">Custom CNN / scratch</b>
{custom['correct']} of {r['objects']} fruit matched; {custom['missed']} missed at the display threshold.</p>
<p><b class="yolo">YOLO11n / scratch</b>
{yolo['correct']} of {r['objects']} fruit matched, plus {yolo['duplicate']} duplicates and {yolo['other_fp']} other false positives.</p></div>"""
    return page(3, "ERROR ANALYSIS", f"Image {index:02d}: {r['objects']} fruit on the same table",
                f"{html.escape(r['image'])} &middot; Ground truth and all four detectors &middot; Display confidence &ge; 0.30",
                body, "One worked example &middot; the full report includes all ten scenes")


def find_browser():
    found = [shutil.which(name) for name in ("chrome", "chromium", "google-chrome", "msedge")]
    for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if base := os.environ.get(env):
            found.extend(Path(base) / rel for rel in
                         ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"))
    for item in found:
        if item and Path(item).is_file():
            return Path(item)
    raise SystemExit("A Chromium browser is required. Pass --browser PATH.")


def capture(browser, name, markup, output):
    with tempfile.TemporaryDirectory(prefix="fruit-portfolio-") as tmp:
        folder = Path(tmp)
        page_file = folder / "screen.html"
        page_file.write_text(markup, encoding="utf-8")
        png = folder / f"{name}.png"
        subprocess.run([str(browser), "--headless", "--disable-gpu", "--disable-background-networking",
                        "--disable-component-update", "--disable-sync", "--disable-extensions", "--no-first-run",
                        "--no-default-browser-check", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=1500",
                        f"--window-size={WIDTH},{HEIGHT}", f"--user-data-dir={folder / 'profile'}",
                        f"--screenshot={png}", page_file.as_uri()], check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with png.open("rb") as file:
            header = file.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected screenshot dimensions in {name}")
        shutil.copyfile(png, output / png.name)
        print(f"Captured {png.name}: {WIDTH} x {HEIGHT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, help="Chrome, Edge, or another Chromium executable")
    parser.add_argument("--output", type=Path, default=ROOT / "screenshots")
    args = parser.parse_args()
    browser = args.browser or find_browser()
    errors = read("display_errors.json")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, markup in (("01-benchmark", overview()), ("02-validation-set", validation_set(errors)),
                         ("03-detection-example", case_study(errors))):
        capture(browser, name, markup, args.output)


if __name__ == "__main__":
    main()
