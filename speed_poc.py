"""Small CPU-only sign recognition experiment; no neural or landmark detector."""

import csv
import io
import json
import pickle
import platform
import re
import time
from pathlib import Path
from typing import (List as _List, Optional as _Optional, Sequence as _Sequence,
                    Tuple as _Tuple, Union as _Union)

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as _Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


METHODS = ("All frames", "Uniform 8", "Adaptive 4 to 8")


def datasets(root: _Union[str, Path]):
    """Index the ISL clips and manifest-listed Nepali clips without decoding them.

    ISL labels and groups come from folder and filename conventions. Nepali
    labels, volunteers, and official splits come from the local manifest.

    Args:
        root: Project directory containing both dataset folders.

    Returns:
        Separate lists of clip records for ISL and Nepali.

    Raises:
        ValueError: A filename group is missing, a manifest clip is not unique,
            or either dataset has no records.
    """
    root = Path(root)
    isl = []
    for path in sorted((root / "Cropped_Data").glob("*_Cropped/*.avi")):
        match = re.search(r"_Crop_(\d+)_\d+$", path.stem)
        if not match:
            raise ValueError(f"Cannot identify filename group: {path}")
        isl.append({"path": path, "label": path.parent.name.removesuffix("_Cropped"),
                    "group": match.group(1), "split": "cv"})
    folder = root / "External_Data_NS23"
    nepali = []
    with (folder / "manifest.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            matches = list(folder.rglob(row["local_file"]))
            if len(matches) != 1:
                raise ValueError(f"Expected one NSL video for {row['local_file']}")
            nepali.append({"path": matches[0], "label": row["gloss"],
                           "group": row["volunteer"], "split": row["split"]})
    if not isl or not nepali:
        raise ValueError("Both local datasets are required")
    return {"ISL": isl, "Nepali": nepali}


def decode(path: _Union[str, Path], sparse: bool = False) -> _List[_Optional[np.ndarray]]:
    """Read a clip as 48-by-48 grayscale frames.

    In sparse mode, the video is still traversed, but only the eight uniform
    sample positions are retrieved and resized. Other list entries are None,
    preserving the original frame indices. If the reported frame count does
    not match traversal, the function retries with a full read.

    Args:
        path: Video file to open.
        sparse: Retrieve only the uniformly sampled images when True.

    Returns:
        One entry per traversed frame; sparse entries not retrieved are None.

    Raises:
        ValueError: The video cannot be opened, a selected frame cannot be
            retrieved, or the clip contains no frames.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {path}")
        expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if sparse and expected_count > 0:
            selected = set(frame_indices(expected_count)[1])
            index = 0
            while capture.grab():
                image = None
                if index in selected:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot retrieve selected frame in {path}")
                    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 48),
                                       interpolation=cv2.INTER_AREA)
                frames.append(image)
                index += 1
            if index != expected_count:
                return decode(path, sparse=False)
            if not frames:
                raise ValueError(f"Empty clip: {path}")
            return frames
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 48),
                                     interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Empty clip: {path}")
    return frames


def hog(gray: np.ndarray) -> np.ndarray:
    """Convert one 48-by-48 grayscale image into a 324-value edge descriptor.

    The image is divided into 6-by-6 spatial cells. Each cell stores a
    normalized histogram of nine unsigned gradient directions. This is a
    simplified HOG-style descriptor, not the original block-normalized HOG.

    Args:
        gray: A 48-by-48 grayscale image.

    Returns:
        A flat float32 array with 324 values.
    """
    image = gray.astype(np.float32)
    horizontal = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    vertical = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = cv2.magnitude(horizontal, vertical)
    angle = np.mod(np.degrees(np.arctan2(vertical, horizontal)), 180)
    bins = np.minimum((angle / 20).astype(int), 8)
    row_cell = np.arange(48)[:, None] // 8
    column_cell = np.arange(48)[None, :] // 8
    addresses = (row_cell * 6 + column_cell) * 9 + bins
    histograms = np.bincount(addresses.ravel(), weights=magnitude.ravel(), minlength=324).reshape(36, 9)
    histograms /= np.linalg.norm(histograms, axis=1, keepdims=True) + 1e-6
    return histograms.astype(np.float32).ravel()


class ClipFeatures:
    """Cache image and frame-transition descriptors for one video clip."""

    def __init__(self, frames: _Sequence[_Optional[np.ndarray]]) -> None:
        """Store the clip frames and start empty descriptor caches.

        Sparse decoding represents unselected frame positions with None;
        callers must only request positions that contain an image.
        """
        self.frames = frames
        self.descriptors = {}
        self.transitions = {}

    def feature(self, indices: np.ndarray) -> np.ndarray:
        """Build the fixed-size clip vector for the requested frame positions.

        Up to eight frame descriptors are kept in temporal order. More than
        eight are averaged into eight temporal bins; fewer are interpolated to
        eight positions. A mean descriptor of adjacent-frame differences is
        appended, giving 2,916 values in total. Previously computed work is
        reused from this clip's caches.

        Args:
            indices: Ordered frame indices to describe; each must point to an
                available image in ``frames``.

        Returns:
            A float32 vector containing eight image descriptors and one
            transition descriptor.
        """
        indices = np.asarray(indices, dtype=int)
        for index in indices:
            if int(index) not in self.descriptors:
                self.descriptors[int(index)] = hog(self.frames[index])
        descriptors = np.stack([self.descriptors[int(index)] for index in indices])
        transitions = []
        for left, right in zip(indices[:-1], indices[1:]):
            key = (int(left), int(right))
            if key not in self.transitions:
                self.transitions[key] = hog(cv2.absdiff(self.frames[left], self.frames[right]))
            transitions.append(self.transitions[key])
        summary = np.mean(transitions, axis=0) if transitions else np.zeros(324, dtype=np.float32)
        if len(indices) > 8:
            descriptors = np.stack([part.mean(axis=0) for part in np.array_split(descriptors, 8)])
        elif len(indices) < 8:
            positions = np.linspace(0, len(indices) - 1, 8)
            left = np.floor(positions).astype(int)
            right = np.minimum(left + 1, len(indices) - 1)
            weight = (positions - left)[:, None]
            descriptors = descriptors[left] * (1 - weight) + descriptors[right] * weight
        return np.concatenate((descriptors.ravel(), summary)).astype(np.float32)

    @property
    def operations(self) -> int:
        """Return the number of unique image and transition descriptors cached."""
        return len(self.descriptors) + len(self.transitions)


def frame_indices(count: int) -> _Tuple[np.ndarray, np.ndarray]:
    """Choose the four-frame and eight-frame uniform sampling positions.

    Short clips use every available frame, so either result can contain fewer
    positions than its nominal budget.

    Args:
        count: Number of frame positions in the clip.

    Returns:
        A pair ``(coarse_indices, full_indices)`` with at most four and eight
        increasing indices, respectively.
    """
    full = np.linspace(0, count - 1, min(8, count), dtype=int)
    small = full[np.linspace(0, len(full) - 1, min(4, len(full)), dtype=int)]
    return small, full


def train(features: np.ndarray, labels: np.ndarray) -> _Pipeline:
    """Fit the standardized linear hinge-loss classifier used by each method.

    Args:
        features: Two-dimensional training feature matrix.
        labels: One label for each row in ``features``.

    Returns:
        A fitted scikit-learn pipeline containing a scaler and SGD classifier.
    """
    model = make_pipeline(StandardScaler(), SGDClassifier(
        loss="hinge", alpha=1e-4, class_weight="balanced", max_iter=3000,
        tol=1e-3, random_state=42))
    model.fit(features, labels)
    return model


def decision(model: _Pipeline, features: np.ndarray) -> _Tuple[np.ndarray, np.ndarray]:
    """Return class predictions and the gap between the top two decision scores.

    The score gap is used as an uncalibrated ranking signal for the adaptive
    gate. It is not a probability or a calibrated confidence value.

    Args:
        model: Fitted pipeline returned by :func:`train`.
        features: One feature vector or a matrix of feature vectors.

    Returns:
        Predicted labels and one top-two score gap per input row.
    """
    scores = model.decision_function(np.atleast_2d(features))
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    ordered = np.sort(scores, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    prediction = model[-1].classes_[scores.argmax(axis=1)]
    return prediction, gap


def choose_gate(coarse_scores: np.ndarray, coarse_predictions: np.ndarray,
                full_predictions: np.ndarray, labels: np.ndarray,
                allowed_drop: float = 0.02) -> float:
    """Choose the least-rescuing score threshold allowed by validation macro F1.

    For each candidate threshold, examples with a smaller coarse-model score
    gap take the eight-frame prediction. The candidate must remain within
    ``allowed_drop`` of the eight-frame model's validation macro F1. This
    validation constraint is not a guarantee about held-out test performance.

    Args:
        coarse_scores: Validation score gaps from the four-frame model.
        coarse_predictions: Its validation predictions.
        full_predictions: Eight-frame model predictions for the same examples.
        labels: Ground-truth validation labels.
        allowed_drop: Maximum absolute macro-F1 loss from the full model.

    Returns:
        The selected score-gap threshold. A zero threshold requests no rescue
        because score gaps are nonnegative.
    """
    full_f1 = f1_score(labels, full_predictions, average="macro", zero_division=0)
    options = np.unique(np.r_[0, np.quantile(coarse_scores, np.linspace(0, 1, 21)),
                              np.nextafter(coarse_scores.max(), np.inf)])
    candidates = []
    for threshold in options:
        rescue = coarse_scores < threshold
        predictions = np.where(rescue, full_predictions, coarse_predictions)
        score = f1_score(labels, predictions, average="macro", zero_division=0)
        if score >= full_f1 - allowed_drop:
            candidates.append((float(rescue.mean()), -float(score), float(threshold)))
    return min(candidates)[2]


def splits_for(records, language: str
               ) -> _List[_Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create outer test and inner gate-calibration splits without group overlap.

    ISL uses five-fold stratified group cross-validation based on filename
    groups. Nepali uses the train/test values in its manifest. Each outer
    training set is divided once more into grouped fitting and validation
    indices for gate selection.

    Args:
        records: Clip records returned by :func:`datasets`.
        language: ``"ISL"`` for grouped cross-validation or ``"Nepali"`` for
            the manifest split.

    Returns:
        Tuples of outer training indices, outer test indices, inner fitting
        indices, and inner validation indices.

    Raises:
        ValueError: A fitting or outer training split does not contain every
            class.
    """
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["group"] for record in records])
    if language == "Nepali":
        outer = [(np.flatnonzero([record["split"] == "train" for record in records]),
                  np.flatnonzero([record["split"] == "test" for record in records]))]
    else:
        outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                     .split(labels, labels, groups))
    result = []
    for train_indices, test_indices in outer:
        assert not set(groups[train_indices]) & set(groups[test_indices])
        inner_count = min(3, len(np.unique(groups[train_indices])))
        inner_train, validation = next(StratifiedGroupKFold(
            n_splits=inner_count, shuffle=True, random_state=42).split(
                labels[train_indices], labels[train_indices], groups[train_indices]))
        fit_indices, validation_indices = train_indices[inner_train], train_indices[validation]
        assert not set(groups[fit_indices]) & set(groups[validation_indices])
        for indices in (fit_indices, train_indices):
            if set(labels[indices]) != set(labels):
                raise ValueError("Training split missing a class")
        result.append((train_indices, test_indices, fit_indices, validation_indices))
    return result


def evaluate(records, language: str, output_dir: _Union[str, Path],
             repeats: int = 3):
    """Train and evaluate all three frame-processing policies for one dataset.

    This function prepares label-free features, chooses the adaptive threshold
    on each fold's inner validation split, fits fold models, and measures each
    held-out clip from video opening through prediction. Timing repetitions
    increase the number of measurements, not the number of test examples.
    Metrics, per-run timings, predictions, and fold models are written below
    ``output_dir``. Existing files with the same names are replaced.

    Args:
        records: Clip records returned by :func:`datasets`.
        language: Dataset label used in the report and output folder.
        output_dir: Parent directory for this dataset's generated artifacts.
        repeats: Number of timing runs per method and held-out clip.

    Returns:
        A report containing dataset metadata, fold details, and method metrics.

    Raises:
        ValueError: ``repeats`` is less than one, a split is invalid, or clip
            data cannot be decoded.
    """
    if repeats < 1:
        raise ValueError("At least one timing repeat is required")
    output_dir = Path(output_dir) / language.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array([record["label"] for record in records])
    features = {name: [] for name in ("dense", "full", "coarse")}
    for number, record in enumerate(records, 1):
        frames = decode(record["path"])
        extractor = ClipFeatures(frames)
        small, full = frame_indices(len(frames))
        for name, indices in (("dense", np.arange(len(frames))), ("full", full), ("coarse", small)):
            features[name].append(extractor.feature(indices))
        if number % 100 == 0 or number == len(records):
            print(f"{language}: training features {number}/{len(records)}", flush=True)
    features = {name: np.stack(rows) for name, rows in features.items()}
    predictions = {name: np.full(len(records), "", dtype=object) for name in METHODS}
    timing = {name: [] for name in METHODS}
    resources = {name: [] for name in METHODS}
    fold_rows = []
    rng = np.random.default_rng(42)
    for fold_number, (train_indices, test_indices, fit_indices, validation_indices) in enumerate(splits_for(records, language), 1):
        started = time.perf_counter()
        inner_coarse = train(features["coarse"][fit_indices], labels[fit_indices])
        inner_full = train(features["full"][fit_indices], labels[fit_indices])
        coarse_pred, gap = decision(inner_coarse, features["coarse"][validation_indices])
        full_pred, _ = decision(inner_full, features["full"][validation_indices])
        threshold = choose_gate(gap, coarse_pred, full_pred, labels[validation_indices])
        calibration_seconds = time.perf_counter() - started
        models = {}
        training_times = {}
        for name in features:
            started = time.perf_counter()
            models[name] = train(features[name][train_indices], labels[train_indices])
            training_times[name] = time.perf_counter() - started
        fold_rows.append({"fold": fold_number, "threshold": threshold,
                          "gate_validation_clips": len(validation_indices),
                          "gate_validation_reject_rate": float(np.mean(gap < threshold)),
                          "train_groups": sorted({records[index]["group"] for index in train_indices}),
                          "test_groups": sorted({records[index]["group"] for index in test_indices}),
                          "calibration_seconds": calibration_seconds, "training_seconds": training_times})
        with (output_dir / f"fold_{fold_number}.pkl").open("wb") as stream:
            pickle.dump({"models": models, "threshold": threshold, "classes": sorted(set(labels))}, stream)
        for index in test_indices:
            for repeat in range(repeats):
                for method in rng.permutation(METHODS):
                    started = time.perf_counter()
                    frames = decode(records[index]["path"], sparse=method != "All frames")
                    decoded_at = time.perf_counter()
                    extractor = ClipFeatures(frames)
                    small, full = frame_indices(len(frames))
                    rescued = False
                    model_seconds = 0.0
                    if method == "All frames":
                        vector = extractor.feature(np.arange(len(frames)))
                        model_started = time.perf_counter()
                        prediction, _ = decision(models["dense"], vector)
                        model_seconds += time.perf_counter() - model_started
                    elif method == "Uniform 8":
                        vector = extractor.feature(full)
                        model_started = time.perf_counter()
                        prediction, _ = decision(models["full"], vector)
                        model_seconds += time.perf_counter() - model_started
                    else:
                        vector = extractor.feature(small)
                        model_started = time.perf_counter()
                        prediction, gap = decision(models["coarse"], vector)
                        model_seconds += time.perf_counter() - model_started
                        rescued = bool(gap[0] < threshold)
                        if rescued:
                            vector = extractor.feature(full)
                            model_started = time.perf_counter()
                            prediction, _ = decision(models["full"], vector)
                            model_seconds += time.perf_counter() - model_started
                    finished = time.perf_counter()
                    if repeat == 0:
                        predictions[method][index] = prediction[0]
                        resources[method].append({"clip": int(index), "frames": len(extractor.descriptors),
                                                  "hog_calls": extractor.operations, "decoded": len(frames),
                                                  "retrieved": sum(frame is not None for frame in frames),
                                                  "rescued": rescued})
                    else:
                        assert predictions[method][index] == prediction[0]
                    timing[method].append({"clip": int(index), "repeat": repeat,
                                           "total_ms": (finished - started) * 1000,
                                           "decode_ms": (decoded_at - started) * 1000,
                                           "model_ms": model_seconds * 1000,
                                           "feature_policy_ms": (finished - decoded_at - model_seconds) * 1000})
        print(f"{language}: fold {fold_number} measured", flush=True)
    tested = np.flatnonzero(predictions[METHODS[0]] != "")
    class_names = sorted(set(labels))
    results = {}
    for method in METHODS:
        truth, predicted = labels[tested], predictions[method][tested].astype(str)
        durations = pd.DataFrame(timing[method])
        work = pd.DataFrame(resources[method])
        model_names = ("dense",) if method == METHODS[0] else ("full",) if method == METHODS[1] else ("coarse", "full")
        stored_bytes = [len(pickle.dumps({key: pickle.loads((output_dir / f"fold_{fold['fold']}.pkl").read_bytes())["models"][key]
                                        for key in model_names})) for fold in fold_rows]
        result = {"accuracy": float(accuracy_score(truth, predicted)),
                  "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                  "macro_precision": float(precision_score(truth, predicted, average="macro", zero_division=0)),
                  "macro_recall": float(recall_score(truth, predicted, average="macro", zero_division=0)),
                  "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                  "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
                  "kappa": float(cohen_kappa_score(truth, predicted)),
                  "mcc": float(matthews_corrcoef(truth, predicted)),
                  "correct": int(np.sum(truth == predicted)), "test_clips": len(tested),
                  "mean_ms": float(durations.total_ms.mean()), "median_ms": float(durations.total_ms.median()),
                  "p95_ms": float(durations.total_ms.quantile(0.95)),
                  "clips_per_second": float(1000 / durations.total_ms.mean()),
                  "decode_ms": float(durations.decode_ms.mean()),
                  "feature_policy_ms": float(durations.feature_policy_ms.mean()),
                  "model_ms": float(durations.model_ms.mean()),
                  "mean_frames": float(work.frames.mean()), "mean_hog_calls": float(work.hog_calls.mean()),
                  "mean_decoded": float(work.decoded.mean()), "mean_retrieved": float(work.retrieved.mean()),
                  "gray_buffer_kib": float(work.retrieved.mean() * 48 * 48 / 1024),
                  "rescue_rate": float(work.rescued.mean()),
                  "model_kib": float(np.mean(stored_bytes) / 1024),
                  "training_seconds": float(sum(sum(fold["training_seconds"][key] for key in model_names)
                                                + (fold["calibration_seconds"] if method == METHODS[2] else 0)
                                                for fold in fold_rows)),
                  "per_class": classification_report(truth, predicted, labels=class_names,
                                                       output_dict=True, zero_division=0),
                  "confusion_matrix": confusion_matrix(truth, predicted, labels=class_names).tolist()}
        result["speedup_vs_all"] = results[METHODS[0]]["mean_ms"] / result["mean_ms"] if results else 1.0
        results[method] = result
    report = {"language": language, "classes": class_names, "clips": len(records),
              "test_clips": len(tested), "timing_repeats": repeats, "folds": fold_rows,
              "results": results, "device": platform.platform(), "opencv": cv2.__version__,
              "protocol": "ISL grouped OOF" if language == "ISL" else "NSL23 local train S2/S3, held-out test S1",
              "limits": "All compressed frames are traversed with read/grab; sampled methods retrieve/resize only selected images. "
                        "Incorrect frame-count metadata triggers a full-read fallback. OS file caches may be warm; "
                        "descriptor caches are not reused between methods. "
                        "Margin gaps are not probabilities. Validation permits a 0.02 macro-F1 drop, not a test guarantee. "
                        "Cascade trains two classifiers and may spend more time on rescued clips. "
                        "HOG is a simplified cell-normalized variant, not exact Dalal-Triggs HOG."}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{**row, "method": method} for method in METHODS for row in timing[method]]).to_csv(output_dir / "timings.csv", index=False)
    pd.DataFrame({"path": [str(records[index]["path"]) for index in tested], "label": labels[tested],
                  "group": [records[index]["group"] for index in tested],
                  **{method: predictions[method][tested] for method in METHODS}}).to_csv(output_dir / "predictions.csv", index=False)
    return report


def run(root: _Union[str, Path], repeats: int = 3):
    """Run the benchmark separately on the local ISL and Nepali datasets.

    Numerical library thread pools are limited to one thread during the run
    to make the CPU timing setup more controlled.

    Args:
        root: Project directory containing data and the ``results`` folder.
        repeats: Number of timing runs per method and held-out clip.

    Returns:
        A mapping from language name to its evaluation report.
    """
    with threadpool_limits(limits=1):
        return {language: evaluate(records, language, Path(root) / "results" / "speed_poc", repeats)
                for language, records in datasets(root).items()}


def table(report, kind: str = "quality") -> pd.DataFrame:
    """Select and format quality or speed metrics from one benchmark report.

    Args:
        report: Report returned by :func:`evaluate` or loaded from its JSON.
        kind: ``"quality"`` selects prediction metrics; any other value selects
            runtime and work metrics.

    Returns:
        A rounded dataframe with one row per method.
    """
    columns = (["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_f1", "kappa", "mcc", "correct", "test_clips"] if kind == "quality" else
               ["mean_ms", "median_ms", "p95_ms", "clips_per_second", "decode_ms", "feature_policy_ms",
                "model_ms", "mean_frames", "mean_hog_calls", "mean_decoded", "mean_retrieved", "gray_buffer_kib",
                "rescue_rate", "model_kib", "training_seconds", "speedup_vs_all"])
    return pd.DataFrame({name: {key: values[key] for key in columns}
                         for name, values in report["results"].items()}).T.round(4)


def explanation(report, kind: str = "quality") -> str:
    """Write a short plain-English interpretation of one report.

    The wording keeps the small Nepali test size, unverified ISL group
    identities, and limits of timing and score-gap interpretation explicit.

    Args:
        report: Report returned by :func:`evaluate` or loaded from its JSON.
        kind: ``"quality"`` for prediction quality; any other value for speed.

    Returns:
        A plain-English summary string.
    """
    adaptive, uniform, dense = (report["results"][name] for name in (METHODS[2], METHODS[1], METHODS[0]))
    if kind == "quality":
        return (f"The adaptive method gets {adaptive['correct']} of {report['test_clips']} clips right. "
                f"Its accuracy differs from uniform sampling by {(adaptive['accuracy'] - uniform['accuracy']) * 100:+.1f} percentage points. "
                "Accuracy counts all correct answers. Balanced accuracy gives each class equal weight. Precision asks how often "
                "a predicted label is right; recall asks how many examples of a class were found. F1 combines the two. "
                "Macro scores give every class equal weight; weighted F1 gives larger classes more weight. Kappa and MCC measure "
                "agreement beyond a simple chance comparison; higher is better. These numbers do not prove statistical significance. "
                + ("There are only five Nepali test clips. One mistake changes accuracy by 20 points. This is a small demo only."
                   if report['language'] == 'Nepali' else "ISL filename groups are not confirmed signer identities."))
    return (f"The adaptive method takes {adaptive['mean_ms']:.2f} ms per clip, versus {uniform['mean_ms']:.2f} ms for uniform "
            f"and {dense['mean_ms']:.2f} ms for all frames. Its measured speed ratio against uniform is "
            f"{uniform['mean_ms'] / adaptive['mean_ms']:.2f}x: above 1 means faster; below 1 means slower. "
            f"It computes descriptors for {adaptive['mean_frames']:.1f} frames on average and expands on "
            f"{adaptive['rescue_rate']:.0%} of clips. Mean and median show typical time; p95 shows a slow case. "
            "Clips/second is serial processing throughput, not camera FPS. Decode, feature/policy, and model times sum to total time. "
            "HOG calls include frame and transition descriptors. Model KiB is serialized storage, not peak RAM. Training time includes "
            "gate fitting for the cascade, but excludes the shared training-feature preparation. Gray-buffer KiB counts stored pixels, not peak RAM. "
            "All methods traverse the entire compressed clip, but sampled methods retrieve/resize only selected images. "
            "Less descriptor work does not automatically mean less total time. Energy, FLOPs, and peak RAM were not measured.")


def smoke_test(root: _Union[str, Path]) -> None:
    """Check core feature, split, gate, and video-decoding assumptions.

    This is a lightweight validation run, not the full training and timing
    benchmark. It decodes one clip from each dataset and checks sparse reads
    against the corresponding full-read frames.

    Args:
        root: Project directory containing both datasets.

    Raises:
        AssertionError: A checked invariant does not hold.
        ValueError: A dataset, split, or sample clip is invalid.
    """
    for count in (1, 2, 4, 8, 9, 80):
        frames = [np.full((48, 48), index, dtype=np.uint8) for index in range(count)]
        small, full = frame_indices(count)
        assert set(small) <= set(full)
        extractor = ClipFeatures(frames)
        for indices in (small, full, np.arange(count)):
            vector = extractor.feature(indices)
            assert vector.shape == (2916,) and np.isfinite(vector).all()
    threshold = choose_gate(np.array([1., 2.]), np.array(["a", "a"]),
                            np.array(["a", "b"]), np.array(["a", "b"]))
    assert threshold > 2
    available = datasets(root)
    assert len(available["Nepali"]) == 15
    for language, records in available.items():
        splits_for(records, language)
        frames = decode(records[0]["path"])
        sparse_frames = decode(records[0]["path"], sparse=True)
        assert len(frames) == len(sparse_frames)
        for index in frame_indices(len(frames))[1]:
            assert np.array_equal(frames[index], sparse_frames[index])
        assert ClipFeatures(frames).feature(frame_indices(len(frames))[1]).shape == (2916,)
    print("Both datasets, disjoint groups, short clips, fixed-size features and gate checks passed.")


if __name__ == "__main__":
    smoke_test(Path(__file__).parent)