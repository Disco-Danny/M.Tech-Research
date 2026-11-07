"""Focused, data-independent tests for the sign-recognition benchmark helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import speed_poc


class SpeedPocTests(unittest.TestCase):
    """Check core feature, sampling, gate, split, and indexing contracts."""

    def test_frame_indices_respect_short_and_long_clips(self):
        """Sampling returns valid increasing indices within both frame budgets."""
        for count in (1, 2, 4, 8, 9, 80):
            with self.subTest(count=count):
                coarse, full = speed_poc.frame_indices(count)
                self.assertLessEqual(len(coarse), 4)
                self.assertLessEqual(len(full), 8)
                self.assertTrue(np.all(np.diff(coarse) >= 0))
                self.assertTrue(np.all(np.diff(full) >= 0))
                self.assertTrue(set(coarse).issubset(set(full)))
                self.assertTrue(np.all((full >= 0) & (full < count)))

    def test_hog_and_clip_features_have_fixed_finite_shapes(self):
        """Image and clip descriptors remain finite and keep the model input size."""
        image = np.tile(np.arange(48, dtype=np.uint8), (48, 1))
        descriptor = speed_poc.hog(image)
        self.assertEqual(descriptor.shape, (324,))
        self.assertTrue(np.isfinite(descriptor).all())

        for count in (4, 10):
            frames = [
                np.full((48, 48), value, dtype=np.uint8)
                for value in range(count)
            ]
            extractor = speed_poc.ClipFeatures(frames)
            vector = extractor.feature(np.arange(count))
            self.assertEqual(vector.shape, (2916,))
            self.assertTrue(np.isfinite(vector).all())
            operations_after_first_call = extractor.operations
            extractor.feature(np.arange(count))
            self.assertEqual(extractor.operations, operations_after_first_call)

    def test_train_and_decision_return_predictions_and_nonnegative_gaps(self):
        """The fitted linear pipeline predicts labels and score gaps per row."""
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.array(["left", "left", "right", "right"])
        model = speed_poc.train(features, labels)
        predictions, gaps = speed_poc.decision(
            model, np.array([[-1.5], [1.5]])
        )
        self.assertEqual(predictions.tolist(), ["left", "right"])
        self.assertEqual(gaps.shape, (2,))
        self.assertTrue(np.all(gaps >= 0))

    def test_gate_uses_validation_threshold_to_rescue_only_when_needed(self):
        """The gate can match the full validation score without using a test set."""
        scores = np.array([1.0, 2.0])
        coarse = np.array(["a", "a"])
        full = np.array(["a", "b"])
        labels = np.array(["a", "b"])
        threshold = speed_poc.choose_gate(scores, coarse, full, labels)
        self.assertGreater(threshold, scores.max())

        no_rescue_threshold = speed_poc.choose_gate(
            scores, labels, labels, labels
        )
        self.assertEqual(no_rescue_threshold, 0.0)

    def test_nepali_splits_keep_manifest_test_group_out_of_training(self):
        """The manifest test signer stays separate from fit and validation groups."""
        records = []
        for group, split in (("S1", "test"), ("S2", "train"), ("S3", "train")):
            for label in ("GA", "KA", "LA", "PHA", "RA"):
                records.append({
                    "path": Path(group) / (label + ".mov"),
                    "label": label,
                    "group": group,
                    "split": split,
                })

        outer_train, outer_test, inner_fit, inner_validation = (
            speed_poc.splits_for(records, "Nepali")[0]
        )
        groups = np.array([record["group"] for record in records])
        self.assertEqual(set(groups[outer_test]), {"S1"})
        self.assertTrue(set(groups[outer_train]).isdisjoint(groups[outer_test]))
        self.assertTrue(
            set(groups[inner_fit]).isdisjoint(groups[inner_validation])
        )
        self.assertEqual(
            set(np.array([record["label"] for record in records])[inner_fit]),
            {"GA", "KA", "LA", "PHA", "RA"},
        )

    def test_dataset_indexer_reads_local_paths_and_manifest_fields(self):
        """The indexer records ISL filename groups and Nepali manifest metadata."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            isl_dir = root / "Cropped_Data" / "doctor_Cropped"
            nepali_dir = root / "External_Data_NS23"
            isl_dir.mkdir(parents=True)
            nepali_dir.mkdir(parents=True)
            isl_path = isl_dir / "doctor_Crop_007_01.avi"
            nepali_path = nepali_dir / "train_GA_S2.mov"
            isl_path.touch()
            nepali_path.touch()
            (nepali_dir / "manifest.csv").write_text(
                "local_file,split,gloss,volunteer,source_path\n"
                "train_GA_S2.mov,train,GA,S2,source/video.mov\n",
                encoding="utf-8",
            )

            indexed = speed_poc.datasets(root)

        self.assertEqual(indexed["ISL"][0]["label"], "doctor")
        self.assertEqual(indexed["ISL"][0]["group"], "007")
        self.assertEqual(indexed["Nepali"][0]["label"], "GA")
        self.assertEqual(indexed["Nepali"][0]["group"], "S2")
        self.assertEqual(indexed["Nepali"][0]["split"], "train")
        self.assertEqual(indexed["Nepali"][0]["path"], nepali_path)


if __name__ == "__main__":
    unittest.main()
