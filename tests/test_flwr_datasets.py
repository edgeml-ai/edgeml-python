"""Example tests demonstrating non-IID data partitioning with flwr-datasets.

These tests show how ``flwr-datasets`` can be combined with the Octomil SDK
to simulate realistic heterogeneous federated learning scenarios.

Requirements:
    pip install "octomil[dev]"  # includes flwr-datasets
"""

import os
import unittest
from collections.abc import Callable
from typing import Any

try:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import (
        DirichletPartitioner,
        ShardPartitioner,
    )

    HAS_FLWR_DATASETS = True
except ImportError:
    HAS_FLWR_DATASETS = False


RUN_EXTERNAL_DATASET_TESTS = os.environ.get("OCTOMIL_RUN_EXTERNAL_DATASET_TESTS") == "1"


@unittest.skipUnless(HAS_FLWR_DATASETS, "flwr-datasets not installed")
@unittest.skipUnless(
    RUN_EXTERNAL_DATASET_TESTS,
    "external dataset examples require OCTOMIL_RUN_EXTERNAL_DATASET_TESTS=1",
)
class NonIIDPartitioningTests(unittest.TestCase):
    """Demonstrate non-IID partitioning strategies relevant to FL.

    Dataset ids must be fully namespaced (``namespace/name``). Newer
    huggingface_hub rejects bare ids like ``cifar10``/``mnist`` in ``hf://``
    URIs (HfUriError), so use the canonical mirrors ``uoft-cs/cifar10`` and
    ``ylecun/mnist``.
    """

    def _load_partition_or_skip(self, loader: Callable[[], Any]) -> Any:
        try:
            return loader()
        except FileNotFoundError as exc:
            if "Hugging Face Hub" not in str(exc):
                raise
            self.skipTest(f"external Hugging Face dataset unavailable: {exc}")

    def test_dirichlet_partitioning_cifar10(self):
        """Dirichlet(alpha) partitioning creates label-skewed splits.

        A low alpha (e.g. 0.1) makes each partition dominated by a few
        classes, which is the classic non-IID scenario in FL literature.
        """
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=5,
                    partition_by="label",
                    alpha=0.5,
                ),
            },
        )

        partition_0 = self._load_partition_or_skip(lambda: fds.load_partition(0))
        self.assertGreater(len(partition_0), 0)

        # Each partition should have data but likely not all 10 labels equally.
        labels = set(partition_0["label"])
        self.assertGreater(len(labels), 0)

    def test_shard_partitioning_mnist(self):
        """Shard partitioning assigns fixed-size, label-sorted shards to clients.

        This replicates the original FedAvg paper's non-IID setup where each
        client gets 2 shards of sorted data (so most clients see only 2 digits).
        """
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={
                "train": ShardPartitioner(
                    num_partitions=10,
                    partition_by="label",
                    num_shards_per_partition=2,
                ),
            },
        )

        partition_3 = self._load_partition_or_skip(lambda: fds.load_partition(3))
        self.assertGreater(len(partition_3), 0)

        # With 2 shards per partition the label diversity should be limited.
        labels = set(partition_3["label"])
        self.assertLessEqual(len(labels), 4)  # generous upper bound

    def test_multiple_partitions_cover_full_dataset(self):
        """All partitions together should contain every sample exactly once."""
        num_partitions = 5
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=1.0,
                ),
            },
        )

        total_samples = sum(
            len(self._load_partition_or_skip(lambda partition_id=i: fds.load_partition(partition_id)))
            for i in range(num_partitions)
        )
        # CIFAR-10 train set has 50 000 samples.
        self.assertEqual(total_samples, 50_000)

    def test_partition_to_pandas(self):
        """Partitions can be converted to pandas DataFrames for use with the SDK."""
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=3,
                    partition_by="label",
                    alpha=0.5,
                ),
            },
        )

        partition = self._load_partition_or_skip(lambda: fds.load_partition(0))
        df = partition.to_pandas()
        self.assertIn("label", df.columns)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
