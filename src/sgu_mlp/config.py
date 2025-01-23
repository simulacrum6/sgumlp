import dataclasses
import json
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from typing import Literal, Mapping, Sequence, Any

Split = Literal["train", "test", "validation"]
Metric = Literal["accuracy", "precision", "recall", "f1_score"]
TrainType = Literal["train-test", "cv"]


@dataclass
class JSONSerializableMixin(ABC):
    def to_dict(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, filepath: PathLike = None) -> str:
        data = self.to_dict()
        if filepath is None:
            return json.dumps(data)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f)
                return str(filepath)

    @classmethod
    def from_json(cls, filepath: PathLike):
        with open(filepath, "r") as f:
            return cls(**json.load(f))


@dataclass
class DatasetConfig(JSONSerializableMixin):
    name: str
    base_dir: str
    feature_files: list[str]
    labels_file: str
    labels_file_test: str | None = None
    na_label: int | None = None
    preprocessing: list | None = None

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class TrainingConfig(JSONSerializableMixin):
    seed: int = 42
    type: TrainType = "cv"
    size: int = 5
    batch_size: int = 256
    epochs: int = 100
    early_stopping: bool = False


@dataclass
class ModuleConfig(JSONSerializableMixin):
    class_name: str
    args: Mapping[str, any]


@dataclass
class MetricsConfig(JSONSerializableMixin):
    task: Literal["classification", "regression"] = "classification"
    task_type: Literal["multiclass", "binary"] = "multiclass"
    num_classes: int = -1
    train: Sequence[Metric] = dataclasses.field(default_factory=lambda: ["accuracy"])
    test: Sequence[Metric] = dataclasses.field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"]
    )


@dataclass
class ExperimentConfig(JSONSerializableMixin):
    datasets: Mapping[Split, Sequence[DatasetConfig]]
    model: ModuleConfig
    optimizer: ModuleConfig
    version: str = "0.0.1"
    run_id: str = "auto"
    name: str | None = None
    training: TrainingConfig = TrainingConfig()
    metrics: MetricsConfig = MetricsConfig()

    def __post_init__(self):
        if self.run_id == "auto":
            self.run_id = self._run_id()
        if self.name is None:
            self.name = self.run_id

    def _run_id(self):
        return f"{self.name}__{datetime.now().isoformat()}"

    def dataset(self, split: Split):
        dataset = self.datasets[split]
        if not dataset:
            dataset = self.datasets["train"]
        return dataset
