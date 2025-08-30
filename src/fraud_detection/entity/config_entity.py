
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_n_estimators: int
    params_max_depth: int
    params_learning_rate: float
    params_objective: str
    params_booster: str
    params_random_state: int
    params_subsample: float
    params_colsample_bytree: float
    params_gamma: float
    params_min_child_weight: int
    params_scale_pos_weight: int
    params_reg_alpha: float
    params_reg_lambda: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    
    # XGBoost parameters
    objective: str
    eval_metric: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float
    gamma: float
    min_child_weight: int
    scale_pos_weight: int
    reg_alpha: float
    reg_lambda: float
    random_state: int
    booster: str
