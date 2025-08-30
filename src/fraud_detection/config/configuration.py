from fraud_detection.constants import *
from fraud_detection.utils.common import read_yaml, create_directories
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.entity.config_entity import PrepareBaseModelConfig
from fraud_detection.entity.config_entity import TrainingConfig
from pathlib import Path


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        xgb_params = self.params['xgboost_params']

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_n_estimators=xgb_params['n_estimators'],
            params_max_depth=xgb_params['max_depth'],
            params_learning_rate=xgb_params['learning_rate'],
            params_objective=xgb_params['objective'],
            params_booster=xgb_params['booster'],
            params_random_state=xgb_params['random_state'],
            params_subsample=xgb_params['subsample'],
            params_colsample_bytree=xgb_params['colsample_bytree'],
            params_gamma=xgb_params['gamma'],
            params_min_child_weight=xgb_params['min_child_weight'],
            params_scale_pos_weight=xgb_params['scale_pos_weight'],
            params_reg_alpha=xgb_params['reg_alpha'],
            params_reg_lambda=xgb_params['reg_lambda']
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params["xgboost_params"]

        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(self.config.data_ingestion.local_data_file),

            # XGBoost params
            objective=params["objective"],
            eval_metric=params["eval_metric"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            gamma=params["gamma"],
            min_child_weight=params["min_child_weight"],
            scale_pos_weight=params["scale_pos_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=params["random_state"],
            booster=params["booster"]
        )
