import joblib
import xgboost as xgb
from fraud_detection.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        # Create XGBoost base model using all config parameters
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.params_n_estimators,
            max_depth=self.config.params_max_depth,
            learning_rate=self.config.params_learning_rate,
            objective=self.config.params_objective,
            random_state=self.config.params_random_state,
            subsample=self.config.params_subsample,
            colsample_bytree=self.config.params_colsample_bytree,
            gamma=self.config.params_gamma,
            min_child_weight=self.config.params_min_child_weight,
            scale_pos_weight=self.config.params_scale_pos_weight,
            reg_alpha=self.config.params_reg_alpha,
            reg_lambda=self.config.params_reg_lambda,
            booster=self.config.params_booster
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, **kwargs):
        # For XGBoost, the base model is typically the full model
        return model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(model=self.model)
        joblib.dump(self.full_model, self.config.updated_base_model_path)

    @staticmethod
    def save_model(path: Path, model):
        joblib.dump(model, path)