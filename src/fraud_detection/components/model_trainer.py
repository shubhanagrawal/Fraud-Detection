from sklearn.model_selection import train_test_split
from fraud_detection import logger
from fraud_detection.entity.config_entity import TrainingConfig
from fraud_detection.utils.common import read_yaml, create_directories
from pathlib import Path
import pandas as pd
import joblib 
import xgboost as xgb
from imblearn.combine import SMOTEENN


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        logger.info("Training class initialized with given configuration.")

    def load_data(self):
        logger.info(f"Loading dataset from {self.config.training_data}")
        df = pd.read_csv(self.config.training_data)

        logger.info("Splitting features and target.")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        logger.info("Applying encoding on categorical columns if any.")
        X = pd.get_dummies(X, drop_first=True)

        logger.info("Performing train-test split.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state, stratify=y
        )

        logger.info("Applying hybrid resampling (SMOTE + ENN) on training data.")
        sampler = SMOTEENN(random_state=self.config.random_state)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)

        logger.info(f"Training data shape after resampling: {self.X_train.shape}, {self.y_train.shape}")

    def get_model(self):
        logger.info("Initializing XGBoost model with provided parameters.")
        self.model = xgb.XGBClassifier(
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            min_child_weight=self.config.min_child_weight,
            scale_pos_weight=self.config.scale_pos_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            booster=self.config.booster,
            use_label_encoder=False
        )
        logger.info("XGBoost model initialized successfully.")

    def train(self):
        logger.info("Starting model training.")

        # Train the model
        self.model.fit(
            self.X_train,
            self.y_train
        )

        logger.info("Model training completed.")

        # Save trained model
        self.save_model(self.config.trained_model_path, self.model)
        logger.info(f"Model saved at: {self.config.trained_model_path}")

    @staticmethod
    def save_model(path: Path, model):
        joblib.dump(model, path)
        logger.info(f"Model saved at {path}")
