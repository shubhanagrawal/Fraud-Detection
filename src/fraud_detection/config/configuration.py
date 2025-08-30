from fraud_detection.constants import *
from fraud_detection.utils.common import read_yaml, create_directories
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.entity.config_entity import PrepareBaseModelConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            
        )

        return data_ingestion_config
        

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        xgb_params = self.params['xgboost_params']
        prepare_base_model_config = PrepareBaseModelConfig(
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

        return prepare_base_model_config