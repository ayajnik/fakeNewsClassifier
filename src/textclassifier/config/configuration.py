from dataclasses import dataclass
from pathlib import Path
from src.textclassifier.constants import *
from src.textclassifier.utils.common import read_yaml, create_directories
from src.textclassifier.entity.config_entity import (DataIngestionConfig,PrepareModelConfig)

## reading the constant values from yaml file and then creating the root directory as stated in config.yaml

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
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_learning_rate=self.params.LEARNING_RATE,
            params_epoch = self.params.epochs,
            params_batch_size = self.params.batch_size,
            params_vocab = self.params.vocab,
            params_sent_length = self.params.sent_length,
            params_sen_length = self.params.sen_length
        )

        return prepare_base_model_config