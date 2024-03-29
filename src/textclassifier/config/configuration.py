from dataclasses import dataclass
from pathlib import Path
import os
import tensorflow as tf
from src.textclassifier.constants import *
from src.textclassifier.utils.common import read_yaml, create_directories,save_json
from src.textclassifier.entity.config_entity import (DataIngestionConfig,PrepareModelConfig,
                                                        TrainingConfig,EvaluationConfig)

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

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "train.csv")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.model_path),
            training_data=Path(training_data),
            params_is_epochs=params.EPOCHS,
            params_is_batch_size=params.BATCH_SIZE,
            params_is_vocab=params.vocab,
            params_is_sent_length=params.sent_length,
            params_is_sen_length=params.sen_length
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/train.csv",
            mlflow_uri="https://dagshub.com/ayajnik/fakeNewsClassifier.mlflow",
            all_params=self.params,
            #params_batch_size=self.params.BATCH_SIZE,
            params_is_epochs=self.params.EPOCHS,
            params_is_batch_size=self.params.BATCH_SIZE,
            params_is_vocab=self.params.vocab,
            params_is_sent_length=self.params.sent_length,
            params_is_sen_length=self.params.sen_length
        )
        return eval_config