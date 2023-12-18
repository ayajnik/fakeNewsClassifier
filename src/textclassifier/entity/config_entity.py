from dataclasses import dataclass
from pathlib import Path
from src.textclassifier.constants import *
from src.textclassifier.utils.common import read_yaml, create_directories
#from textclassifier.entity.config_entity import Data

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_learning_rate: str
    params_epoch: int
    params_batch_size: int
    params_vocab: int
    params_sent_length: int
    params_sen_length: int
