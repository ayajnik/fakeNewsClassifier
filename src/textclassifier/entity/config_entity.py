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
