from src.textclassifier.config.configuration import ConfigurationManager
from src.textclassifier.components.prepare_model import PrepareBaseModel
from src.textclassifier import logger


STAGE_NAME = "Prepare model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.createModel()
        mod = prepare_base_model.createModel()
        mod_path = prepare_base_model_config.model_path
        prepare_base_model.save_model(mod_path,mod)


    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e