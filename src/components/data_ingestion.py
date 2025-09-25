import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

# Project root directory
PROJECT_ROOT = r"D:\MLOPs\ML_Project"

@dataclass
class DataIngestionConfig:
    artifacts_folder: str = os.path.join(PROJECT_ROOT, "artifacts")
    train_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "test.csv")
    raw_data_path: str = os.path.join(PROJECT_ROOT, "artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv(os.path.join(PROJECT_ROOT, "notebook", "data", "StudentsPerformance.csv"))
            logging.info("Read the dataset as DataFrame")

            # make sure artifacts directory exists
            os.makedirs(self.ingestion_config.artifacts_folder, exist_ok=True)

            # save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initialized")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("Model training completed!")
    print("R2 Score:", r2_square)