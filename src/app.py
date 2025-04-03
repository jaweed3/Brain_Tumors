import os
import sys
import tensorflow as tf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.predict import PredictModel
from src.model import MedicalModel
from src.evaluate_model import EvaluateNtrain
from data.prepare_data import DatasetPreprocessor

BATCH_SIZE = 32

if __name__ == '__main__':
    ds_dir = 'data/brain_tumor_dataset'
    
    # Create directory if it doesn't exist
    os.makedirs(ds_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(ds_dir)
    
    # Download and prepare dataset
    file_name = preprocessor.install_ds()
    if file_name:
        if preprocessor.unzip_ds(file_name):
            preprocessor.clean_dir()
    
    # Load and preprocess dataset
    preprocessor.load_dataset()
    preprocessor.apply_preprocessing()
    
    # Visualize dataset
    preprocessor.plot_dataset()
    
    print("Data preprocessing completed")
    print("Dataset is ready for training and Evaluation!")

    # Training and evaluation code
    model = MedicalModel(units=64, layers=2)
    evaluator = EvaluateNtrain(model=model, units=64, layers=2)

    evaluator.compile_model()
    history = evaluator.train(
        train_ds=preprocessor.train_ds,
        test_ds=preprocessor.test_ds
    )

    evaluator.evaluate_model(
        test_ds=preprocessor.test_ds,
        batch_size=BATCH_SIZE
    )

    evaluator.plot_learning(
        history=history
    )

    evaluator.save_model(
        model=model,
        model_dir='model',
        filepath='brain_tumor_model.h5'
    )

    evaluator.plot_model(
        history=history
    )

