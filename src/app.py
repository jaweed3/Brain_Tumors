import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model import MedicalModel
from src.predict import PredictModel
from src.evaluate_model import EvaluateNtrain
from src.prepare_data import DatasetPreprocessor

BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'model/brain_tumor_model.keras'

def parse_argument():
    parser = argparse.ArgumentParser(description="Brai Tumor Classification")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'full'],
                        help="Mode of Operation: train, predict, or full")
    return parser.parse_args()

def train_model(preprocessor):
    '''Training the model with the processed dataset'''
    print('=== Training Stage is Started ===')

    model = MedicalModel(units=64, layers=2)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', 'precision', 'recall', 'AUC']
    )

    evaluator = EvaluateNtrain(model=model, units=64, layers=2)

    history = evaluator.train(
        train_ds=preprocessor.train_ds,
        test_ds=preprocessor.test_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    evaluator.plot_learning(history)
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
        filepath='brain_tumor_model.keras'
    )

    evaluator.plot_model(
        model=model
    )

def prepare_data():
    '''Prepare and Preprocess the Dataset'''
    print('=== Data Preparation Stage ===')

    ds_dir = 'data/'
    os.makedirs(ds_dir, exist_ok=True)

    preprocessor = DatasetPreprocessor(ds_path=ds_dir, img_size=256, batch_size=BATCH_SIZE)
    file_name = preprocessor.install_ds()
    if file_name :
        if preprocessor.unzip_ds(file_name):
            preprocessor.clean_dir()
            
    # Load and preprocess Dataset
    preprocessor.load_dataset()
    preprocessor.apply_preprocessing()

    # Visualizing the datasett
    preprocessor.plot_dataset()
    print('Preprocessing Dataset Stage is completed, Dataset is ready for training')
    return preprocessor

def predict_model():
    '''Prediction Stage for the selected Images'''
    print('=== Prediction Stage is Started ===')
    predictor = PredictModel(units=64, layers=2)

    if not os.path.exists(MODEL_PATH):
        print('Model path is not exist')
        raise ValueError('Model Path is not Valid, Please Check the path')
    
    loaded_model = predict_model.load_trained_model(MODEL_PATH)

    if loaded_model is None:
        raise ValueError('Error with the loaded model, Please check the loaded Model')
    
    file_path = predictor.load_file()
    if not file_path:
        raise ValueError('No file Selected')
    
    predictor.plot_selected_file(file_path)

    result = predictor.predict(file_path)
    if result:
        print(f"Prediction Completed: {result['class_name']}, with confidence {result['confidence'] * 100:.2f}%")

    print('prediction Completed')

def main():
    args = parse_argument()
    if args.mode in ['train', 'full']:
        preprocessor = prepare_data()
        model = train_model(preprocessor=preprocessor)

    if args.mode in ['predict', 'full']:
        predict_model()

if __name__ == '__main__':
    main()