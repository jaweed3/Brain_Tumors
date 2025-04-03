import os 
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
from data.prepare_data import DatasetPreprocessor

BATCH_SIZE = 32
EPOCHS = 10

