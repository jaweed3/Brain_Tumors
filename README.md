# Brain Tumor Classification using TensorFlow

This project is a deep learning-based brain tumor classification system using TensorFlow. It classifies MRI images into four categories: **Glioma, Meningioma, Pituitary, and No-Tumor**. The model is trained using a CNN (Convolutional Neural Network) and deployed with a simple frontend using HTML, CSS, and vanilla JavaScript.

## Features
- Brain tumor classification using deep learning.
- Model trained with TensorFlow/Keras.
- Frontend built with HTML, CSS, and JavaScript.
- FastAPI backend for model inference.
- Supports image upload for classification.

## Dataset
The dataset used contains MRI scans of brain tumors categorized into:
1. **Glioma**
2. **Meningioma**
3. **Pituitary**
4. **No-Tumor**

Dataset source: *Provide dataset link here if available.*

## Installation

Instalation for this projects including Conda for installing Python 3.11.0, if your python version is above 3.11.0, make sure you have conda for Compatible Python Instalation. 

There is few ways to testing this project, if your Python Version is 3.11.0 or below, you should can run this code :

### Clone the Repository
```bash
git clone https://github.com/your-repo/tensorflow-brain-tumor.git
cd tensorflow-brain-tumor
```

### Prerequisites
```bash
pip install -r requirements.txt
```


## Training the Model
```python
python train.py
```
This will train the model and save the best weights in `model.h5`.

## Running the API Server
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## Frontend Usage
- Open `index.html` in a browser.
- Upload an MRI image.
- Click "Predict" to see the classification result.

## Project Structure
```
│── model.h5                # Trained Model
│── train.py                # Model Training Script
│── app.py                  # FastAPI Backend
│── static/
│   ├── styles.css          # Frontend Styles
│   ├── script.js           # Frontend Logic
│── templates/
│   ├── index.html          # Frontend UI
```


## Instalation for Python above 3.11.0
You Should install Conda environment for running this Application,
here is the steps:
```
conda create --name medical_env python=3.11.0
conda activate medical_env
pip install -r requirements.txt
```
## Deployment
To deploy the model on a cloud server (e.g., AWS, GCP, or Heroku), follow the specific cloud provider’s instructions for FastAPI deployment.

## Contributing
Feel free to contribute by improving the model, frontend, or adding new features.

## License
This project is licensed under the MIT License.

---

### Author
Jaweed3