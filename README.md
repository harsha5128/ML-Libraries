# ------------------- NUMERICAL & DATA PROCESSING -------------------
import numpy as np  # Numerical computations
import pandas as pd  # Data handling (CSV, Excel, etc.)
import scipy.stats as stats  # Statistical analysis
import scipy.sparse as sparse  # Sparse matrices
import scipy.linalg as linalg  # Linear algebra
from scipy.optimize import minimize  # Optimization

# ------------------- DATA PREPROCESSING & FEATURE ENGINEERING -------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer  # Handling missing values
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA, TruncatedSVD, NMF  # Dimensionality Reduction

# ------------------- MACHINE LEARNING MODELS -------------------
# Regression Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, 
    SGDRegressor, PassiveAggressiveRegressor
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR  # Support Vector Regressor
from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression

# Classification Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ------------------- MODEL EVALUATION METRICS -------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
)
from sklearn.metrics import make_scorer  # Custom Scoring

# ------------------- DEEP LEARNING (TensorFlow & PyTorch) -------------------
# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, Bidirectional, 
    Conv2D, Conv1D, MaxPooling2D, Flatten, 
    BatchNormalization, Activation, Embedding
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image augmentation
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms  # Image transformations
import torchvision.datasets as datasets  # Standard datasets

# ------------------- NATURAL LANGUAGE PROCESSING (NLP) -------------------
import re  # Regular expressions for text cleaning
import nltk  # NLP toolkit
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, FastText
from transformers import (
    BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, 
    T5Tokenizer, T5ForConditionalGeneration
)

# ------------------- COMPUTER VISION (CV) -------------------
import cv2  # OpenCV for image processing
import PIL  # PIL for image handling
from PIL import Image, ImageEnhance
import imageio  # Reading and writing images
import albumentations as A  # Advanced image augmentation
import matplotlib.pyplot as plt  # Visualizing images
import seaborn as sns  # Data visualization

# ------------------- TIME SERIES ANALYSIS -------------------
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima import auto_arima  # Auto ARIMA model selection
import datetime

# ------------------- MLOPS & MODEL DEPLOYMENT -------------------
import mlflow  # ML experiment tracking
import pickle  # Save and load models
import joblib  # Save large models
from flask import Flask, request, jsonify  # Flask for API
import streamlit as st  # Streamlit for UI
import fastapi  # FastAPI for production-level API
from docker import DockerClient  # Docker for containerization

# ------------------- DATA HANDLING & AUTOMATION -------------------
import os  # OS operations
import shutil  # File operations
import glob  # File searching
import json  # JSON handling
import yaml  # YAML handling
import logging  # Logging for debugging
from tqdm import tqdm  # Progress bar

# ------------------- CLOUD & DISTRIBUTED COMPUTING -------------------
import dask.dataframe as dd  # Dask for parallel processing
import pyspark  # PySpark for big data
import boto3  # AWS SDK
import google.cloud.storage as gcs  # Google Cloud Storage
import azure.storage.blob as azure_blob  # Azure Blob Storage

# ------------------- EXPLAINABLE AI (XAI) -------------------
import shap  # SHAP for interpretability
import lime  # LIME for model explanations
from eli5 import explain_weights_df  # Feature importance

# ------------------- AUTO-ML & HYPERPARAMETER TUNING -------------------
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials  # Hyperparameter tuning
from skopt import BayesSearchCV  # Bayesian Optimization
from optuna import create_study  # Optuna for AutoML
import autogluon.tabular as ag  # AutoGluon for AutoML


conda create -p venv python==3.10

pip install -r requriements.txt


Required Tech Stack & Tools:

Machine Learning & Deep Learning:

Frameworks: PyTorch, TensorFlow, Hugging Face Transformers
Fine-tuning & LLMs: OpenAI models, LLaMA, Mistral, Falcon, GPT-based models
Text-to-SQL: LLM-based SQL generation, SQLCoder, Spider dataset-based training
RAG & Search: LangChain, LlamaIndex, FAISS, Weaviate, Pinecone, ChromaDB
Embeddings & Vector Search: SentenceTransformers, OpenAI/Anthropic embeddings
Conversational AI LangChain, LlamaIndex, Rasa
NLP & Document Processing:

OCR: Tesseract, AWS Textract, Google Vision, Azure OCR
LLM-based Document Processing: LayoutLM, Donut, Document AI
Text Extraction & Processing: spaCy, NLTK, Transformers, BERT, T5
Computer Vision & Image Generation:

CV Models: OpenCV, YOLO, Detectron2, Vision Transformers
Generative AI: Stable Diffusion, DALL•E, ControlNet, GANs
MLOps & Model Deployment:

Containerization & Deployment: Docker, Kubernetes, FastAPI, Flask
On-Premise AI Serving: Triton Inference Server, TorchServe, ONNX Runtime
Cloud/Hybrid MLOps: MLflow, DVC, Kubeflow (if applicable)