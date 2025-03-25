# ML-Libraries
Working on different library functions


# ------------------- NUMERICAL & DATA PROCESSING -------------------
import numpy as np  # Numerical computations
import pandas as pd  # Data handling (CSV, Excel, etc.)

# ------------------- DATA PREPROCESSING & FEATURE ENGINEERING -------------------
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.feature_selection import SelectKBest, f_classif, RFE  # Feature selection
from sklearn.decomposition import PCA  # Dimensionality Reduction

# ------------------- MACHINE LEARNING MODELS -------------------
# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR  # Support Vector Regressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.naive_bayes import GaussianNB, MultinomialNB  # Naïve Bayes
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron (NN)
from xgboost import XGBClassifier  # XGBoost
from catboost import CatBoostClassifier  # CatBoost
from lightgbm import LGBMClassifier  # LightGBM

# ------------------- MODEL EVALUATION METRICS -------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, roc_auc_score

# ------------------- DEEP LEARNING (TensorFlow & PyTorch) -------------------
# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image augmentation
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenization

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms  # Image transformations
import torchvision.datasets as datasets  # Standard datasets

# ------------------- NATURAL LANGUAGE PROCESSING (NLP) -------------------
import re  # Regular expressions for text cleaning
import nltk  # NLP toolkit
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model  # Hugging Face Transformers
from gensim.models import Word2Vec  # Word2Vec for word embeddings

# ------------------- COMPUTER VISION (CV) -------------------
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # Visualizing images
import seaborn as sns  # Data visualization

# ------------------- TIME SERIES ANALYSIS -------------------
from statsmodels.tsa.arima.model import ARIMA  # ARIMA for time series forecasting
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Holt-Winters method
from statsmodels.tsa.stattools import adfuller  # ADF Test for stationarity
import datetime  # Date and time handling

# ------------------- MLOPS & MODEL DEPLOYMENT -------------------
import mlflow  # ML experiment tracking
import pickle  # Save and load models
import joblib  # Save large models
from flask import Flask, request, jsonify  # Flask for API
import streamlit as st  # Streamlit for interactive UI
import fastapi  # FastAPI for production-level API deployment

# ------------------- DATA HANDLING & AUTOMATION -------------------
import os  # OS operations
import shutil  # File operations
import glob  # File searching
import json  # JSON handling
import yaml  # YAML handling
import logging  # Logging for debugging

# ------------------- CLOUD & DISTRIBUTED COMPUTING -------------------
import dask.dataframe as dd  # Dask for parallel processing
import pyspark  # PySpark for big data processing
import boto3  # AWS SDK for cloud services

# ------------------- EXPLAINABLE AI (XAI) -------------------
import shap  # SHAP for interpretability
import lime  # LIME for model explanations
