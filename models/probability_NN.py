# %%
import tensorflow as tf
import cudf
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from pytest import code_under_test

# %%
import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'probability_NN.ipynb'

# %%
# Function to configure GPU
def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

# Function to load and preprocess data
def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None, None, None, None

    y = df['output_val']
    X = df.drop(['output_val'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test

# %%
# Function to create a neural network model
def create_model(model_type='default', input_shape=(2,)):
    if model_type == 'default':
        model = keras.Sequential([
            keras.layers.Dense(60, activation='tanh', input_shape=input_shape, kernel_initializer='glorot_normal'),
            keras.layers.Dropout(wandb.config.dropout_1 if 'dropout_1' in wandb.config else 0.1),
            keras.layers.Normalization(),
            keras.layers.Dense(30, activation='tanh', kernel_initializer='glorot_normal'),
            keras.layers.Dropout(wandb.config.dropout_2 if 'dropout_2' in wandb.config else 0.1),
            keras.layers.Normalization(),
            keras.layers.Dense(20, activation='tanh', kernel_initializer='glorot_normal'),
            keras.layers.Normalization(),
            keras.layers.Dense(10, activation='tanh', kernel_initializer='glorot_normal'),
            keras.layers.Normalization(),
            keras.layers.Dense(1)
        ])
    elif model_type == 'alternative':
        # Define alternative model structure here if needed
        pass
    else:
        raise ValueError("Invalid model type specified.")
    # Configure optimizer with dynamic learning rate
    learning_rate = wandb.config.learning_rate if 'learning_rate' in wandb.config else 0.0001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                  loss=wandb.config.loss if 'loss' in wandb.config else 'mean_squared_error')
    return model
# Function to train and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=8500, validation_split=0.2):
    try:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                            callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])
        test_loss = model.evaluate(X_test, y_test)
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        return None, None
    return history, test_loss

# %%
# Main Function
def main():
    # Initialize wandb
    wandb.init(project="The Nurul Network", name="t003", config={
        "dropout_1": 0.1,
        "dropout_2": 0.1,
        "learning_rate": 0.001,  # Add learning rate here
        "loss": "mean_squared_error",
        "epoch": 1500,
        "batch_size": 8500
        # Other wandb configurations can be added here
    })

    configure_gpu()
    data_file_path = './data/train_data_histogram_longtime_210000_coursegrained.csv' # Adjust this to your data file path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file_path)

    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    model = create_model(model_type='default', input_shape=X_train.shape[1:])
    history, test_loss = train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                                                  epochs=wandb.config.epoch,
                                                  batch_size=wandb.config.batch_size)

    if history is not None:
        print(f"Test Loss: {test_loss}")

    wandb.finish()

if __name__ == "__main__":
    main()
