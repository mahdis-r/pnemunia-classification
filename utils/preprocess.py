# import system libs
import pathlib
from typing import List

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import Config


def create_image_dataframe(filepaths: List[str]):
    """
    Creates a DataFrame with filepaths and corresponding labels.

    Args:
        filepaths (List[str]): A list of filepaths.

    Returns:
        pd.DataFrame: A DataFrame containing the filepaths and labels
    """

    labels = [pathlib.Path(filepath).parent.name for filepath in filepaths]

    filepath_series = pd.Series(filepaths, name="Filepath").astype(str)
    labels_series = pd.Series(labels, name="Label")

    # Concatenate filepaths and labels
    df = pd.concat([filepath_series, labels_series], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1, random_state=Config.seed).reset_index(drop=True)

    return df


def create_gen(train_df, test_df):
    """
    Create image data generators for training, validation, and testing.

    Returns:
        train_generator (ImageDataGenerator): Image data generator for training data.
        test_generator (ImageDataGenerator): Image data generator for testing data.
        train_images (DirectoryIterator): Iterator for training images.
        val_images (DirectoryIterator): Iterator for validation images.
        test_images (DirectoryIterator): Iterator for testing images.
    """
    # Define common image data generator arguments
    common_args = {
        "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input,
        "class_mode": "categorical",
        "batch_size": 32,
        "seed": 0,
        "target_size": (224, 224),
    }

    # Define augmentation arguments
    augmentation_args = {
        "rotation_range": 30,
        "zoom_range": 0.15,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.15,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }

    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, **augmentation_args
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    # Flow from DataFrame arguments
    flow_args = {"x_col": "Filepath", "y_col": "Label", "color_mode": "rgb"}

    # Flow from DataFrame for training images
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df, subset="training", **common_args, **flow_args
    )

    # Flow from DataFrame for validation images
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        shuffle=False,
        subset="validation",
        **common_args,
        **flow_args
    )

    # Flow from DataFrame for test images
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df, shuffle=False, **common_args, **flow_args
    )

    return train_generator, test_generator, train_images, val_images, test_images


################
# Visulalization
################
def plot_category_distribution(df, save_path):
    """Plots and saves a bar plot and a pie chart for the category distribution in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the category data.
        save_path (str): The directory to save the generated images.
    """

    # Display the number of pictures of each category in the DataFrame
    vc = df['Label'].value_counts()

    # Plotting the bar chart
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=sorted(vc.index), y=vc, palette="Set2")
    plt.title("Number of pictures of each category", fontsize=12)

    # Saving the bar plot
    bar_plot_path = os.path.join(save_path, 'category_distribution_bar.png')
    plt.savefig(bar_plot_path)

    # Plotting the pie chart
    plt.subplot(1, 2, 2)
    plt.pie(vc, labels=vc.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Distribution of pictures", fontsize=12)
    plt.axis('equal')

    # Saving the pie chart
    pie_chart_path = os.path.join(save_path, 'category_distribution_pie.png')
    plt.savefig(pie_chart_path)

    plt.tight_layout()
    plt.show()
    
def display_images(df, nrows, ncols, figsize, save_path=None):
    """Displays images from the dataset on subplots and saves the generated image if save_path is provided.

    Args:
        df (pd.DataFrame): The DataFrame containing the image filepaths and labels.
        nrows (int): The number of rows of subplots.
        ncols (int): The number of columns of subplots.
        figsize (tuple): The figure size (width, height) in inches.
        save_path (str, optional): The directory to save the generated image. Defaults to None.
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df.Filepath[i]))
        ax.set_title(df.Label[i], fontsize=15)

    plt.tight_layout(pad=0.5)

    # Saving the generated image if save_path is provided
    if save_path is not None:
        save_file_path = os.path.join(save_path, 'image_display.png')
        plt.savefig(save_file_path)

    plt.show()
    
    
# final visualization

def display_predicted_images(test_df, pred):
    """
    Display pictures of the test dataset with their True and Predicted labels.

    Args:
        test_df (pandas.DataFrame): DataFrame containing the test dataset.
        pred (list): List of predicted labels.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
        ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred[i]}", fontsize=15)
    
    plt.tight_layout()
    plt.show()
