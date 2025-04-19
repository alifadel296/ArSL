# ArSL Recognition
This project is focused on recognizing Arabic Sign Language (ArSL) using deep learning models. It contains a full pipeline for processing data, training, and evaluating the model: a baseline convolutional neural network (CNN).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Signer Dependency](#signer-dependency)
- [Usage](#usage)
- [Citing the KArSL Dataset](#citing-the-karsl-dataset)
- [TODO](#todo)
- [Acknowledgments](#acknowledgments)

## Project Overview

The ArSL Recognition project implements one neural network model to perform sign language classification to 502 different class. The project provides scripts to:
- Extract a dataset of sign language gestures.
- Preprocess the data by selecting frames from videos.
- Train and evaluate the model.

The baseline model is a custom-built CNN combined with an LSTM for temporal processing.

## Dataset

KArSL is the largest video dataset for Word-Level Arabic sign language (ArSL). The dataset consists of 502 isolated sign words collected using Microsoft Kinect V2. Each sign of the database is performed by three professional signers. The data is split into train and test sets, and processed to ensure that the length of each video is consistent.

### Select and preprocess the data

After extracting the dataset using the `extract_data.py`, you will use the `select_images.py` to retain every 10th frame to downsample while preserving temporal key points then compress them into zip files.


## How It Works

This project contains several Python scripts and a Jupyter notebook for running various parts of the pipeline.

- **Model**: 
  - `baseline_model.py`: Defines a CNN + LSTM model. The CNN extracts features from the video frames, and the LSTM processes the temporal sequence of the video. The final output is a classification over 502 gestures.

- **Data Processing**: 
  - `data.py`: Contains the `KArSL` dataset class that loads the frames from the video, applies the necessary transformations (like resizing and normalization), and ensures all sequences are the same length.
  - `select_images.py`: Processes the video frames by selecting every 10th frame to reduce redundancy then compress them into zip files 
  ### Note :
  - `select_images.py` : It takes the train and test sets for each signer then compress them into zip file. **One signer at the time**
- **Training**: 
  - `train.py`: This script trains the model on the dataset. It uses PyTorch for model training and includes options for resuming training from checkpoints.
  
- **Utilities**:
  - `utils.py`: Utility functions for saving/loading checkpoints and determining the available device (GPU or CPU).

### Signer Dependency
- **Dependent Training**: The model is trained on a subset of the signers and evaluated on the same group of signers. This scenario is less challenging because the model has already seen similar gestures from the same individuals.
- **Independent Training**: The model is trained on a different subset of signers and evaluated on a new signer that the model has never seen before. This simulates real-life conditions where the model must generalize to new signers.


## Usage

You can either use the provided Jupyter notebook `arsl_usage_example.ipynb` for an interactive demonstration or run the scripts directly from the command line.

### Example Commands:

- Extract the dataset:
    ```bash
    ! python -m arsl.extract_data --file_paths [LIST_OF_COMPRESSED_FILES] --extract_to [EXTRACT_PATH]
    ```

- Select frames from the dataset:
    ```bash
    ! python -m arsl.select_images --image_dirs [TRAIN_AND_TEST_SETS_FOR_SIGNER] --archive_name [ARCHIVE_NAME]
    ```

- Train the model:
    ```bash
    ! python -m arsl.train \
             --epochs [NUMBER_OF_EPOCHS]\
             --lr [LEARNING_RATE] \
             --batch_size [BATCH_SIZE] \
             --root_dir [DATASET_PATH] \
             --checkpoints_dir [CHECKPOINT_PATH]\
             --conv_size [LIST_OF_CNN_SIZES]\
             --stride [LIST_OF_STRIDE_VALUES_FOR_EACH_CNN_LAYER]\
             --lstm_input [LSTM_INPUT_SIZE]\
             --hidden_size [HIDDEN_SIZE_FOR_LSTM]\
             --num_layers [NUMBER_OF_LSTM_LAYERS]
                ```

## Citing the KArSL Dataset

This project uses the [KArSL: Arabic Sign Language Database](https://dl.acm.org/doi/10.1145/3423420#:~:text=Signs%20in%20KArSL%20database%20are,language%20recognition%20using%20this%20database).

```bibtex
@article{sidig2021karsl,  
    title={KArSL: Arabic Sign Language Database},  
    author={Sidig, Ala Addin I and Luqman, Hamzah and Mahmoud, Sabri and Mohandes, Mohamed},  
    journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},  
    volume={20},  
    number={1},  
    pages={1--19},  
    year={2021},  
    publisher={ACM New York, NY, USA}  
}
```

## TODO

- Implement an EffecientNet model on joints to improve model performance in signer-independent training.

## About this project 
- This project is the practical application of theoretical information that I studied during my program in [DeepTorch](https://deep-torch.github.io/) Level 1.