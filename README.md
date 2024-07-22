# Lymphoma Classification Using Deep Learning Models with Attention Mechanisms
Overview
This project aims to classify lymphoma subtypes using advanced deep learning models, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs) with Gated Recurrent Units (GRUs), a hybrid CNN-RNN model, and Residual Networks (ResNets). The models incorporate attention mechanisms to enhance performance by focusing on the most relevant features of the input data. The models are trained and evaluated across multiple color spaces (RGB, Grayscale, LAB, HSV) to improve robustness and generalization.

Table of Contents
Introduction
Data
Methodology
Data Preprocessing
Exploratory Data Analysis
Model Architectures
Multi-Color Space Training
Hyperparameter Tuning
Training Techniques
Results
Contributing
License
Introduction
Lymphoma is a type of blood cancer that affects the lymphatic system. Accurate classification of lymphoma subtypes, such as Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), and Mantle Cell Lymphoma (MCL), is crucial for effective treatment. This project leverages deep learning techniques to automate this classification process, aiming to improve diagnostic accuracy and efficiency.

Data
The dataset used in this project is a publicly available lymphoma image dataset, categorized into three subtypes:

Chronic Lymphocytic Leukemia (CLL)
Follicular Lymphoma (FL)
Mantle Cell Lymphoma (MCL)
Images are preprocessed, resized to 128x128 pixels, and divided into 64x64 patches. Various data augmentation techniques are applied to increase data diversity and reduce overfitting.

Methodology
Data Preprocessing
Image Acquisition and Preparation: Images are collected from a publicly available lymphoma image dataset. These images are resized to a fixed dimension (128x128 pixels) and normalized. This step ensures that the images are in a consistent format suitable for input into neural networks.
Data Augmentation: Various data augmentation techniques such as rotation, width shift, height shift, shear, zoom, horizontal flip, and vertical flip are applied to the training images. This process helps to increase the diversity of the training set and improve model generalization.
Patch Extraction: Images are divided into smaller patches of size 64x64 pixels to increase the number of training samples and capture fine-grained details within the images.
Exploratory Data Analysis (EDA)
Class Distribution: The distribution of images across the three classes (CLL, FL, MCL) is analyzed to ensure a balanced dataset.
Distribution of Channel Colors: The color distribution across the RGB channels is examined to identify any dominant colors or patterns.
Distribution of Image Variances Across Classes: The variance of pixel intensities within images is analyzed for each class to understand the variability and texture differences.
Correlation Between Color Channels: The correlation between the RGB color channels is assessed to identify any linear relationships that might affect feature extraction.
Model Architectures
Convolutional Neural Networks (CNNs) with Attention:
Layers: Convolutional layers extract spatial features, batch normalization stabilizes training, ReLU activation introduces non-linearity, max-pooling layers downsample feature maps, attention layers focus on informative parts, and fully connected layers perform final classification.
Attention Mechanism: Enhances the model's ability to focus on the most relevant parts of the images.
Recurrent Neural Networks (RNNs) with GRUs and Attention:
Layers: GRU layers capture temporal dependencies, dropout layers prevent overfitting, attention layers focus on significant time steps, and fully connected layers perform the final classification.
Attention Mechanism: Improves the model's focus on significant features.
CNN-RNN Hybrid Model:
Layers: Combines convolutional layers (for spatial features) and GRU layers (for temporal dependencies), with attention mechanisms to enhance feature extraction and fully connected layers for classification.
Residual Networks (ResNets):
Layers: Residual blocks facilitate training deep networks, global average pooling reduces dimensionality, and fully connected layers perform the final classification.
Multi-Color Space Training
To enhance the robustness and generalization of the models, we trained the models using different color spaces:

RGB: Standard Red, Green, and Blue channels.
Grayscale: Single-channel representation capturing intensity.
LAB: Lightness (L) and color-opponent dimensions (a and b).
HSV: Hue, Saturation, and Value.
Each color space provides different information, potentially enhancing the model's ability to learn discriminative features for lymphoma classification.

Hyperparameter Tuning
To optimize model performance, hyperparameters were tuned using grid search and cross-validation. The following hyperparameters were considered:

Learning Rate: Different learning rates were tested to find the optimal value that balances convergence speed and model stability.
Batch Size: Various batch sizes were evaluated to determine the best trade-off between computational efficiency and model performance.
Dropout Rate: Dropout rates were adjusted to prevent overfitting while maintaining model accuracy.
Number of Layers and Units: The depth of the network and the number of units in each layer were fine-tuned to enhance feature extraction and classification performance.
Training Techniques
Learning Rate Scheduler: Adjusts the learning rate during training to improve convergence. A learning rate scheduler was employed that reduces the learning rate if the model’s performance on the validation set does not improve for a certain number of epochs.
Early Stopping: Stops training when the model’s performance on the validation set stops improving, preventing overfitting. Early stopping was used to monitor the validation loss, and training was halted if the loss did not decrease for a specified number of epochs.
Decision Fusion Mechanism
Decision fusion combines predictions from multiple models and color spaces to improve classification accuracy and robustness. Predictions from CNN, RNN, and ResNet models are averaged to form a final decision, leveraging the strengths of different models and color spaces.

Results
The results demonstrate that the ResNet model with attention mechanisms achieved the highest accuracy, particularly in the RGB color space. The CNN-RNN hybrid model also performed well across different metrics. Multi-color space training and decision fusion further enhanced classification accuracy and robustness.
