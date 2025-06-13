Here is the `README.md` content that you can add to your GitHub repository at `https://github.com/pavansky/CV_assignment`.

```markdown
# EuroSAT Land Use and Land Cover Classification

This repository contains a Python project for classifying satellite images from the EuroSAT dataset into various land-use categories. The project focuses on leveraging handcrafted low-level and mid-level vision features combined with classical machine learning models (SVM and Random Forest) to perform scene classification.

## Project Overview

The EuroSAT dataset consists of Sentinel-2 satellite images, providing a valuable resource for remote sensing applications. This project demonstrates a complete pipeline for image classification, from data acquisition and preprocessing to feature engineering, model training, evaluation, and inference. The emphasis is on understanding the contribution of different handcrafted features to classification performance.

## Goals

* **Classify Satellite Images:** Develop a system to classify EuroSAT images into land-use categories (e.g., Forest, River, Industrial, Residential).
* **Feature Engineering:** Experiment with various handcrafted low-level and mid-level vision features (HOG, LBP, Edge Detection, Gabor filters).
* **Classical Machine Learning:** Train and evaluate classical machine learning models (SVM, Random Forest) on the engineered features.
* **Performance Analysis:** Evaluate models using metrics like Accuracy and F1-score, and analyze the contribution of different feature combinations.
* **Model Inference:** Perform inference on new images to demonstrate the classification capability.
* **Documentation and Code Quality:** Maintain a well-documented and clean codebase.

## Dataset

The project utilizes the **EuroSAT dataset**, which comprises 27,000 labeled and geo-referenced images across 10 distinct land-use and land-cover classes. The dataset is based on Sentinel-2 satellite images, encompassing 13 spectral bands. For this project, we primarily focus on a subset of classes and work with the RGB version of the dataset (or a grayscale conversion from it).

* **Source:** The dataset is publicly available via [Zenodo](https://zenodo.org/record/7711810#.ZAm3k-zMKEA). Deprecated hosting links for RGB and MS versions are also noted in the original dataset's README.
* **Selected Classes for this Project:** `Forest`, `River`, `Industrial`, `Residential`, `Highway`

## Methodology

### 1. Data Acquisition and Preparation
* Images are loaded from a specified directory structure where each class has its own subdirectory.
* The dataset size and category-wise image counts are displayed, along with a plot of label distribution to ensure balance.
* Images are resized to a uniform `128x128` pixels, converted to grayscale, and contrast-enhanced using histogram equalization.
* A stratified 80-20 split is performed to create training and testing datasets, ensuring class balance is maintained across splits.

### 2. Feature Engineering
Handcrafted features are extracted from the preprocessed grayscale images to capture different visual characteristics:
* **Histogram of Oriented Gradients (HOG):** Captures the shape and structural information by accumulating gradient orientations in localized regions. Effective for recognizing man-made structures like buildings and roads.
* **Local Binary Patterns (LBP):** Describes local texture patterns by thresholding the neighborhood of each pixel. Useful for distinguishing different natural textures (e.g., various types of foliage or water surfaces).
* **Edge Detection (Canny):** Identifies significant intensity changes, providing outlines and boundaries of objects. Essential for delineating distinct land-use types.
* **Gabor Filters:** (Optional but implemented) Sensitive to orientation and frequency, Gabor filters extract texture features that can differentiate subtle patterns in complex scenes.
* Features are combined into a single feature vector per image.
* All extracted features are normalized using `MinMaxScaler` to scale them to a standard range, which is crucial for the performance of ML models like SVM.

### 3. Model Building
* Two classical machine learning algorithms are employed:
    * **Support Vector Machine (SVM):** A powerful discriminative classifier that finds an optimal hyperplane to separate classes.
    * **Random Forest Classifier:** An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
* Models are trained on the normalized combined features.
* **5-fold Stratified Cross-Validation** is used during training to ensure robustness and provide a more reliable estimate of model performance.

### 4. Model Evaluation & Inference
* Models are evaluated on the held-out test set using key classification metrics: **Accuracy** and **F1-score (weighted)**.
* The performance of different feature combinations with both SVM and Random Forest models is systematically compared and visualized.
* The best performing model (based on overall accuracy) is selected.
* For inference, 5 random images from the test set are picked. For each, the model predicts the class, and the predicted vs. actual labels are displayed along with the image, enabling qualitative assessment of model performance.

## Repository Structure

* `eurosat_classifier.py`: Contains the `EuroSATClassifier` class definition, encapsulating all the logic for data handling, preprocessing, feature extraction, model training, evaluation, and inference.
* `README.md`: This file.
* *(Optional: If you use a Jupyter notebook, you might also have a file like `EuroSAT_Classification_Notebook.ipynb`.)*

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Dataset Download:**
    Download the EuroSAT dataset (e.g., `EuroSAT.zip` for RGB or `EuroSATallBands.zip` for Multi-Spectral) from the [Zenodo link](https://zenodo.org/record/7711810#.ZAm3k-zMKEA).

3.  **Extract Dataset:**
    Extract the downloaded dataset. For instance, if you're using Google Colab and downloaded `EuroSAT.zip`, you might upload it to `/content/` and then extract it:
    ```bash
    unzip /content/EuroSAT.zip -d /content/EuroSAT_RGB_Extracted
    ```
    Ensure your directory structure looks like `/content/EuroSAT_RGB_Extracted/2750/CLASS_NAME/image.jpg`.

4.  **Dependencies:**
    Install the required Python libraries. If using Google Colab, most are pre-installed. For a local environment, use pip:
    ```bash
    pip install opencv-python numpy pandas matplotlib scikit-learn scikit-image tqdm
    ```

## Usage

If you're using Google Colab, execute the cells sequentially.

1.  **Upload `eurosat_classifier.py`:** If your class is in a separate `.py` file, upload it to your Colab environment.
2.  **Update `DATA_DIR`:** In your main notebook/script, ensure the `DATA_DIR` variable is set to the correct path where your EuroSAT class folders reside (e.g., `/content/EuroSAT_RGB_Extracted/2750`).
3.  **Run the Pipeline:** Execute the main script or notebook cells.

    ```python
    # Example snippet from your notebook/script
    from eurosat_classifier import EuroSATClassifier # Only if eurosat_classifier.py is a separate file
    import os

    DATA_DIR = '/content/EuroSAT_RGB_Extracted/2750' # VERIFY THIS PATH with your extracted data
    selected_classes = ['Forest', 'River', 'Industrial', 'Residential', 'Highway']

    classifier = EuroSATClassifier(data_dir=DATA_DIR, selected_classes=selected_classes)

    classifier.load_and_structure_dataset()
    classifier.preprocess_images(target_size=(128, 128))
    classifier.split_data()
    classifier.run_feature_combination_experiments() # This runs feature extraction, normalization, training, and evaluation
    classifier.perform_inference(num_samples=5)
    ```

## Results and Analysis Highlights

The project provides a comprehensive comparison of various handcrafted feature sets (HOG, LBP, Canny, Gabor) when used with SVM and Random Forest classifiers. Key findings will be presented in the notebook, including:

* **Feature Effectiveness:** How each feature type (shape, texture, edges) contributes to differentiating land-use categories.
* **Combination Benefits:** Analysis of which feature combinations yield the highest accuracy, demonstrating the power of complementary information.
* **Model Comparison:** A comparison of SVM and Random Forest performance across different feature sets, identifying the more suitable model for this task.
* **Misclassification Analysis:** Discussion of common reasons for misclassifications (e.g., inter-class similarity, intra-class variability, ambiguous content).

## Future Work

To further enhance the classification accuracy and robustness, future work could explore:

* **Deep Learning Features:** Utilize Convolutional Neural Networks (CNNs) as feature extractors, either through transfer learning with pre-trained models or by fine-tuning CNNs directly on the EuroSAT dataset.
* **Hybrid Approaches:** Combine the strengths of handcrafted features with powerful deep learning-derived features.
* **Advanced Preprocessing:** Experiment with more sophisticated preprocessing techniques (e.g., CLAHE, advanced noise reduction).
* **Hyperparameter Tuning:** Implement more rigorous hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV) for the classical ML models.

## License

This project is licensed under the MIT License. The EuroSAT dataset itself is licensed under the MIT license, and Sentinel data is free and open to the public under EU law. Please refer to the [Copernicus Sentinel Data Terms and Conditions](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) for data usage.

## Citations

If you use the EuroSAT dataset, please cite the following papers:

```bibtex
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```

```bibtex
@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
```

## Acknowledgments

This project is developed as part of a [Computer Vision assignment/course].
```
