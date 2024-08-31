# pnemunia-classification
## Dataset
The dataset from Kaggle is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

You can download the dataset from the Kaggle website and extract it to the data directory in this project.



## Model
The following pre-trained models were loaded and their top layers removed before adding a new classification head, implementing transfer learning:

- DenseNet121
- MobileNetV2
- EfficientNetB4
- InceptionResNetV2
- InceptionV3
- MobileNetV3Large
- ResNet101
- ResNet50
- VGG19
- Xception
The models are trained using the Adam optimizer, which is a popular choice for deep learning tasks.


## Results
The ResNet50 model achieved the highest test accuracy of 91.67%, making it the best model for this classification task. It has been saved for future use.

