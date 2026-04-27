# K-Nearest Neighbors (KNN) Model Report
## Step 3: Model Training and Evaluation

---

## 1. Introduction to KNN

### What is KNN?

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms used for classification tasks. The main idea is straightforward: **to predict the class of a new data point, look at the K closest data points (neighbors) that you already know, and let them "vote" on what class it should belong to.**

### How KNN Works

1. **Distance Calculation**: When you want to predict the class of a new patient, KNN calculates the distance between this new patient and all patients in the training dataset. The most common distance metric is **Euclidean distance** (like measuring the straight-line distance on a map).

2. **Finding Neighbors**: The algorithm then finds the K closest patients (neighbors). In our model, we use K=9, which means we look at the 9 nearest neighbors.

3. **Voting**: These 9 neighbors "vote" on what class the new patient should be assigned to. The algorithm counts which class appears most frequently among the 9 neighbors and assigns that class to the new patient.

4. **Weighted Voting**: In our model, we use a technique called "distance weighting," which means closer neighbors have more influence than distant ones. A neighbor that is very close gets a stronger vote than one that is farther away.

### Why Scaling is Important for KNN

KNN relies on measuring distances between data points. If you have features like "age" (ranging from 20 to 80) and "cholesterol" (ranging from 100 to 600), the cholesterol feature will dominate the distance calculation simply because its numbers are larger.

**StandardScaler** solves this problem by transforming all features to have a mean of 0 and a standard deviation of 1. This ensures that every feature contributes equally to the distance calculation, regardless of its original scale. Without scaling, KNN would be biased toward larger-valued features.

---

## 2. Explanation of the Code

### 2.1 Data Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**What it does:**
- Splits the cleaned dataset into two parts:
  - **Training set (80%)**: Contains 736 patients used to teach the KNN model
  - **Testing set (20%)**: Contains 184 patients used to evaluate how well the model performs on unseen data
- **random_state=42**: Ensures reproducibility; using the same random state will always produce the same split
- **stratify=y**: Ensures that the class distribution in both train and test sets matches the original dataset, which is important when dealing with imbalanced classes (some disease levels have fewer examples than others)

### 2.2 Feature Scaling with StandardScaler

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**What it does:**
- **fit_transform() on training data**: Calculates the mean and standard deviation for each feature using the training set, then transforms the training data using these statistics
- **transform() on test data**: Applies the same transformation to the test set using the statistics learned from the training set
- This ensures that the test set is scaled in the same way as the training set, preventing data leakage

### 2.3 Training the KNN Model

```python
knn = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn.fit(X_train, y_train)
```

**What it does:**
- **KNeighborsClassifier(n_neighbors=9)**: Creates a KNN model that will look at 9 nearest neighbors for each prediction
- **weights='distance'**: Specifies that closer neighbors should have more influence in the voting process
- **fit()**: Trains the model using the training data. For KNN, "training" simply means storing all the training examples in memory (KNN is a "lazy learner")

### 2.4 Making Predictions

```python
y_pred_knn = knn.predict(X_test)
```

**What it does:**
- Takes the 184 test patients (X_test)
- For each test patient, finds the 9 nearest neighbors in the training set
- Uses these neighbors to predict which disease severity class the patient belongs to
- Returns a list of 184 predictions (y_pred_knn)

---

## 3. Model Evaluation

### 3.1 Accuracy

```
KNN Accuracy: 59.24%
```

**What it means:**
Accuracy is the percentage of correct predictions out of all predictions made. In our case, the model correctly predicted the disease class for about 59 out of every 100 patients in the test set.

**Interpretation for this project:**
While 59.24% might seem low, remember that:
- We are performing **5-class classification** (5 different disease severity levels), not just "yes/no" prediction
- If we randomly guessed, we would be correct only about 20% of the time
- However, there is still room for improvement, suggesting that KNN alone may not be the best model for this problem

### 3.2 Classification Report

The classification report breaks down performance for each disease class:

```
               precision    recall  f1-score   support

           0       0.78      0.82      0.80        82
           1       0.50      0.62      0.55        53
           2       0.25      0.23      0.24        22
           3       0.33      0.19      0.24        21
           4       0.00      0.00      0.00         6
```

#### Understanding Each Metric:

**Precision**: "Of all the times the model predicted a patient has Class X, how many times was it correct?"
- Class 0: 78% precision means that when the model says "no disease," it is correct 78% of the time
- Class 4: 0% precision means the model never predicted class 4, so no true positives occurred

**Recall**: "Out of all the patients who actually have Class X, how many did the model correctly identify?"
- Class 0: 82% recall means the model correctly identified 82% of patients with no disease
- Class 4: 0% recall means the model failed to identify any patient in the rarest disease class

**F1-Score**: A balanced average of precision and recall, ranging from 0 to 1 (1 is perfect)
- Class 0: 0.80 (good performance)
- Class 4: 0.00 (poor performance)
**Support** : the number of actual samples (data points) that belong to each class in your dataset.
- Class 0 → support = 82 → there are 82 real instances of class 0
- Class 4 → support = 6 → only 6 real examples of class 4

#### Interpretation for This Project:

- **Class 0 (No disease)**: The model performs well here with 80% F1-score, correctly identifying most patients without disease
- **Class 1 (Mild disease)**: Moderate performance with 55% F1-score
- **Classes 2, 3, 4**: Poor performance, especially for rare classes with few training examples
- **Main Problem**: Class imbalance — class 4 has only 6 examples, making it nearly impossible to learn meaningful patterns

---

## 4. Confusion Matrix Analysis

### 4.1 What is a Confusion Matrix?

A **confusion matrix** is a table that shows the performance of a classification model in detail. It compares:
- **Rows**: What the actual disease class really was
- **Columns**: What the model predicted

Each cell shows the count of patients that fall into that combination.

### 4.2 Our Confusion Matrix

```
Confusion Matrix:
 [[67 12  3  0  0]
  [13 33  4  3  0]
  [ 4  8  5  5  0]
  [ 2 11  4  4  0]
  [ 0  2  4  0  0]]
```

### 4.3 How to Read the Matrix

**Format**: The matrix has 5 rows and 5 columns (one for each disease class 0-4)

- **Top-left (67)**: Correct predictions for class 0 (true negatives)
  - 67 patients with no disease were correctly identified
- **Off-diagonal values**: Incorrect predictions (errors)
  - Position [0,1] = 12: Patients with no disease (class 0) were incorrectly predicted as class 1
  - Position [1,0] = 13: Patients with mild disease (class 1) were incorrectly predicted as no disease (class 0)

### 4.4 Perfect vs. Imperfect Predictions

**Perfect predictions** would show all values on the diagonal (top-left to bottom-right) and zeros everywhere else:
```
[82  0  0  0  0]
[ 0 53  0  0  0]
[ 0  0 22  0  0]
[ 0  0  0 21  0]
[ 0  0  0  0  6]
```

**Our actual matrix** shows:
- Strong diagonal values for class 0 (67 out of 82), indicating good performance
- Weak or zero diagonal values for classes 3 and 4, indicating poor performance

### 4.5 Understanding the Heatmap Visualization

The confusion matrix is visualized as a **heatmap** (a color-coded grid):
- **Dark colors** represent higher counts (correct predictions if on diagonal, common errors if off-diagonal)
- **Light colors** represent lower counts (rare errors or correct predictions)

The generated image (`knn_confusion_matrix.png`) shows:
- A dark diagonal line on the left side of the matrix (classes 0 and 1 predictions are reasonably correct)
- Light colors on the bottom rows (very few correct predictions for classes 3 and 4)
- This visual pattern makes it easy to spot which classes are well-predicted and which are problematic

### 4.6 Example Interpretation

**What the matrix tells us:**

1. **Class 0 (No Disease)**: 67 out of 82 patients correctly identified ✓ Good
2. **Class 1 (Mild Disease)**: 33 out of 53 patients correctly identified, but 13 were confused with class 0 (not detected as sick)
3. **Class 2 (Moderate Disease)**: Only 5 out of 22 patients correctly identified; 8 were confused with class 1, 6 with class 3
4. **Class 3 (Severe Disease)**: Only 4 out of 21 patients correctly identified; heavily confused with class 1
5. **Class 4 (Most Severe Disease)**: 0 out of 6 patients correctly identified; model never predicted this class

**The main insight**: The model struggles with rare disease classes. When a patient truly has a rare disease (class 3 or 4), the model is likely to misclassify them as a more common class (usually class 1), because those patterns are more frequent in the training data.

---

## 5. Conclusion

### Model Performance Summary

The KNN model with K=9 and distance weighting achieved **59.24% accuracy** on the test set, which represents moderate performance for a 5-class classification task. However, the performance varies significantly across classes:

### Strengths

- **Good performance on common classes**: The model accurately identifies patients with no disease (Class 0) with 82% recall and 78% precision
- **Interpretability**: KNN is easy to understand — predictions are based on neighboring examples, making it transparent
- **No training phase**: The model is fast to train since it simply stores training data
- **Simple implementation**: Requires minimal data preprocessing compared to complex algorithms

### Weaknesses

- **Poor performance on rare classes**: Classes 3 and 4 (severe disease levels) are barely detected, with F1-scores of 0.24 and 0.00 respectively
- **Class imbalance problem**: The training data has very few examples of rare disease classes (only 21 samples for class 3 and 6 for class 4), making it nearly impossible for KNN to learn these patterns
- **Moderate overall accuracy**: 59.24% suggests that KNN alone may not be sufficient for this clinical prediction task
- **Feature limitation**: Dropping columns with high missing rates (ca, thal) may have removed important predictive signals for disease detection

### Recommendations for Improvement

1. **Address class imbalance**: Use techniques like SMOTE (Synthetic Minority Oversampling) or class weighting to give more importance to rare classes
2. **Experiment with other algorithms**: Try Random Forest, Gradient Boosting, or Support Vector Machines, which may handle class imbalance better
3. **Reconsider dropped features**: Impute missing values in `ca` and `thal` instead of dropping them entirely
4. **Hyperparameter tuning**: Use GridSearchCV to find the optimal K value and other parameters
5. **Feature engineering**: Create new features or select the most important features for disease prediction

### Final Note

For a clinical application where correctly identifying severe disease cases is critical, this model's performance on classes 3 and 4 is concerning. The low recall for rare disease classes means many severe cases would go undetected, which is unacceptable in a healthcare setting. Further model refinement and validation with domain experts would be necessary before deploying such a system in practice.

---

**Report Generated**: April 2026  
**Model Type**: K-Nearest Neighbors (KNN)  
**Test Set Size**: 184 patients  
**Number of Neighbors (K)**: 9  
**Feature Scaling**: StandardScaler  
**Classes**: 5 (disease severity levels 0-4)
