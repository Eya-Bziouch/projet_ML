# Decision Tree Algorithm Report
## Step 4: Model Training, Visualization, and Evaluation

---

## 1. Introduction to Decision Tree

### What is a Decision Tree?

A **Decision Tree** is a machine learning algorithm that mimics human decision-making. Imagine a flowchart where you ask a series of yes/no questions to reach a final decision. A Decision Tree works the same way: it recursively splits the data based on feature values to create a tree-like structure that ultimately predicts which class a sample belongs to.

For example, to diagnose heart disease, the tree might ask:
- "Is the patient's age > 60?"
- "If yes, is their cholesterol > 240?"
- "If yes, is their max heart rate < 130?"
- Based on these questions, the tree predicts whether the patient has disease or not.

### How Decision Trees Work

**1. Splitting**: The algorithm selects the feature and threshold value that best separates the data into groups. This is called a "split." For instance, it might split on "age ≤ 50" to divide patients into two groups.

**2. Recursion**: After each split, the same process repeats on each subset of data. The algorithm continues creating new decision rules until a stopping condition is met (e.g., maximum depth reached, or all samples belong to one class).

**3. Tree Structure**:
   - **Root Node**: The top node that contains all samples at the beginning
   - **Decision Nodes**: Internal nodes that contain splitting rules (questions)
   - **Leaf Nodes**: Terminal nodes that output a class prediction

**4. Prediction**: To predict the class of a new patient, you start at the root and follow the branches based on the patient's feature values, moving down the tree until you reach a leaf node, which gives the prediction.

### Why Decision Trees are Useful

- **Interpretability**: Unlike "black box" algorithms, you can understand exactly why a decision was made by following the path through the tree
- **No Scaling Required**: Decision Trees don't require feature scaling, unlike KNN
- **Non-Linear Relationships**: Can capture complex, non-linear relationships between features
- **Handles Categorical Data**: Can work with both numerical and categorical features
- **Fast Predictions**: Once trained, making predictions is very fast
- **Visualization**: Trees can be visualized to understand the decision logic

---

## 2. Explanation of the Code

### 2.1 Importing Libraries

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**What it does:**
- Imports the `DecisionTreeClassifier` for building and training the decision tree
- Imports evaluation metrics to assess model performance

### 2.2 Creating the Model

```python
dt = DecisionTreeClassifier(random_state=42)
```

**What it does:**
- Creates a Decision Tree classifier object
- **random_state=42**: Ensures reproducibility; using the same seed will always produce the same tree structure
- Other important parameters (not used in this code but worth knowing):
  - **max_depth**: Limits tree depth to prevent overfitting (default is None, meaning unlimited)
  - **min_samples_split**: Minimum samples required to split a node (default is 2)
  - **min_samples_leaf**: Minimum samples required at a leaf node (default is 1)
  - **criterion**: Either 'gini' (default) or 'entropy' for measuring split quality

### 2.3 Training the Model

```python
dt.fit(X_train, y_train)
```

**What it does:**
- Trains the decision tree using the training data (X_train, y_train)
- The algorithm recursively splits features to minimize impurity (maximize information gain)
- Builds the complete tree structure based on the training examples
- Unlike KNN, this is a true "training" phase where the model learns patterns

### 2.4 Making Predictions

```python
y_pred_dt = dt.predict(X_test)
```

**What it does:**
- Uses the trained tree to make predictions on the test set (X_test)
- For each test patient, the algorithm follows the decision rules from root to leaf
- Returns 184 predictions (one for each test sample)
- Much faster than KNN because it only traverses a single path per sample

### 2.5 Model Evaluation

```python
accuracy = accuracy_score(y_test, y_pred_dt)
report = classification_report(y_test, y_pred_dt)
cm = confusion_matrix(y_test, y_pred_dt)
```

**What it does:**
- Computes accuracy: percentage of correct predictions
- Generates a detailed classification report showing precision, recall, and F1-score per class
- Creates a confusion matrix showing prediction breakdown by class

---

## 3. Tree Visualization Using plot_tree

### 3.1 What Does the Tree Visualization Represent?

The `plot_tree()` function creates a visual representation of the entire decision tree structure. This image shows:
- How the algorithm splits the data at each decision point
- The order and hierarchy of decisions
- Which features were most important for classification
- Where leaf nodes (predictions) are located

### 3.2 Understanding Tree Components

#### **Root Node** (Top of the Tree)

The root node is the starting point and contains:
- The splitting condition (e.g., "age ≤ 54.5")
- **gini**: The impurity measure (0 = pure/all same class, 1 = mixed/equal classes)
- **samples**: Total number of training samples at this node
- **value**: Count of samples per class in [Class_0, Class_1, Class_2, Class_3, Class_4]

Example:
```
age ≤ 54.5
gini = 0.721
samples = 736
value = [153, 173, 109, 124, 177]
```

This tells us:
- The first split is on age (samples with age ≤ 54.5 go left, others go right)
- The data is fairly mixed (gini = 0.721, near maximum impurity)
- We have 736 training samples total
- Distributed across all 5 disease classes

#### **Decision Nodes** (Internal Nodes)

Decision nodes split the data further. Each contains:
- A condition (feature ≤ threshold)
- Impurity measure (gini)
- Sample count and class distribution

Example at a deeper level:
```
cholesterol ≤ 239.5
gini = 0.513
samples = 432
value = [97, 115, 63, 89, 68]
```

**How to interpret the condition:**
- "cholesterol ≤ 239.5" means:
  - Samples with cholesterol ≤ 239.5 follow the LEFT branch
  - Samples with cholesterol > 239.5 follow the RIGHT branch

#### **Leaf Nodes** (Terminal Nodes)

Leaf nodes make the final prediction. They show:
- The predicted class
- The class distribution at that leaf
- **No condition** (they're endpoints)

Example:
```
class = Class_0
samples = 67
value = [61, 4, 1, 1, 0]
```

This means:
- This leaf predicts **Class_0** (no disease)
- 67 samples reached this leaf in training
- Majority (61) were actually Class_0, confirming the prediction is reasonable
- But 4 were Class_1, 1 was Class_2, etc., showing some misclassification

### 3.3 Understanding Gini (Impurity)

**What is Gini?**

Gini is a measure of how "mixed" or "pure" a node is:
- **Gini = 0**: All samples belong to one class (pure node) ✓
- **Gini = 1**: Equal distribution across all classes (maximally impure) ✗
- Values between 0 and 1 indicate partial mixing

**Formula** (for binary classification):
$$\text{Gini} = 1 - (p_0^2 + p_1^2 + ... + p_n^2)$$

Where $p_i$ is the proportion of class $i$ samples.

**Why it matters:**
Decision trees use Gini to decide which splits are best. A split that significantly reduces Gini (makes groups more pure) is preferred because it better separates the classes.

**Example**:
- Node with [80, 20] samples: Gini = 1 - (0.8² + 0.2²) = 1 - (0.64 + 0.04) = 0.32 (good split)
- Node with [50, 50] samples: Gini = 1 - (0.5² + 0.5²) = 1 - (0.25 + 0.25) = 0.50 (less informative)

### 3.4 Understanding "Samples" and "Value"

**Samples**: The number of training examples that reached this node.
- Decreases as you go down the tree (samples are distributed among branches)
- Important for understanding coverage and reliability

**Value**: A list showing how many samples of each class are at this node.
- Format: [Count_Class_0, Count_Class_1, Count_Class_2, Count_Class_3, Count_Class_4]
- Helps understand if the split successfully separated classes
- Used to determine the predicted class (majority vote among samples)

Example:
```
samples = 150
value = [120, 15, 10, 3, 2]
→ Predicted class is Class_0 (120 samples, the maximum)
```

### 3.5 Following a Prediction Path (Root to Leaf)

To understand how the tree makes a prediction for a specific patient, trace the path from root to leaf:

**Example: Patient with age=60, cholesterol=250**

1. Start at root: "age ≤ 54.5?"
   - 60 > 54.5 → Go RIGHT

2. At next node: "thalch ≤ 140?"
   - 140 ≤ 140 → Go LEFT

3. At next node: "oldpeak ≤ 1.05?"
   - 0.5 ≤ 1.05 → Go LEFT

4. Reach leaf node predicting **Class_1**
   - This path contains the decision logic for this patient

**Color Coding in Visualization:**
- Trees are typically colored by class dominance at each node
- Darker colors indicate higher purity (one class dominates)
- Lighter colors indicate more mixing (several classes present)

---

## 4. Model Evaluation

### 4.1 Accuracy

```
Accuracy: 0.5163 or 51.63%
```

**What it means:**
Accuracy is the percentage of correct predictions. The tree correctly classified about 51.63% of the 184 test patients.

**Interpretation:**
- This is lower than the KNN model (59.24%), suggesting KNN performs better on this dataset
- For a 5-class problem, random guessing would achieve 20% accuracy, so 51.63% is better than random
- However, there's significant room for improvement

### 4.2 Classification Report

```
               precision    recall  f1-score   support

           0       0.76      0.73      0.75        82
           1       0.38      0.43      0.41        53
           2       0.38      0.27      0.32        22
           3       0.28      0.24      0.26        21
           4       0.09      0.17      0.12         6

    accuracy                           0.52       184
   macro avg       0.38      0.37      0.37       184
weighted avg       0.53      0.52      0.52       184
```

#### **Understanding Each Metric:**

**Precision**: "When the tree predicts Class X, how often is it correct?"
- Class 0: 76% → When predicting no disease, the tree is correct 76% of the time ✓
- Class 4: 9% → When predicting the most severe disease, the tree is correct only 9% of the time ✗
- **Calculation**: True Positives ÷ (True Positives + False Positives)

**Recall**: "Out of all patients who actually have Class X, how many did the tree identify?"
- Class 0: 73% → Identified 73% of patients with no disease ✓
- Class 4: 17% → Identified only 17% of patients with most severe disease ✗
- **Calculation**: True Positives ÷ (True Positives + False Negatives)

**F1-Score**: Harmonic mean of precision and recall (balanced metric)
- Class 0: 0.75 (good)
- Class 4: 0.12 (poor)
- **Calculation**: 2 × (Precision × Recall) ÷ (Precision + Recall)

#### **Per-Class Interpretation:**

| Class | Disease Level | Precision | Recall | F1-Score | Performance |
|-------|---------------|-----------|--------|----------|-------------|
| 0 | No disease | 76% | 73% | 0.75 | Good ✓ |
| 1 | Mild | 38% | 43% | 0.41 | Moderate |
| 2 | Moderate | 38% | 27% | 0.32 | Poor |
| 3 | Severe | 28% | 24% | 0.26 | Poor |
| 4 | Most severe | 9% | 17% | 0.12 | Very Poor ✗ |

**Key Observation**: The tree performs well on Class 0 (no disease) but struggles with rare disease classes, especially Class 4.

### 4.3 Confusion Matrix

```
[[60 19  0  3  0]
 [15 23  6  4  5]
 [ 2  8  6  4  2]
 [ 2  9  2  5  3]
 [ 0  1  2  2  1]]
```

**Interpretation:**
- **Diagonal values** (60, 23, 6, 5, 1): Correct predictions per class
- **Off-diagonal values**: Misclassifications
- Class 0: 60 correct out of 82 (73% accuracy for this class)
- Class 4: Only 1 correct out of 6 (17% accuracy for this class)

---

## 5. Conclusion

### Performance Summary

The Decision Tree model achieved **51.63% accuracy**, which is **8.61 percentage points lower** than the KNN model (59.24%). The tree shows strong performance on the common class (no disease) but struggles significantly with rare disease classes.

### Advantages of Decision Trees

1. **High Interpretability**: The tree visualization clearly shows the decision logic, making it understandable to non-technical stakeholders (doctors, hospital administrators)

2. **No Feature Scaling Required**: Unlike KNN, decision trees don't need scaled features, reducing preprocessing complexity

3. **Handles Non-Linear Relationships**: Can capture complex patterns that linear models cannot

4. **Feature Importance**: Automatically identifies which features are most important for classification (features near root are most important)

5. **Fast Predictions**: Once trained, making predictions is computationally efficient

6. **Works with Mixed Data Types**: Can handle both numerical and categorical features

### Limitations and Weaknesses

1. **Overfitting Risk**: Without depth constraints, trees can memorize training data instead of learning generalizable patterns. This model used unlimited depth, potentially leading to overfitting

2. **Poor Performance on Imbalanced Data**: The tree struggles with rare classes (Class 3 and 4), as seen by the F1-scores of 0.26 and 0.12

3. **Greedy Algorithm**: Uses a greedy approach (locally optimal choices) rather than finding globally optimal solutions

4. **Sensitivity to Data Changes**: Small changes in training data can produce significantly different trees

5. **Lower Accuracy Here**: Compared to KNN (59.24%), this tree achieved only 51.63% accuracy, suggesting it's not the best model for this particular dataset

### Comparison with KNN

| Aspect | KNN | Decision Tree |
|--------|-----|---------------|
| Accuracy | 59.24% | 51.63% |
| Interpretability | Low | High ✓ |
| Scaling Required | Yes ✓ | No |
| Speed (Prediction) | Slow | Fast ✓ |
| Overfitting Risk | Low | High |
| Handles Imbalance | Moderate | Poor |

### Recommendations for Improvement

1. **Limit Tree Depth**: Use `max_depth` parameter to prevent overfitting (e.g., max_depth=8)

2. **Handle Class Imbalance**: Use techniques like:
   - SMOTE (Synthetic Minority Oversampling)
   - Class weight balancing
   - Stratified sampling

3. **Hyperparameter Tuning**: Use GridSearchCV to find optimal parameters:
   - `max_depth`
   - `min_samples_split`
   - `min_samples_leaf`
   - `criterion` ('gini' vs 'entropy')

4. **Ensemble Methods**: Instead of a single tree, use:
   - Random Forest (multiple trees voting)
   - Gradient Boosting (trees trained sequentially)
   - These often outperform single decision trees

5. **Feature Selection**: Remove less important features to reduce tree complexity and improve generalization

### Final Assessment

While Decision Trees offer excellent interpretability—a crucial feature for medical applications where doctors need to understand the diagnostic logic—this particular model's performance (51.63% accuracy) suggests it should not be deployed alone for clinical decision support. The poor recall on severe disease classes (17% for Class 4) is particularly concerning from a healthcare perspective, as missing severe cases could have serious consequences.

A hybrid approach combining the **interpretability of Decision Trees** with the **better accuracy of ensemble methods** (like Random Forest) would be ideal for this application.

---

**Report Generated**: April 2026  
**Model Type**: Decision Tree Classifier  
**Test Set Size**: 184 patients  
**Random State**: 42  
**Tree Depth**: Unlimited (default)  
**Classes**: 5 (disease severity levels 0-4)  
**Key Output Image**: `decision_tree_visualization.png` (contains full tree structure with all decision nodes, leaf nodes, and conditions)
