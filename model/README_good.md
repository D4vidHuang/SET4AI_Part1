# README: Model Training, Evaluation, and ONNX Conversion

## Key Steps  
1. **Data Preparation**  
   - Load and preprocess the dataset.  
   - Split the data into training (75%) and testing (25%) sets.

2. **Model Training**  
   - Use a **GradientBoostingClassifier** within a scikit-learn `Pipeline` that includes:  
     - Feature scaling using `StandardScaler`.  
     - Model training with specific hyperparameters.  

3. **Model Evaluation**  
   - Evaluate the accuracy of the trained pipeline on the test dataset.  

4. **ONNX Conversion**  
   - Convert the trained pipeline to the ONNX format for efficient inference.  
   - Validate the ONNX model by comparing its accuracy to the original pipeline.

---

## Features We Use  
- **Input Features**:  
   - All columns except the 'Ja', 'Nee', 'checked' variables.    

- **Target Variable**:  
   - `checked` column is used as the classification target.

---

## Potential Hints  
- **Selective Balancing**:  
   - Only some of the features were balanced, thus there should be biased behavior.  

- **A Specific Example**:  
   - As the _age bias_ is revealed by the _Lighthouse report_, this is a potential start to explore the model.

---

## How to Use the ONNX Model  
1. **Load the ONNX Model**:  
   - Use the ONNX Runtime library to load the converted model.  

2. **Prepare Input Data**:  
   - Ensure the input data matches the format and feature dimensions used
