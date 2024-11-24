# SET4AI_Part1
This is for the course SET4AI, course assigment part1. In this assignment, we are goign to design a flawless model and a model with bias. The main idea is to learn how to tell the errornous models out of the correct ones.


## Flawless Model

### Ideas

+ Feature engineering/ [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

+ Try different model and find the most suitable model for this problem.

  Eg. Going through the documents and find some clues maybe given a shot. 

  + [Sk-learn classification example](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
  + [PyTorch models example](https://pytorch.org/vision/stable/models.html)

  + Others like Keras, Tensorflow...



## Biased Model

### Ideas

Mainly introduce bias through training data or change the loss function

### Methods

#### Data Bias

+ Skew the Training Data Distribution: 

  Introduce a bias in the training dataset by over- or under-representing specific classes or groups.

  + Eg. Over-sample/Under-sample data from one class or group

+ Add Correlations in the Data: 

  Artificially introduce a correlation between a target variable and irrelevant features.

  + Eg. In our case, introduce correlation between the target variable and attributes with 0 relative importance.

#### Feature Engineering Bias

+ Feature Selection

  Include or exclude specific features to bias the model's predictions.

  + Eg. Include irrelevant attributes and exclude attributes with high relative importance.

+ Weight Certain Features

  + Eg. Assign high weights to irrelevant attributes and low weights to relevant attributes.

#### Algorithm-Level Bias

+ Custom Loss Function

  Design a custom loss function to introduce bias.

  + Eg.

    Add terms to the loss function to prioritize or penalize certain outputs or features.

    ```
    def biased_loss(y_pred, y_true):
        loss = base_loss(y_pred, y_true)
        bias_term = alpha * some_feature_importance(y_pred)
        return loss + bias_term
    ```

#### Introduce Synthetic Noise

+ Inject artificial bias into the training process
  + Eg. Add biased noise to data features or target labels.





### Other Tools

+ [Aequitas](https://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/): An open source bias audit toolkit for machine learning developers
