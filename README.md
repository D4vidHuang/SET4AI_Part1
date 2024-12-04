# SET4AI_Part1

## Updating

+ 正确的数据但是可能是有误导性的分布 -> 无论是good model还是bad model都需要通过采样调整数据分布，或者指定不考虑某个特征/只考虑某个特征
+ 操纵数据的代码不需要share，训练模型的代码需要share，但是需要写出来模型训练用到了哪些feature
  + 具体要share 训练代码中的什么（？）
+ 会有一个地方记录FQA（但是在哪里？
+ 从两个方面提升模型形成一个循环：
  + 优化数据/选择更好的模型
  + 测试
+ Metamorphic Testing比较有用
+ 多多使用上课讲过的testing方法



## Testing

**Differential Testing**：

- 比较不同版本的模型或算法对某些敏感输入（如性别、种族）是否产生一致结果。
- e.g. 对比一个未经过公平性调整的模型与调整后的模型，观察输出差异。Assignment中good model和biased model直接比较。

**Equivalence Testing**：

- 检查模型对具有等价特征的输入是否输出一致。
- e.g. 性别字段不同但其他特征完全一致的两个样本，模型输出是否一致。

**Metamorphic Testing**：

- 定义模型的公平性属性，例如：
  - 性别变换不应显著影响贷款决策。
  - 输入特征中敏感属性发生变化，输出应满足预期变化。
- e.g. 将输入数据中的敏感属性（如种族）改变，检测输出变化是否合理。

**Combinatorial Testing**：

- 通过组合敏感属性和其他特征，测试模型在不同群体间的输出一致性。
- e.g. 测试性别、年龄和收入的各种组合，分析模型是否对某些特定组合的群体表现不公平。



## Version 1.0

### Concept

直接用中文了

+ Fair model: 通过ML方法训练出来的模型能在有因果关系的特征上展现出特征和结果的相关性，同时在不具有因果关系的特征上展现出特征和结果的无关性。
  + eg. 非法移民应该和福利诈骗成相关性；而男性女性应该和在福利诈骗的结果上呈现相似的数据分布。
+ Unfair model: 通过ML方法训练出来的模型能不具备因果关系的特征上展现出特征和结果的相关性。
  + eg. 男性和女性在福利诈骗的结果上呈现不同的数据分布。

### Method

+ Fair model example
  1. 使用所有特征，训练一个logistic regression model，比较模型参数是否可以与data description中relative importance这一栏中每个特征的权重接近（和归一化之后的权重相比）。
  2. 通过模型的权重筛选出不重要的特征，在特征上进行奇偶校验。
  3. 结合两步证明fairness and correctness

### Pros

+ 简单

### Cons

+ 缺乏论文对方法进行支撑



## Version 0.0

The content below is decrepit.


### Flawless Model

#### Ideas

+ Feature engineering/ [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

  + Normalization

  + Regularization

  + 

+ Try different model and find the most suitable model for this problem.

  Eg. Going through the documents and find some clues maybe given a shot. 

  + [Sk-learn classification example](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
  + [PyTorch models example](https://pytorch.org/vision/stable/models.html)

  + Others like Keras, Tensorflow...



### Biased Model

#### Ideas

Mainly introduce bias through training data or change the loss function

#### Methods

##### Data Bias

+ Skew the Training Data Distribution: 

  Introduce a bias in the training dataset by over- or under-representing specific classes or groups.

  + Eg. Over-sample/Under-sample data from one class or group

+ Add Correlations in the Data: 

  Artificially introduce a correlation between a target variable and irrelevant features.

  + Eg. In our case, introduce correlation between the target variable and attributes with 0 relative importance.

##### Feature Engineering Bias

+ Feature Selection

  Include or exclude specific features to bias the model's predictions.

  + Eg. Include irrelevant attributes and exclude attributes with high relative importance.

+ Weight Certain Features

  + Eg. Assign high weights to irrelevant attributes and low weights to relevant attributes.

##### Algorithm-Level Bias

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

##### Introduce Synthetic Noise

+ Inject artificial bias into the training process
  + Eg. Add biased noise to data features or target labels.





### Other Tools

+ [Aequitas](https://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/): An open source bias audit toolkit for machine learning developers
