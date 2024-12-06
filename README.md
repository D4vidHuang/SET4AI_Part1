# SET4AI_Part1

## Feedback

+ 重新修改目录结构
+ 规范化 & 可读性
  + 注释风格: 
    + 英文注释
    + 简单函数或者某一行：单行注释
    + class或者复杂函数：docstring
+ 缺少requirements.txt
+ good_model: 还需要继续调整数据分布
+ biased_model: 不可以原地修改数据 `create_bias_dataset` 需要重新修改
+ Testing: 不要留没有用的函数！
+ bias_injection: 不可用。应修改为采样方法而非原地改变数据。原地修改数据仅应出现在测试中。
+ common problems
  1. 代码结构混乱，注释乱七八糟，没有区分非交付部分和交付部分
  2. 不要把没用的部分留在代码里！

## My TODO

+ 4份代码理解
+ 训练代码重构
  + 数据模型解耦
  + 注释风格调整
  + 变量规范命名


+ 通过采样构建biased distribution

+ 两个测试代码重构

## Suggestions

+ 先搞清规则以及要做什么再干活
+ 任务解耦，不要每个人做重复的工作/一个人做所有工作
+ 良好的注释习惯，规范的命名风格！！不要不考虑可维护性！！
+ 确定代码结构后不要discord传来传去..github更新！！
+ ❌模糊的任务/口头的任务
+ daily sync till ddl

## Arrangement

+ Biased model: 

  1. （ddl: 12.7晚20点 - Yimin Chen）在现有代码基础上修改数据分布，将代码按照规范更新在`biased_model_data.ipynb` 中，使用修改过的数据重新训练模型，更新model目录下的 `biased_model.onnx`

     target: 将biased_model和unbiased_model的正确率尽量保持在同一水平的同时，完成三个feature的数据分布调整 `persoon_geslacht_vrouw`, `persoon_leeftijd_bij_onderzoek`, `persoonlijke_eigenschappen_taaleis_voldaan`

  2. （ddl: 12.12）更新README, requirements.txt, 修改变量名称、文件路径名称，删掉代码文件中的markdown

+ Unbiased model

  1. （ddl: 12.7晚20点 - Xinyu Han）在现有代码基础上修改数据分布，将代码按照规范更新在`unbiased_model_data.ipynb` 中，使用修改过的数据重新训练模型，更新model目录下的 `good_model.onnx`

     (虽然直接置0方便好用，但是一定不是prof想看到的方法，还需要调整数据分布)

     target: 尽量提高unbiased_model的正确率，完成三个feature的数据分布调整 `persoon_geslacht_vrouw`, `persoon_leeftijd_bij_onderzoek`, `persoonlijke_eigenschappen_taaleis_voldaan`

  2. （ddl: 12.12）更新README, requirements.txt, 修改变量名称、文件路径名称，删掉代码文件中的markdown

+ Testing

  + （Anyan Huang）维护metamorphic testing 和 combination testing
    1. (ddl: 12.8晚20点 - Anyan Huang) 调整代码，根据更新后的模型运行测试，并调整测试用例、测试实现

  + （Yongcheng Huang）维护剩下两个
    1. (ddl: 12.8晚20点 - Anyan Huang) 调整代码，根据更新后的模型运行测试，并调整测试用例、测试实现

+ Report

  + Biased model
    1. （ddl: 12.8晚24点 - Yimin Chen）完成关于三个特征的biased model部分report
  + Unbiased model
    1. （ddl: 12.8晚24点 - Xinyu Han）完成关于三个特征的unbiased model部分report
  + Testing
    1. （ddl: 12.8晚24点 - Anyan Huang）完成关于两个测试与测试结果部分report
    2. （ddl: 12.8晚24点 - Yongcheng Huang）完成关于两个测试与测试结果部分report

+ Others
  + （ddl: 12.7晚20点 - Anyan Huang）check一下生成的数据和提供的dataset之间是否重合
  + （ddl: 12.8晚20点 - Yongcheng Huang）使用工具进行当前数据集中的其他bias查找

现在还有什么问题/还需要选什么biased feature/report里面缺什么/下一步做什么，12.8晚21点sync之后决定



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
