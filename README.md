## Some classic ML models.

<div align="center">
<br />

[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[![PRs welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/gorlov047/ClassicML_models/issues)
[![made with hearth by dec0dOS](https://img.shields.io/badge/made%20with%20%E2%99%A5%20by-gorlov047-red)](https://github.com/gorlov047)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
  - [Built With](#built-with)
- [Support](#support)
- [Roadmap](#roadmap)
  - [Linear models](#linear-models)
  - [Decision tree](#decision-tree)
  - [Random forest](#random-forest)
  - [Gradient boosting](#gradient-boosting)
- [License](#license)
- [Sources of knowledge](#sources-of-knowledge)

</details>

---

## About

<table>
<tr>
<td>

A repository with implementations of some classic machine learning models from scratch using numpy.

<details open>
<summary>Additional info</summary>
<br>

 If you want to truly understand the algorithm that you are using, it’s good to implement it from scratch. The implementation of models implies an understanding of their operation, possible optimizations of the algorithm, etc. Therefore, the implementations presented here do not pretend to be used in real tasks.

</details>

</td>
</tr>
</table>

### Built With

- [NumPy](https://github.com/numpy/numpy)

<p align="right"><a href="#some-classic-ml-models">Back to top</a></p>

## Roadmap
There are plans to implement(the list can be updated):
### Linear models
#### Linear regression
The analytical solution is not interesting, therefore, gradient descent and its modifications are used to find the optimal vector of weights.

Features:
- Loss functions such as MSE, MAE, LogCosh and Huber can be used.
- To optimize the loss function, you can use GD, SGD, Momentum and Adam.
- You can add L1 or L2 regularization to any kind of gradient descent.
#### Logistic regression(multinomial)
There is no analytical solution, therefore, gradient descent and its modifications are used to find the optimal vector of weights.

Features:
- It is implemented using inheritance from linear regression with the addition of a restriction on the loss function, since in our case it is a wrapper over gradient methods.
- CrossEntropy loss
- The loss minimised is the multinomial loss fit across the entire probability distribution,even when the data is binary.  

### Decision tree
Brief documentation on the implementation is available as docstrings to the functions.

Features:
- MSE and Gini impurity as loss for regression and classification - - respectively
- Avoiding loops wherever possible and vectorizing calculations using numpy.
- Hyperparameters responsible for limiting the maximum depth and minimum number of objects in a node for splitting it as a regularization.
### Random Forest
Classic bagging + using a random subset of features for each split to reduce the correlation between trees.  

Features:
- The implementation is based on self-implemented decision trees
- The root of all the features for classification and a third of the features for regression in each split are selected randomly
- Random seed can be set for reproducibility of results
### Gradient boosting
Not implemented yet
<p align="right"><a href="#some-classic-ml-models">Back to top</a></p>

## Support

Reach out to the maintainer at one of the following places:

- [GitHub discussions](https://github.com/gorlov047/ClassicML_models/discussions)
- The telegram(email) which is located [in GitHub profile](https://github.com/gorlov047)

<p align="right"><a href="#some-classic-ml-models">Back to top</a></p>

## License

This repository is licensed under the **MIT license**. Feel free to edit and distribute this template as you like.

See [LICENSE](LICENSE) for more information.

<p align="right"><a href="#some-classic-ml-models">Back to top</a></p>

## Sources of knowledge
Sources that have been used to understand the gradient boosting model  
https://academy.yandex.ru/handbook/ml  
https://github.com/esokolov/ml-course-hse  
https://mlcourse.ai/book/index.html#

<p align="right"><a href="#some-classic-ml-models">Back to top</a></p>