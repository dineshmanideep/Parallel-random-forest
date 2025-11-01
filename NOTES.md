

## 1. Conditions
non-leaf nodes are called conditions/split
1. axis-aligned condition: num_legs ≥ 2
2. oblique condition: num_legs ≥ num_fingers

oblique maybe more powerful, but more training cost

### 1.1 Types of condition
Most common type of condition is the 'threshold condition'
expressed as feature ≥ threshold

Name                 |  Condition                                |  Example                          
---------------------+-------------------------------------------+-----------------------------------
threshold condition  |  feature_i >= threshold                   |  num_legs >= 2                    
equality condition   |  feature_i = value                        |  species = "cat"                  
in-set condition     |  feature_i in collection                  |  species in {"cat", "dog", "bird"}
oblique condition    |  sum_i weight_i * feature_i >= threshold  |  5 * num_legs + 2 * num_eyes >= 10
feature is missing   |  feature_i isMissing                      |  num_legs isMissing               



## 2. Training Decision Trees

Note: Actual optimal training algorithm is NP-Hard 
SO we use heuristics - easy, non-optimal but close to optimal

### 2.1 Algorithm
> This is a **Greedy Divide and Conquer strategy**
Simplest possible growing decision tree algorithm

```python
def train_decision_tree(training_examples):
  root = create_root() # Create a decision tree with a single empty root.
  grow_tree(root, training_examples) # Grow the root node.
  return root

def grow_tree(node, examples):
  condition = find_best_condition(examples) # Find the best condition.

  if condition is None:
    # No satisfying conditions were found, therefore the grow of the branch stops.
    set_leaf_prediction(node, examples)
    return

  # Create two childrens for the node.
  positive_child, negative_child = split_node(node, condition)

  # List the training examples used by each children.
  negative_examples = [example for example in examples if not condition(example)]
  positive_examples = [example for example in examples if condition(example)]

  # Continue the growth of the children.
  grow_tree(negative_child, negative_examples)
  grow_tree(positive_child, positive_examples)
```

Another option is to optimize nodes globally instead of using a divide and conquer method like this.


### 2.2 Splitter

Splitter is responsible for find the best condition. They're the BOTTLENECK when training a decision tree.
The "score" maximized by the splitter depends on the task
1. Gini score and Information Gain score -> used for classification
2. MSE -> used for regression

Simples algorithm
---
For a single threshold condition on numerical data.
Shannon Entropy
-> quantifies improvement in label seperation achieved by a split
-> sorts values, uses each mid value as possible thresholds and tests O(nlogn)
-> Decision tree training with this splitter = O(mnlog^2(n)) m = features, n = examples 
   (because each node receives about ~1/2 of its prent)
-> insensitive to scale/distribution, eliminating need for feature normalization / scaling 

Formulas 
https://developers.google.com/machine-learning/decision-forests/binary-classification
Binary classification example formulas
```tex
\begin{eqnarray}
T & = & \{ (x_i,y_i) | (x_i,y_i) \in D \ \textrm{with} \ x_i \ge t \} \\[12pt]
F & = & \{ (x_i,y_i) | (x_i,y_i) \in D \ \textrm{with} \ x_i \lt t \}  \\[12pt]
R(X) & = & \frac{\lvert \{ x | x \in X \ \textrm{and} \ x = \mathrm{pos}  \} \rvert}{\lvert X \rvert} \\[12pt]
H(X) & = & - p \log p - (1 - p) \log (1-p) \ \textrm{with} \ p = R(X)\\[12pt]
IG(D,T,F) & = & H(D) - \frac {\lvert T\rvert} {\lvert D \rvert } H(T) - \frac {\lvert F \rvert} {\lvert D \rvert } H(F)
\end{eqnarray}
```
### 2.3 Pruning
Regularization parameters commonly used are,
1. Max depth for tree -> decrease -> reduces overfitting
2. Min examples in tree to be considered for split -> increase -> reduces overfitting

For pruning after training, 
A common solution to select the branches to remove is to use a validation dataset. That is, if removing a branch improves the quality of the model on the validation dataset, then the branch is removed.
Can only prune when you have more leaf nodes than categories you're trying to predict

### 2.4 Interpretability
Decision trees allow for interpretation of the data itself by looking at the structure

### 2.5 Variable Importances
Some features are more important than others.




## Random forests
Looks like what we need is the "multi-class classification random forest"

>  Wisdom of the crowd: In certain situations, collective opinion provides very good judgment.

We'll try both bagging and random attribute sampling

### Use of regularization

Pure random forests train without maximum depth or minimum number of observations per leaf. In practice, limiting the maximum depth and minimum number of observations per leaf is beneficial. By default, many random forests use the following defaults:

maximum depth of ~16
minimum number of observations per leaf of ~5.
You can tune these hyperparameters.