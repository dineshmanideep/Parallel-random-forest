non-leaf nodes are called conditions/split

## Conditions
1. axis-aligned condition: num_legs ≥ 2
2. oblique condition: num_legs ≥ num_fingers

oblique maybe more powerful, but more training cost

### Most common type of condition
is the 'threshold condition'
expressed as feature ≥ threshold

Name                 |  Condition                                |  Example                          
---------------------+-------------------------------------------+-----------------------------------
threshold condition  |  feature_i >= threshold                   |  num_legs >= 2                    
equality condition   |  feature_i = value                        |  species = "cat"                  
in-set condition     |  feature_i in collection                  |  species in {"cat", "dog", "bird"}
oblique condition    |  sum_i weight_i * feature_i >= threshold  |  5 * num_legs + 2 * num_eyes >= 10
feature is missing   |  feature_i isMissing                      |  num_legs isMissing               


## Training Decision Trees
Actual optimal training algorithm is NP-Hard 
SO we use heuristics - easy, non-optimal but close to optimal

### Algorithm
Greedy Divide and Conquer strategy
