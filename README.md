# distributed-convergent-model
## Installation
This repositry needs Python 3.9 or later, with packages listed in requirements.txt.
Ensure that the Python and pip are installed and added to the PATH.
To install the packages, run following
```shell-session
$ pip install -r requirements.txt
```

## How to run
The following training is implemented in Pytorch.
They contain the training of the projection model, the Hamiltonian model, and the S-MPNN model.
They also contain dataset generation and long-term prediction.
Before running the training, move to this directory.
### Train aggregation
Run following 
```shell-session
$ python3 train_aggregation.py "result_dir"
```
where "result_dir" is the directory to save the results.
You can specify the hyperparameters by adding the arguments (see train_aggregation.py for details).
### Train Boids
Run following 
```shell-session
$ python3 train_boids.py "result_dir"
```
where "result_dir" is the directory to save the results.
The arguments are also available (see train_boids.py for details).
### Train both target
Run following 
```shell-session
$ bash train_all
```
This script runs the training of the aggregation and Boids models.
The results are saved in the directories `result/aggregation` and `result/boid`.

>[!WARNING]
>The training may take a long time.
>It is recommended to run the training of Boids or both on a machine with a GPU.

## Implementation
### Graph utility
Consider a graph $(\mathcal{N},\mathcal{E})$ with node set $\mathcal{N}\triangleq\{1,...,n\}$ and edge set $\mathcal{E}\in\mathcal{N}\times\mathcal{N}$ and vectors $x_1,...,x_n$ indexed by $\mathcal{N}$.
To treat the edge set, this repositry use two arrays `row` and `col`.
The length of `row` and `col` equal to $|\mathcal{E}|$ and the pair of `k`-th components of `row` and `col` is included in $\mathcal{E}$.
Additionally, treat $x_1,...,x_n$ as `x`.
Consider the implementation of $v_{ij}$ or S-MPNN in the paper.
First, make pairs $(x_i,x_j)$ for $(i,j)\in\mathcal(E)$ by obtaining pair of `x[row],x[col]`, e.g., `torch.cat((x[row],x[col]),axis=1)`.
Next, apply the function $\phi$ for each components of `torch.cat((x[row],x[col]),axis=1)`.
Then, the output is $\phi(x_i,x_j)$.
Finally, to calculate $m_i=\sum_{j\in\mathcal{N}_i}\phi(x_i,x_j)$, we implement `unsorted_segment_sum(x,segment_idxs,n)` with Pytorch, which is counterpart of `tf.math.unsorted_segment_sum` (https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum).
This function compute $output[i]=\sum_{j...}x[j...]$ where the sum is over the tuples j... such that `segment_idxs[j...]==i`.
Therefore, if `segment_idxs=row`, that calculates $m_i=\sum_{j\in\mathcal{N}_i}\phi(x_i,x_j)$.

### Gradient of scalar functions $V(x)$
The gradient is calculated using automatic differentiation function `torch.autograd.grad` in Pytorch.
For simple calculation, the scalar functions are added for minibatch.
