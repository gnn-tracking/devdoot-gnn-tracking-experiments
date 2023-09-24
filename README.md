# Devdoot's GNN Tracking experiments

### Multi-objective optimization:
Object condensation involves learning coordinates for each vertex in a clustering space. This is achieved by optimizing the potential loss (L<sub>V</sub>) which comprises two components- the attractive and the repulsive potential loss which ensure that the vertices belonging to the same object are pulled towards the condensation point with the highest charge, and vertices not belonging to the object are pushed away, respectively. Another loss function, L<sub>β</sub> (background loss) is also added to L<sub>V</sub> to ensure one condensation point per object and none for the background. If we are also interested in estimating the transverse momentum (p<sub>T</sub>) for every particle, our loss function will need to include another component, L<sub>p</sub>.

Metric learning involves optimizing the Hinge Embedding loss for graph construction. The Hinge Embedding loss itself has a repulsive and an attractive term.

Since, we are dealing with multi-objective optimization, having non-static/dynamic weights for different loss terms can be beneficial because it allows for better control over the optimization process. There can be multiple techniques to achieve multi-objective optimization. One such method called the Constrained Differential Optimization has been explored in this repository. This method was first introduced in a [NIPS paper](https://papers.nips.cc/paper/1987/file/a87ff679a2f3e71d9181a67b7542122c-Paper.pdf) from 1998 by John C. Platt and Alan H. Barr.

Let’s consider two loss functions L1 and L2, which are conflicting in nature. We can frame our optimization problem as a Lagrangian optimization problem:

minimize L<sub>1</sub>(θ) subject to L<sub>2</sub>(θ) ≤ ε where ε is some threshold.

The Lagrangian looks like:

L(θ,λ) = L<sub>1</sub>(θ) + λ(L<sub>2</sub>(θ)-ε)

The solution corresponding to the original constrained optimization is always a saddle point of the Lagrangian function. Therefore, gradient decent won’t work here. Hence, we have to perform gradient ascent for the Lagrangian multipliers.

θ = θ - η∙∇θ

λ = λ + η∙δλ

To ensure more robust convergence, we can also introduce another hyperparameter, the damping factor c. A quadratic damping term is added to the original optimization equation:

L(θ,λ) = L<sub>1</sub>(θ) + λ(L<sub>2</sub>(θ) - ε) + c(L<sub>2</sub>(θ) - ε)<sup>2</sup>/2

### MDMM Module:

The `mdmm.py` file contains a class called `MDMMModule`. This class can serve as a foundation for any module that requires multi-objective optimization. It takes two dictionaries as inputs:
1.	A dictionary of main loss functions and their respective weights, identified by their names.
2.	A dictionary of constraint loss functions, along with their specific settings (type, weights, epsilons, and damping factors), also identified by their names.

In addition to the two dictionaries mentioned, it accepts other arguments that are thoroughly explained within the source code comments.

#### Example Use Case:
One instance where this multi-objective optimization comes into play is in computing Hinge embedding loss for Metric learning in graph construction.

### MDMM Metric Learning Module:
In the `mdmmml.py` file, you'll find a specialized class called `MDMMMLModule`. This class is derived from the `MDMMModule`. It computes the Hinge embedding loss to perform Metric learning.

### Noise Classifier:
In the repository, you'll find a noise classifier implemented in the gnn_noise.ipynb notebook. Here I have explored a noise classifier to filter out some noise from the point clouds before graph construction. I trained a simple MLP which could perform binary-classification to distinguish noise and actual hits.
It was able to achieve an accuracy of around 96% and the Recall-score (TPR) of 0.99 (negative class corresponds to noise). On an average, around 6.4% of the hits are noise. On average, the classifier identifies about 6.4% of the data points as noise.
