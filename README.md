<h1 align="center" >VECTORIZED LOGISTIC REGRESSION </h1>

#### Logistic Regression is a statistical machine learning model used for binary classification tasks, where the goal is to predict a binary output (e.g., 0 or 1) given some input features. The model is based on the logistic function, which maps any real-valued input to a value between 0 and 1, representing the probability of the binary output being 1.

#### In logistic regression, the model is trained using a labeled dataset, where each data point consists of a set of input features and a corresponding binary label. The model learns the relationship between the input features and the binary output by minimizing a loss function, such as the binary cross-entropy loss, using an optimization algorithm, such as gradient descent.

#### Logistic regression has many applications in various fields, such as finance, healthcare, and marketing. It can be used to predict the likelihood of an event occurring based on a set of input features, such as the likelihood of a customer buying a product given their demographic information. Additionally, logistic regression can be extended to handle multi-class classification tasks using techniques such as one-vs-all and softmax regression.  
<h3 align="center">------------------------------</h3>
### In vectorized form, the logistic regression model can be expressed as:


```h(x) = σ(Xθ)```

#### where:

1) h(x) is the predicted probability that the input x belongs to the positive class
2) X is a matrix of input features, with each row representing a single data point and each column representing a single feature
3) θ is a vector of weights, with each weight corresponding to a feature in X 
4) σ(z) is the sigmoid function, which maps any real-valued number to a value between 0 and 1:
5) σ(z) = 1 / (1 + e^-z) (SIGMOID FUNCTION)


#### NOTE THAT WE ADD A ONES COLUMN AND A PARAMETER IN WEIGHTS FOR BIAS

<h3 align="center">------------------------------</h3>
### The cost function for logistic regression is the negative log-likelihood:

```J(θ) = -1/m * (y^T log(h) + (1-y)^T log(1-h))```

#### where:

1) m is the number of data points 
2) y is a vector of target labels (0 or 1)
3) y^T is the transpose of y 
4) log(h) is the element-wise natural logarithm of h(x)
5) log(1-h) is the element-wise natural logarithm of 1-h(x)

6) <h3 align="center">------------------------------</h3>

#### To minimize the cost function and find the optimal values of θ, we can use gradient descent. The gradient of the cost function with respect to θ is:

```∇J(θ) = 1/m * X^T (h-y)```

#### where:

1) ∇ is the gradient operator 
2) X^T is the transpose of X

#### We can update θ using the following rule:

```θ = θ - α * ∇J(θ)```

#### where α is the learning rate. This process is repeated until convergence or a maximum number of iterations is reached.