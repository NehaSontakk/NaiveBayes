# NaiveBayes

## NaiveBayesfromScratch.py 
Developing the naive bayes classifier from scratch, file is heavily commented and can be used as a tutorial. All the math involved is programmed from scratch without any libraries. I've explained the math used below.

Files bikeshare.py and output.txt use sklearns Gaussian Naive Bayes classifier to predict class of user from a bikesharing website.

#### Naive Bayes

It is a probabilistic classifier and assumes that some features that occur in the data are independent of other features.
###### For example:
If I want to check if the fruit is apple/not then I'll have the shape, size and colour. All these properties will individually contribute to the probability of the fruit being an apple.

#### Bayes Theorem

Probability of event B occuring given event A has occured  = Probability of event A occuring given B has occured * Probability of A occuring in the dataset whole / by Probability of B.
P(A|B) = P(B|A)P(A)/P(B)

###### Bayes for ML
The independent variables (x) can be added as a feature vector
eg: x = {size of fruit, colour of fruit, shape of fruit}

The dependent variable will belong to two categories since its a classification problem.
