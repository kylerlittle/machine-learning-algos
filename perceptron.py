
import numpy as np

"""
perceptron:
   - a supervised learning model which determines (via the perceptron learning algorithm) whether or not an input with
     a feature vector belongs to a particular class
   - this model assumes the training_data is linearly separable (i.e. a hyperplane can completely and correctly
     divide the training_data)
   - this model also assumes output is binary in the form of +1 or -1

"""
class perceptron:
    """
    __init__: constructor
    """
    def __init__(self, training_data, max_iterations):
        self.w = np.zeros(len(training_data) + 1)   # Augment by one for convenient threshold slot (index 0)
        self.training_data = training_data
        self.max_iterations = max_iterations

    """
    __sign__: returns +1 if number is greater than 0; otherwise, -1 
    """
    def __sign__(num):
        return 1 if num > 0 else -1

    """
    train: train perceptron using training data; return the weighting vector which perfectly classifies the training_data
    """
    def train(self):
        cleanPass = True
        t = 0
        while t < self.max_iterations: 
            cleanPass = True
            for data_val in self.training_data:
                if __sign__(data_val.x * self.w) != data_val.y:   # Misclassified Item
                    cleanPass = False
                    w = w + self.training_data.y * self.training_data.x
                    break
            t += 1
        return self.w
