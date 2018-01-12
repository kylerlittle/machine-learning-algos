
import numpy as np

"""
perceptron:
   - a supervised learning model which determines (via the perceptron learning algorithm) whether or not an input 
     defined by its associated feature vector belongs to a particular class
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
        while cleanPass and t < self.max_iterations: 
            cleanPass = True
            for data_val in self.training_data:
                if __sign__(np.dot(data_val.get_x * self.w)) != data_val.get_y:   # Misclassified Item
                    cleanPass = False
                    self.w = self.w + (self.training_data.get_y * self.training_data.get_x)
                    break
            t += 1
        return self.w

"""
training_set: class wrapper for training data; produces a linearly separable data set
"""
class training_set:
    """
    __init__: constructor
    """
    def __init__(self):
        self.x = []
        self.y = []
        self.__load_data__()

    """
    __load_data__: load data
    """
    def __load_data__(self):
        pass

    """
    get_x: simple accessor
    """
    def get_x(self):
        return self.x

    """
    get_y: simple accessor
    """
    def get_y(self):
        return self.y


"""
test driver: only executed when run as "python perceptron.py"; not as import
"""
if __name__ == "__main__":
    max_iterations = 10000
    training_dataset = training_set()
    model = perceptron(training_dataset, max_iterations)
