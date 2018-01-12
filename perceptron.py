
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
        self.w = np.zeros(training_data.len())   # Assumed that training_data is augmented by one for convenient threshold slot (index 0)
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
              num_elements is the size of the training set; num_features is the dimension of the input data
              max_x_0 is the magnitude of maximum value for input vectors in element 0; likewise for max_x_1
              recall x[0] will be the 1-vector by default
    """
    def __init__(self, num_elements, num_features, max_x_0, max_x_1):
        self.num_elements = num_elements + 1   # augment by one for convenient slot for threshold (w_0)
        self.x = np.ones((self.num_elements, num_features))
        self.y = np.zeros(self.num_elements, dtype=np.int8)
        self.max_x_0 = max_x_0
        self.max_x_1 = max_x_1
        self.__create_dataset__()

    """
    __create_dataset__: create randomized data with certain limitations
    """
    def __create_dataset__(self):
        for index, element in enumerate(self.x):
            if index > 0:   # index 0 is the 1-vector by default
                x_0 = np.random.uniform(-self.max_x_0, self.max_x_0)
                x_1 = np.random.uniform(-self.max_x_1, self.max_x_1)
                element[0] = x_0; element[1] = x_1
                self.y[index] = (-1 if x_1 < 0. else 1)    # assign -1 on values below x_0 axis & +1 on values above x_0 axis

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
    len: return number of elements in data set, including the +1 for index 0 slots
    """
    def len(self):
        return self.num_elements


"""
test driver: only executed when run as "python perceptron.py"; not as import
"""
if __name__ == "__main__":
    max_iterations = 10000
    training_dataset = training_set(20, 2, 10, 10)
    print training_dataset.get_x()
    print training_dataset.get_y()
    #model = perceptron(training_dataset, max_iterations)
