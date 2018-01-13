from matplotlib import animation
import matplotlib.pyplot as plt
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
    def __init__(self, training_data, max_iterations, num_features):
        self.w = np.zeros((num_features + 1, 1))# index 0 represents bias weight; indices 1 and 2 are the 2 dimensions of input vectors x_i
        self.training_data = training_data
        self.max_iterations = max_iterations

    """
    __sign__: returns +1 if number is greater than 0; otherwise, -1 
    """
    def __sign__(self, num):
        return 1 if num > 0. else -1

    """
    train: train perceptron using training data; return the weighting vector which perfectly classifies the training_data
    """
    def train(self):
        cleanPass = True; t = 0
        while t < self.max_iterations: 
            cleanPass = True
            data_val = self.training_data[np.random.randint(0, len(self.training_data))]
            if self.__sign__(np.dot(np.transpose(self.w), data_val[0])) != data_val[1]:   # Misclassified Item
                cleanPass = False
                self.w = self.w + (data_val[1] * data_val[0])
            t += 1
        return self.w

"""
training_set: class wrapper for training data; produces a linearly separable data set
"""
class training_set:
    """
    __init__: constructor
              num_elements is the size of the training set; num_features is the dimension of the input data
              max_x_1 is the magnitude of maximum value for input vectors in element 1; likewise for max_x_2
              recall x_0 is always 1.0
    """
    def __init__(self, num_elements, max_x_1, max_x_2):
        self.num_elements = num_elements
        self.data_set = []   # list of tuples of the form (x, y), where x is a 2-dimensional vector
        self.index = 0
        self.__create_data_set__(max_x_1, max_x_2)

    """
    __create_data_set__: create randomized data with certain limitations
    """
    def __create_data_set__(self, max_x_1, max_x_2):
        for index in range(self.num_elements):
            x_1 = np.random.uniform(-max_x_1, max_x_1); x_2 = np.random.uniform(-max_x_2, max_x_2)
            x = np.array([[1.0], [x_1], [x_2]])  # x[0] is fixed to be 1.0
            y = -1 if x_1 < 0. else 1    # assign -1 to values below x_0 axis & +1 to values above x_0 axis
            self.data_set.append((x,y))

    """
    __iter__: overrides __iter__
    """
    def __iter__(self):
        return self

    """
    __next__: overrides __next__
    """
    def __next__(self):
        try:
            data_val = self.data_set[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return data_val

    """
    __len__: overrides __len__
    """
    def __len__(self):
        return self.num_elements

    """
    __getitem__: overrides __getitem__
    """
    def __getitem__(self, key):
        return self.data_set[key]
    
    """
    get_data: basic accessor to return list
    """
    def get_data(self):
        return self.data_set


"""
test driver: only executed when run as "python perceptron.py"; not as import
"""
if __name__ == "__main__":
    max_iterations = 100; num_elements = 20; num_features = 2; max_x_1 = 10; max_x_2 = 10
    training_dataset = training_set(num_elements, max_x_1, max_x_2)
    model = perceptron(training_dataset, max_iterations, num_features)
    weight_vector = model.train()
