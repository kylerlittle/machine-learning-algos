# perceptron
The program creates a randomized linearly separable dataset for training data, runs the perceptron learning algorithm on this training dataset, and produces an animation of the process. No non-training data points are tested, as it is obvious by the animation that the model converges correctly.

## Usage
Run `python perceptron.py` via the terminal. Adjust parameters in the test driver as you see appropriate. Here is a quick summary of what each parameter does.
* `max_iterations`: the number of iterations through the perceptron learning algorithm (i.e. the number of times the algorithm looks at a training data point to determine whether the current model misclassifies it or not)
* `max_x`: a tuple in which max_x[1] is the magnitude of the largest allowable value in the x_1 dimension and max_x[2] is the magnitude of the largest allowable value in the x_2 dimension

## Example
![animation example](https://github.com/kylerlittle/perceptron/blob/master/perceptron_animation.gif)

## Dependencies
Python 3.5, numpy, matplotlib, ffmpeg
