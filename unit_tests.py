"""
Scrit for testing the neuralnet class methods
"""

from neuralnet import NeuralNet

test_NN = NeuralNet([2, 3, 3, 4, 2])
test_NN.initialization()
keys = test_NN.weights.keys()
print("Weights keys:", keys)
#print(test_NN.weights["W2"])
for layer in test_NN.weights.keys():
    print(layer+":")
    print(test_NN.weights[layer])
for layer in test_NN.biases.keys():
    print(layer+":")
    print(test_NN.biases[layer])