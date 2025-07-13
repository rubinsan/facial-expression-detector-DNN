# facial-expression-detector-DNN
The code you will find in this repo represents the results of mi personal training and evolution in concepts related with Deep Neural Networks.

The goal is to build a facial expression detector which can infere and detect different emotions from input images.

## Neural net class
First I developed a class which represent a DNN as an instantiated object. You can model the architecture: number of hidden layers and number of neurons in each layer. There exist several methods to make forward and backward prop, apply cost funtion and run gradient descent. I have also coded a method to evaluate de results of the training vs a test set of input samples. 

## MNIST data set
To validate the nerural net class I used the MNIST dataset which you can find in this same repo as a .csv files. It contains 60,000 training samples plus 10,000 test samples of manuscript digits (0-9) in 28x28 pixel grayscale image format and a color depth of 8 bits per pixel
It is very gratefull to see it works like a charm :)


## facial_emotion_detection_dataset
Work in progress...


## Author

* **Ruben Sanchez** - [rubinsan](https://github.com/rubinsan)

## License

This project is licensed under the [MIT License](LICENSE).
