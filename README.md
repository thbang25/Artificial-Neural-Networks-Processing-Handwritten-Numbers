# -Artificial-Neural-Networks-Processing-Handwritten-Numbers
 Artificial Neural Networks Processing HandWritten Numbers

 **Reading the data from mnist**
To load the MNIST dataset for training and testing the model, we will use two instances: `train_mnist` and `test_mnist`. These objects are responsible for handling the loading of the dataset from files, or downloading it if necessary.

**Preparing the data**
Prepare and standardize data for neural network training by converting it to a tensor and normalizing it for numerical stability and faster convergence.

**Artificial Neural Network class definition**
Defines the structure of the neural network that will learn to classify handwritten digits using the custom neural network architecture 'ArtificialNeuralNetwork' class. This architecture is designed to process input images into classification predictions using PyTorch's `nn.Module`. 

 **Model Training Design**
To set up the training environment, we need to define the optimizer and loss function. For optimizer, we will use Adam as it is efficient in handling sparse gradients and adaptive learning rate features. The loss function we will use is Negative Log Likelihood, which is suitable for classification tasks that involve probability outputs from a softmax layer.

**Training and Testing Implementations**
The purpose of the process is to train the model using the training data, and then evaluate its performance using the testing data. During the training loop, the model's weights are adjusted to minimize loss, as it runs through the dataset in batches. On the other hand, the testing loop evaluates how well the model performs on unseen data, providing a measure of how well the model generalizes.

**Classifier Functionality**
Users can upload images and predict the digits using a trained model. The interface allows image path input, image processing, and digit prediction.
