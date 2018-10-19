---
layout: post
title: Father to Son: Part I
---
# Father to Son : Part I
*Practical advices to train your deep models efficiently*

One should approach modeling with a principle rather than a technique, a principle not only encompasses the techniques but also creates better grounds for creating new techniques. One such principle is "Going from simple to complex", i.e. create a simpler version, make it work and introduce complexity on top of it.

**Tip** : Always keep a copy of these simpler but working basis versions(And create new ones on top of it) such as CNNs, LSTMs etc with all the basic training and logging processes. Makes your life faster.

Divide the modeling process into two parts, The Subject(Our NN architecture) and The Predicate(Data loading, augmentations, logging, training etc):

## Creating a Working Model:

- The architecture should initially be minimal, a single layer, with a simple ReLU.

- You can add a BatchNorm but always keep a precaution switch in mind. While rare, the BatchNorm puts an upper limit to the NaN activation values if present. You might lose sanity debuggging your model watching everything work perfectly but still not getting your model to converge.

- Initially, work with a small subset of the data to make everything work. No Augmentations.

- Once it does, move to Train/Valid split to tune the hyperparameters. Once you do, retrain it again on whole dataset(Train+Valid) before deployment to get even better results.

- Don't introduce any loops initially, keep it serial or single looped.

- Have a look at your data, create a summary, if its images, show some samples with labels.

- Double check your logging formulas, whether you are zeroing the divisor counters in a consistent way. Logs are the only way to know your results, a slight mistake there and your model might be training all the right but you won't know.

- Once everything works, scale up the Predicate first i.e. Data Augmentations, Logging etc.

## Scaling Up:

- Increase the model complexity, the intuition being more complex model leads to overfitting, so more the merrier and you can always add regularization to reduce the complexity implicitly and fit the data perfectly, the downside no one mentions is, bigger the model, more parameters to optimize, longer will it need to converge.

- Adding bias in every layer is essential but if you are introducing a BatchNorm after it, you don't need one.

- A good test of your model is by deactivating all the regularization and allow the model to overfit to the data, it tells about the capacity of the model. If your model reaches well more than 99% and ideally 100%, you're good to go!

- If the model overfits, well and good, now add all the regularizers, augmentations etc and switch to the Train+Valid split dataset.

- Normalize your input data or use batch-norm as first layer in the network.

- 64/128 filters per layer are more than enough.

- Don't bother with Manual weight decay, switch to Adam Optimizer and Chill.

## Efficiency :

- If you want to perform image-reconstructions such as in U-Nets, avoid BatchNorms as it changes input signal.

- If it's a spatial variance dependent problem, such as object detection, avoid Fully Connected layers in the end and switch to Global-Avg Pooling.

- Use ReLU before, not after the Batch Norm

- Avoid using DenseNets, they sound good on paper but take a lot of memory.

- Make Resnets a go-to model and replace all the convolutions with depth-seperable convolutions.

- Don't use Max-pools, replace them with stride-2 convolutions.

- Use Dense Sparse Dense(DSD training): It's a healthy approach, same Model, same Dataset and DSD will increase your model capacity, effectively reducing the amount of needed models in an ensemble. 

- If you want a smaller model, trade-off DSD with pruning.

- To create an ensemble, don't train multiple models again and again, instead use SGD with restarts and create a snapshot at each minima, later combine them to get an ensemble. My personal touch? Take the weighted average of their predictions, weighting later snapshots more as they are better generalized.

- Use Half Tensors(Float16): They are approximators with so many parameters that it doesn't affect accuracy much if at all and you slash your parameter size in half, also Volta and later architectures support half tensors so your model speeds up as well.

- Pre-Computations: If you have a lot of memory to spare you can use Pre-computations to speed-up your transfer-learning. Usually you freeze 'N' no of layers and train the later 'M' layers. If 'M' layers are less and dataset is large enough, you won't require augmentation either. Since the frozen layers are not learning, their output is always same. Just run a dry epoch in which you compute all the outputs/activations upto 'N' layers for once. Re-use these activations for all the epochs to train the 'M' layers.

- Cythonize: Cythonize all the code, it will reduce all the overhead due to python.

- Switch Optimizers: Different Optimizers have different properties, some generalize better, some obtain lower minima, some converge faster. Use a Dynamic framework such as PyTorch and make switch between them to get the most efficiently trained models














