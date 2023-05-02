# Conclusion

In this project we have gained insight into techniques used for data at scale. In particular, we investigated the problem of multi-class classification using images of plant leaves. Whilst the dataset uses images of real leaves, the photographs in the Plant Village data are very consistent in the sense that they are all taken from the same angle, and have a similar background. Many datasets for real world classification problems do not benefit from the same consistency, and so our neural network results may reflect more favourable performance than can be expected for other image datasets. 

As the main focus of this project was to learn about data at scale, we decided to experiment with a broad range of techniques to see what worked well with our data, and gain insight into the whole process of fitting a neural network. In particular, we investigated the role of activation functions, pre-trained models, autoencoders and parallelisation at the pre-processing stage.

In the activation function experiment, we compared the predictive performance of four different activation functions (Swish, Relu, Tanh and Sigmoid) within a fixed neural network architecture. We found that while Swish, tanh and Relu all produced similar prediction accuracy results, Swish was over 10% slower than the other two. This study was conducted only on a very small subset of the data, and so in the case of much larger data, this difference would likely be magnified. Further, when performing data manipulation at scale, the speed at which algorithms can be trained becomes even more important. Hence it is unlikely that Swish would be a cost-effective choice for our image classification task. The Sigmoid model was both slower and yielded a worse accuracy than ReLu and Tanh. This result aligns with the literature, which suggests that the Sigmoid activation, whilst Tanh is better for multi class problems.

After experimenting with several autoencoder models, we found that the model with more layers performed better on our high-quality image dataset. The model's ability to effectively denoise images with added Gaussian noise suggests that it could be applied to other similar tasks.

Furthermore, we investigated the impact of multi-processing on image pre-processing for datasets of different sizes. Our findings showed that multi-processing is beneficial for large datasets as it significantly reduces computational time. However, for smaller datasets, it may even increase processing time. These results can help guide decisions around resource allocation when working with image datasets.


### References
1. https://medium.com/codex/activation-functions-in-neural-network-steps-and-implementation-df2e4c858c21
2. https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
3. https://medium.com/analytics-vidhya/denoising-autoencoder-on-colored-images-using-tensorflow-17bf63e19dad
4. https://github.com/therealcyberlord/tensorflow_keras_color_images_denoiser/blob/master/better_denoiser.ipynb
