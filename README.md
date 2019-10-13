# dog-generator
Using a generative adversarial network to generate images of dogs

After taking [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro), as well as their [course on generative adversarial networks](https://developers.google.com/machine-learning/gan), I was inspired to create a network that generates images of dogs.

This project has three scripts: `loadPics.py`, which takes a directory of dog images and scales/transforms them to be more managable, `loadScaledNumpys.py`, which takes a directory of scaled dog images and preprocesses them into numpy arrays, and `dogPicGenerator.py`, which takes the numpy arrays and trains a GAN to generate dog photos.

The following picture is a result of training the model on 15,000 golden retriever headshots for 11,000 steps:

![Generated Dog Pics](https://user-images.githubusercontent.com/33944844/66719392-ed288100-edbc-11e9-9d8c-f0eb3a70773a.png "Generated Dog Pics")

Not perfect, but there's definitely some dogs in there! :)
