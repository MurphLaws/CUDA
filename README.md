# CUDA

<ul>
    <li>Nicolás Lasso</li>
    <li>Rayane Ghilene</li>
</ul>




This repository contains hands-on examples and practical exercises for programming on NVIDIA GPUs using CUDA (Compute Unified Device Architecture).


<p align="center">
  <picture>
  <img alt="img-name" src="https://guandi1995.github.io/images/classical_cnn/LeNet-5_modified.PNG" width="600">
</picture>
  <br>
    <em>Original image by  <a href="https://guandi1995.github.io/Classical-CNN-architecture/">Di Guan</a></em>
</p>

<br>

The important cuda files required for running the LeNet5 model are `LeNet5.cu` and `matrix_operations.cu`. The resulting executable takes a single argument: the ID of an MNIST image. The output will be a printout of the image along with the results of the last FC layer:

<br>

<div align="center" style="display: flex ;">
    <picture>
    <img src="https://github.com/MurphLaws/CUDA/assets/36343734/cc74c425-472f-4020-8708-aed6bf42a764" alt="image1" width="40%">
    </picture>
    <picture>
    <img src="https://github.com/MurphLaws/CUDA/assets/36343734/b1aa2331-5983-4119-94f2-eae2979bfc40" alt="image2" width="40%">
    </picture>
</div>

<br>

One last appreciation regarding this implementation. The layers weights are defined, exported and imported correctly. However, the defined model behaves in a non-deterministic way (giving different classes with each execution), despite not having introduced any type of randomness in the model, which leads us to think that this problem is found in the assigment of CUDA blocks and grids. With more time it is perfectly possible to solve this problem, since individually all the functions are executed correctly, but once in sequence they present these undesired results.

<br>

Finally, to say that it was a very fun lab :)
