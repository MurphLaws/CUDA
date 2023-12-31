# CUDA

This repository contains hands-on examples and practical exercises for programming on NVIDIA GPUs using CUDA (Compute Unified Device Architecture).


<p align="center">
  <picture>
  <img alt="img-name" src="https://guandi1995.github.io/images/classical_cnn/LeNet-5_modified.PNG" width="600">
</picture>
  <br>
    <em>Original image by  <a href="https://guandi1995.github.io/Classical-CNN-architecture/">Di Guan</a></em>
</p>


Los archivos importantes de cuda necesarios para la ejecucion del modelo LeNet5 son `LeNet5.cu` and `matrix_operations.cu`. El ejecutable resultante toma un unico argumento: la ID de una imagen del MNIST. La output sera una impresion de la imagen junto con los resultados de la ultima capa FC:

