Inference-Python-vs-C++
Description
Ce projet compare les temps d'inférence entre Python et C++ pour un modèle de réseau de neurones convolutifs (CNN) entraîné sur le jeu de données MNIST. L'objectif est d'évaluer la différence de performance entre ces deux langages sur un environnement CPU.

Le modèle est d'abord entraîné en Python avec PyTorch, puis exporté en format TorchScript. Ensuite, le modèle est chargé et utilisé pour l'inférence dans les deux langages, Python et C++.

Fonctionnalités
Entraînement du modèle CNN sur le jeu de données MNIST en utilisant PyTorch.
Sauvegarde du modèle entraîné au format TorchScript (mnist_model.pt).
Inférence du modèle en Python avec PyTorch.
Inférence du modèle en C++ avec LibTorch (la version C++ de PyTorch).
Comparaison des temps d'inférence entre Python et C++ sur un CPU.
Prérequis
Python
Python 3.x
PyTorch
Numpy
Matplotlib (facultatif, pour l'affichage)
C++
C++11 ou version supérieure
LibTorch (version compatible avec PyTorch)
CMake
