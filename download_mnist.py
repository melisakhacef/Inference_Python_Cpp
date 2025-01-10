import torchvision.datasets as datasets


dataset_path = "/home/meli/dnn_inference/data/MNIST/raw/"

# Download the MNIST dataset
datasets.MNIST(root=dataset_path, train=True, download=True)
datasets.MNIST(root=dataset_path, train=False, download=True)
