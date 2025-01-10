import torch
import torchvision
import torchvision.transforms as transforms
import time  # Ajouté pour utiliser time.time()

# Charger le modèle
device = torch.device("cpu")
model = torch.jit.load("mnist_model.pt").to(device)
model.eval()

# Préparer le dataset de test MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Créer un DataLoader avec la taille du batch de 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Tester l'accuracy et mesurer le temps d'inférence
correct = 0
total = 0

# Mesurer le temps d'inférence
start_inference_time = time.time()  # Début du chronométrage pour l'inférence

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_inference_time = time.time()  # Fin du chronométrage pour l'inférence
inference_duration = end_inference_time - start_inference_time

accuracy = 100 * correct / total

# Afficher les résultats
print(f"Total inference time: {inference_duration:.4f} seconds")
print(f"Accuracy: {accuracy:.2f}%")

# Vérification du nombre de lots traités
num_batches = 0
with torch.no_grad():
    for images, labels in test_loader:
        num_batches += 1

print(f"Nombre total de lots traités : {num_batches}")
