#ifndef TEST_MNIST_LIBTORCH_H  // Protection d'inclusion pour éviter les redéfinitions
#define TEST_MNIST_LIBTORCH_H
#include <torch/torch.h>
#include <torch/script.h>  // Nécessaire pour torch::jit::load
#include <torch/data/datasets/mnist.h>  // Correctement inclus ici

#include <iostream>
#include <fstream>
#include <chrono>

// Charger le modèle avec torch::jit::load
torch::jit::script::Module load_model(const std::string& model_path) {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path); // Charger le modèle TorchScript
    } catch (const c10::Error& e) {
        std::cerr << "Erreur lors du chargement du modèle : " << e.msg() << std::endl;
        exit(1);
    }
    return model;
}

int main() {
    // Chemin vers ton modèle sauvegardé
    std::string model_path = "/home/meli/dnn_inference/mnist_model.pt";

    // Charger le modèle
    auto model = load_model(model_path);

    // Charger les données MNIST (en utilisant la bibliothèque de LibTorch)
    auto dataset = torch::data::datasets::MNIST("/home/meli/dnn_inference/data/MNIST/raw/MNIST/raw",
     torch::data::datasets::MNIST::Mode::kTest)// Mode test
        .map(torch::data::transforms::Normalize<>{0.5, 0.5})  // Normalisation des données
        .map(torch::data::transforms::Stack<>()); // Conversion en tensor

    // Créer un DataLoader pour charger les données par lot
    auto data_loader = torch::data::make_data_loader(std::move(dataset), 64); // Taille du batch = 64

    size_t correct = 0; // Nombre de prédictions correctes
    size_t total = 0;   // Nombre total d'exemples
    size_t batch_count = 0; // Compteur de lots

    // Désactiver les gradients pour l'inférence
    torch::NoGradGuard no_grad;

    // Mesurer le temps total d'inférence
    auto inference_start = std::chrono::high_resolution_clock::now();

    // Boucle sur les données pour faire l'inférence et calculer l'accuracy
    for (auto& batch : *data_loader) {
        batch_count++; // Incrémenter le compteur de lots

        torch::Tensor inputs = batch.data.to(torch::kFloat32);  // Assurer que les données sont en float32
        torch::Tensor labels = batch.target; // Les vraies étiquettes

        // Faire l'inférence
        torch::Tensor outputs = model.forward({inputs}).toTensor();

        // Prédictions (classes avec la probabilité la plus élevée)
        torch::Tensor predictions = outputs.argmax(1);

        // Calculer les prédictions correctes
        correct += predictions.eq(labels).sum().item<int64_t>();
        total += labels.size(0); // Ajouter le nombre d'exemples du batch
    }

    auto inference_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> inference_duration = inference_end - inference_start;
    std::cout << "Temps total d'inférence : " << inference_duration.count() << " secondes" << std::endl;

    // Calculer l'accuracy
    float accuracy = static_cast<float>(correct) / total * 100.0;

    // Afficher les résultats
    std::cout << "Précision (accuracy) : " << accuracy << "%" << std::endl;
    std::cout << "Nombre total de lots traités : " << batch_count << std::endl; // Afficher le nombre de lot

    return 0;
}

#endif // TEST_MNIST_LIBTORCH_H  // Fermeture de la protection d'inclusion
