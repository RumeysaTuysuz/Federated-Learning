# Federated Learning Model Comparison

Bu proje, farklı federated learning algoritmaları arasındaki performans farklarını karşılaştırmak için oluşturulmuştur.

## Uygulanan Federated Learning Algoritmaları

1. **FedAvg (Federated Averaging)**: McMahan et al. tarafından önerilen temel federated learning algoritması. Client'lar lokal olarak eğitim yapar ve model parametreleri ağırlıklı ortalama ile birleştirilir.

2. **FedProx**: FedAvg'in bir uzantısı olarak, client'ların heterojen olduğu durumlarda daha iyi performans sağlar. Global modele yakın kalması için proksimal terim ekler.

3. **FedSGD (Federated Stochastic Gradient Descent)**: Client'ların her round'da yalnızca bir mini-batch ile eğitim yaptığı basitleştirilmiş versiyon.

4. **FedNova (Normalized Averaging)**: Client'ların farklı sayıda local update adımı atması durumunda, bu farklılıkları normalize eden bir yaklaşım.

Kodun final çıktısı şu şekildedir:<br>
![fed models](https://github.com/user-attachments/assets/a13346ae-7741-47bd-90cb-86030554539c)
