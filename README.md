# Federated Learning Model Comparison

Bu proje, farklı federated learning algoritmaları arasındaki performans farklarını karşılaştırmak için oluşturulmuştur.

## Uygulanan Federated Learning Algoritmaları

1. **FedAvg (Federated Averaging)**: McMahan et al. tarafından önerilen temel federated learning algoritması. Client'lar lokal olarak eğitim yapar ve model parametreleri ağırlıklı ortalama ile birleştirilir.

2. **FedProx**: FedAvg'in bir uzantısı olarak, client'ların heterojen olduğu durumlarda daha iyi performans sağlar. Global modele yakın kalması için proksimal terim ekler.

3. **FedSGD (Federated Stochastic Gradient Descent)**: Client'ların her round'da yalnızca bir mini-batch ile eğitim yaptığı basitleştirilmiş versiyon.

4. **FedNova (Normalized Averaging)**: Client'ların farklı sayıda local update adımı atması durumunda, bu farklılıkları normalize eden bir yaklaşım.

## Projeyi Çalıştırma

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install tensorflow matplotlib numpy tqdm
```

2. Karşılaştırma skriptini çalıştırın:

```bash
python compare_methods.py
```

3. Sonuçları `results/` klasöründe bulabilirsiniz.

## Parametre Ayarları

Deneylerin parametrelerini `compare_methods.py` dosyasından değiştirebilirsiniz:

- `CLIENT_NUMBER`: Toplam client sayısı
- `CLIENT_RATIO_PER_ROUND`: Her round'da seçilen client oranı
- `EPOCHS`: Eğitim epoch sayısı
- `TEST_NUM`: Test örneklerinin sayısı

## Uygulamada Dikkat Edilmesi Gerekenler

- FedProx için `mu` parametresi (proksimal terimin ağırlığı) değiştirilebilir.
- FedNova, heterojen client'lar için daha iyi performans gösterebilir.
- FedSGD genellikle daha çok round gerektirse de, her round'da daha az hesaplama gerektirir.

## Sonuçların Yorumlanması

Sonuçları değerlendirirken şu faktörleri göz önünde bulundurun:

1. **Doğruluk (Accuracy)**: Her algoritmanın ulaştığı final test doğruluğu
2. **Kayıp (Loss)**: Final test kaybı
3. **Eğitim Süresi**: Algoritmaların toplam çalışma süresi
4. **Yakınsama Hızı**: Hangi algoritmanın daha hızlı yakınsadığı

## Geliştirme İçin Fikirler

- Farklı client seçim stratejileri uygulanabilir
- Non-IID veri dağılımı senaryoları test edilebilir
- Farklı model mimarileri kullanılabilir
- Çeşitli hyperparametreler optimize edilebilir 