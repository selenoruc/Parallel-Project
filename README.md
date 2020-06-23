# YSA Paralel Projesi
YSA Öğrenmesinin Seri ve Paralel Programlama farkı ile ele alan bir projedir. 

Paralel Bilgisayarlar dersi dönem projesi olarak seçilen Yapay Sinir Ağı(YSA) konusunun yapım aşamaları, kullanılan teknolojiler ve paralelleştirmenin günlük hayattaki kullanım alanlarına değinilmiştir.

Proje aşamaları sırasıyla, yönerge doğrultusunda verilen kodların incelemesi, ilgili kod algoritmasının çıkarılması, yeniden YSA yapısının oluşturulup ilgili fonksiyonların yazılması, kodlama kısmının tamamlanmasının ardından ağır CPU hesaplaması gereken kısımların OpenMP pragmaları kullanılarak paralel hale getirilmesi, seri ve paralel testlerin yapılıp raporlanması/video hazırlanması olarak sıralanabilir.

## Paralel Programlama
Günümüzde, ticari uygulamalar daha hızlı bilgisayarların geliştirilmesinde eşit veya daha büyük itici güç sağlamaktadır. Bu uygulamalar büyük miktarda verinin karmaşık yollarla işlenmesini gerektirir. Örneğin:

-	Veritabanı(DataBase)
-	Veri madenciliği(Data Mining)
-	Petrol arama
-	Yapay Zekâ(Artificial Intelligence) geliştirmeleri
-	Search Web arama motorları, web tabanlı iş hizmetleri
-	İma Tıbbi görüntüleme ve teşhis
-	Yapay Sinir Ağı(Artificial Neuron Networks) oluşturma ve işleme
-	Ulusal ve çok uluslu şirketlerin yönetimi
-	Finansal ve ekonomik modelleme
-	Gelişmiş grafikler ve sanal gerçeklik, özellikle eğlence sektörü
-	Ağa bağlı video ve çoklu ortam teknolojileri
-	Ortak çalışma ortamları

gibi birçok alanda büyük veri işlemleri yapılmaktadır. Paralel hesaplama, daha büyük sorunları, paylaşılan bir bellek aracılığıyla iletişim kuran birden çok işlemci tarafından aynı anda yürütülebilen, sonuçları genel bir algoritmanın bir parçası olarak birleştirildiğinde daha küçük, bağımsız, genellikle benzer parçalara bölme işlemini ifade eder. Paralel hesaplamanın temel amacı, daha hızlı uygulama işleme ve problem çözme için mevcut hesaplama gücünü artırmaktır. Paralel hesaplama, yukarıda verilen alanlarda çokça kullanılmaktadır. Bu alanlarda kullanım sebeplerinin arasında;

-	Zamandan ve paradan kazanç sağlamak
-	Büyük problemleri çözmek
-	Aynı anda kullanımı sağlamak
-	Lokal olmayan kaynakların kullanımı
-	Seri programlamanın limitlerini ortadan kaldırılması

yer almaktadır.

## Kurulum 
Projeyi klonlayıp ilgili veri setlerinizi Labels60000_1D_Array.txt ve Train60000_1D_Array.txt olarak eklemeniz yeterli olacaktır.

```
# clone the repository
$ git clone https://github.com/selenkutanoglu/AndroidApp
```

## Proje Dizini 
İlgili proje dizin yapısı aşağıdaki görsel ile de verilmiş olup, ilk dizinde paralel hale getirilmiş proje, ikinci dizinde seri halde olan proje ve etiket ile eğitim verilerinin olduğu txt dosyaları yer almaktadır. Paralel ve seri proje dizinde yer alan HelperFunctions.h dosyalarında ağır CPU yükü gerektiren matris işlemlerinin yapıldığı fonksiyonlar yer almaktadır. NeuralNetwork dosyalarında ilgili YSA ağının gerekli veri setlemelerinin yapıldığı class yapıları yer almaktadır. Diğer dosyalar ise IDE olarak kullanılan Visual Studio tarafından oluşturulan proje dosyalarıdır.
```
| - YSA_cpp_OMP_parallelized
     | - HelperFunctions.h
     | - main.cpp
     | - NeuralNetwork.cpp
     | - NeuralNetwork.h
     | - YSA_cpp_OMP_parallelized.sln
     | - YSA_cpp_OMP_parallelized.vcxproj
     | - YSA_cpp_OMP_parallelized.vcxproj.filters
     | - YSA_cpp_OMP_parallelized.vcxproj.user 
| - YSA_cpp_serial
     | - HelperFunctions.h
     | - main.cpp
     | - NeuralNetwork.cpp
     | - NeuralNetwork.h
     | - YSA_cpp_serial.sln
     | - YSA_cpp_serial.vcxproj
     | - YSA_cpp_serial.vcxproj.filters
     | - YSA_cpp_serial.vcxproj.user
| -  Labels60000_1D_Array.txt
| -  Train60000_1D_Array.txt
```

## Test Aşaması 

Eğitim verilerinin bulunduğu Train60000_1D_Array.txt dosyasındaki 2. satır(veri adedi) sırasıyla 10, 20, 30, 40, 50 ve 60 bin olacak şekilde değişiklikler yapılıp test edilebilmektedir.

Seri derleme için;
```
# cd .\YSA_cpp_serial
$ g++ -o seri .\main.cpp .\NeuralNetwork.cpp
```
Paralel derleme için;
```
# cd .\YSA_cpp_OMP_parallelized
$ g++ -O -Wall -std=c++0x -g -fopenmp -o paralel_10_bin .\main.cpp .\NeuralNetwork.cpp
```
