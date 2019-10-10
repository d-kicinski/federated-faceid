### About faceid
    'Face ID uses a "TrueDepth camera system", which consists of sensors, cameras, and a dot projector at the top of the iPhone display in the notch to create a detailed 3D map of your face. Every time you look at your phone, the system will conduct a secure authentication check, enabling your device to be unlocked or your payment to be authorised quickly and intuitively if it recognises you.'

Mój system:
    - będzie wykorzystywał prostą kamerę bez dodatkowych czujników głębi.
    - najwieksza uwaga zostanei skupiona na implementacji architektóry podejscia federeated


### Models research
Face detection/identification/verification
1. Softmax loss with extra steps
 - [Center loss](https://kpzhang93.github.io/papers/eccv2016.pdf)
 - [SphereFace](https://arxiv.org/abs/1704.08063)
 - [CosFace](https://arxiv.org/abs/1801.09414)
 - [ArcFace(SOTA)](https://arxiv.org/pdf/1801.07698v3.pdf)

loss to zadanie identyfikacji (klasyfikacji twarzy do jednej z n osobowości) z dodanymi bajerami
sprawiającymi, że w przestrzeni cech twarze jednej osoby są umieszczono kompaktowo blisko
siebie(blisko uczonego środka grupy), a grupy są rozmieszczone jak najdalej od pozostałych grup

czyli są dwie składowe:
- te same twarze blisko siebie
- różne twarze dlako od siebie

- na mobilce mogę policzyć wektor cech twarzy, ale nie mogę policzyć dobrego lossa?

experymenty/rozwiązania:
FE = feature extractor
C = classifier o wymiarze N (liczba uzytkownikow w rundzie)

---
* dla każdemu uzytkownikowi przeslij FE i randomowy C
* kilka epok na urządzniu kazdego uzytkownika - tylko dane dla jednej klasy
* przesylamy sa serwer tylko FE
* koniec rundy

- uber Non-IID data
- warstwa klasyfikatora nigdy nie widzi innych danych
- raczej nie bedzie dzialac?
---
* początek rundy -> wybór użytkowników i randomowa inicjalizacja C

* dla każdemu uzytkownikowi przeslij FE i aktualny C
* kilka epok na urządzniu kazdego uzytkownika - tylko dane dla jednej klasy
* przesylamy sa serwer FE i C
* powtórz rundę dla tych samych użytkowników R razy

- Non-IID data
- użytkownik musi często pobrać nowy model
- trudne w implementacji w środowisku federated learningu
- serwer ma klasyfikator = możliwa idendyfikacja przez serwer użytkownika = słabo
---

---
Douczanie ogólnego FE
- klasteryzacja twarzy na urzyądzeniu użytkownika
- dobrze bo model moze lepiej bedzie odroznial bliskich znajomych od nas

* dla każdemu uzytkownikowi przeslij FE i randomowy C
* kilka epok na urządzniu kazdego uzytkownika
* przesylamy sa serwer tylko FE
* koniec rundy
---

2. Triplet loss
- FaceNet [Paper](https://arxiv.org/pdf/1503.03832v3.pdf) [Code](https://github.com/timesler/facenet-pytorch)

---
Douczanie ogólnego FE
- zbierz twarze, które nie są użytkownikiem na urządzeniu

* dla każdemu uzytkownikowi przeslij FE i randomowy C
* kilka epok na urządzniu kazdego uzytkownika
* przesylamy sa serwer tylko FE
* koniec rundy


Lightweight DCNN
- MobileNetv2


### Datasets
Wymagania:
- zdjecia powinny przypominac selfie
- dużo zdjęć do jednej osoby


### Projekt eksperymentów



### Mind dump
    - maybe use generated faces as negative examples, send them from server with new the model
    - maybe use faces of other people gathered on phones
    - somehow send just features from other users that participate in current round


