# IOI Seminar II - Instrumental (Train PT Model)

Repozitorij vsebuje kodo zalednega dela aplikacije, ki omogoča da natreniramo model za prepoznavo gest z roko.

## Navodila za postavitev in uporabo projekta

Najprej poženemo sledeči ukaz, da se naložijo vse potrebne knjižnice: ```pip install -r requirements.txt```.

Nato poženemo datoteko ```server.py```. Tako na naslovu ```http://127.0.0.1:5000/train``` postane dostopen endpoint, ki
ga potrebuje GUI del, ko želimo natrenirati nov model.

Omenjeni endpoint kot vhod sprejme nabor ustrezno formatiranih koordinat rok, ki jih uporabi za učenje modela. Po
uspešnem učenju pa model shrani v datoteko s končnico *.onnx*, ki jo vrne kot odgovor, zato da jo lahko nato
uporabimo na GUI, pri prepoznavi gest rok.

## Evalviranje modelov

Če želimo evalvirati model moramo najprej prekopirati datoteko ```model.pt``` iz mape ```out_models``` v mapo ```trained_models```. Ta datoteka se ob vsakem novem treniranju sicer prepiše z novim modelom.

Nato moramo v skripti ```evaluate.py``` spremeniti vrednost spremenljivke ```TRAINED_MODEL```, da bo kazala na datoteko modela, ki ga želimo evalvirati. Potem lahko skripto še poženemo in izpisali se bodo rezultati pri testiranju modela, na vseh podatkih iz direktorija ```data/test```.