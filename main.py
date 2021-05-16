from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import seed
from random import randrange

####################
# Zmienne globalne #
####################

plikZrodlowy = open("iris.data", "r")  # wczytanie do zmiennej pliku z danymi
P = list()  # zbiór wejściowy (tu znajdują się cechy obiektu)
T = list() # zbiór identyfikatorów (tu znajdują się identyfikatory, po których klasyfikowane są obiekty)
kolumnaT = 4  # indeks kolumny T
zbiorDanych = list()

################
# Algorytm LVQ #
################

def lvq(daneUczace, daneTestowe, iloscWylosowanychWektorow, wspUczenia, iloscEpok):
    zestawDanych = trenuj(daneUczace, iloscWylosowanychWektorow, wspUczenia, iloscEpok) # trenuj zestaw danych
    prognozy = list() # zainicjalizuj pustą listę prognoz
    for wektor in daneTestowe: # dla każdego wektora w danych testowych
        output = prognoza(zestawDanych, wektor) # dopasuj najbliższy wektor
        prognozy.append(output) # dodaj wynik do listy prognoz
    return prognozy

###########
# Funkcje #
###########


def rzutujListeNaTyp(lista, typ):
    # zmienia typ danych dla całej listy a następnie zwraca ją
    return list(map(typ, lista))


def odlegloscEuklidesowa(wektor1, wektor2):
    # oblicza dystans pomiędzy dwoma wektorami danych (im mniejsza odległość tym mniej się różnią)
    dystans = 0.0
    for i in range(len(wektor1)-1): # dla każdej cechy w wektorze utworz sume
        dystans += (wektor1[i] - wektor2[i])**2
    return dystans**(1/2)


def dopasujNajlepszyWektor(dane, wektor_podstawowy):
    # wyszukuje najlepiej dopasowany wektor ze zbioru danych dla podanego wektora podstawowego
    dystanse = list() # zainicjuj pustą listę
    for wektor in dane: # dla kazdego wektora w podanym zbiorze
        dist = odlegloscEuklidesowa(wektor, wektor_podstawowy) # oblicz odległość do podanego wektora
        dystanse.append((wektor, dist)) # i dodaj obliczną wartość do listy
    dystanse.sort(key=lambda tup: tup[1]) # posortuj rosnąco listę z dystansami
    return dystanse[0][0] # wybierz pierwszy element z listy (z najmniejszą odległością)


def losowyWektorZDanych(dane):
    # dla podanego zbioru danych losuje jego losowy wektor
    iloscRekordow = len(dane) # ile rekordow jest w zbiorze
    iloscCech = len(dane[0]) # ile cech maja wektory
    wylosowany = [dane[randrange(iloscRekordow)][i] for i in range(iloscCech)] # wylosuj wektor w przedziale
    return wylosowany


def trenuj(dane, iloscWektorow, wspUczenia, epoki):
    # procedura uczenia zestawu wektorów ze zbioru danych
    wylosowane = [losowyWektorZDanych(dane) for i in range(iloscWektorow)] # wylosowanie z listy zbiorów podanej ilości wektorów
    for epoch in range(epoki): # dla podanej ilości epok
        rate = wspUczenia * (1.0-(epoch/float(epoki))) # aktualny współczynnik uczenia
        sumaErr = 0.0 # utworzenie zmiennej dokładności
        for wektor in dane: # dla każdego wektora w podanym zbiorze danych
            bmu = dopasujNajlepszyWektor(wylosowane, wektor) # dopasuj najlepszy wektor z wylosowanego zbioru danych
            for i in range(len(wektor)-1): # dla każdej cechy
                error = wektor[i] - bmu[i] # oblicz dokładność
                # print(error)
                sumaErr += error**2 # dodaj do sumy dokładności
                if bmu[-1] == wektor[-1]: # jeśli najlepszy wektor dla aktualnego wektora ma taką samą klasę 
                    bmu[i] += rate * error # to dodaj do najlepszego dokładność
                else:
                    bmu[i] -= rate * error # to oddal od najlepszej wartości
        print('>epoka=%d, wsp_uczenia=%.3f, error=%.3f'%(epoch, rate, sumaErr))
    return wylosowane


def prognoza(zestawDanych, wektor):
    # Dokonaj prognozy z wektorami danych z zestawu danych
	bmu = dopasujNajlepszyWektor(zestawDanych, wektor) # wyszukaj z zestawu danych nalepiej dopasowany wektor do podanego 
	return bmu[-1]


def podzielDane(dane, iloscPrzedzialow):
    # Podziel dane na k przedziałów
	podzialDanych = list() # zainicjalizuj pustą liste podzialow
	kopiaDanych = list(dane) # utwórz kopie danych
	szerokoscPrzedzialu = int(len(dane) / iloscPrzedzialow) # ustal szerokosc przedzialu (np. iloscPrzedzialow = 5; => 150/5 = 30 elementowe przedzialy)
	for i in range(iloscPrzedzialow): # dla każdego przedziału
		przedzial = list() # zainicjalizuj pustą listę
		while len(przedzial) < szerokoscPrzedzialu: # dopóki dlugość listy jest mniejsza niż ustalona szerokość przedziału
			indeks = randrange(len(kopiaDanych)) # wylosuj wektor z zestawu danych
			przedzial.append(kopiaDanych.pop(indeks)) # dodaj go do aktualnego przedziału i usuń ze zbioru do losowania
		podzialDanych.append(przedzial) # dodaj przedział do listy przedziałów
	return podzialDanych
 

def wskaznikPrecyzjiDopasowania(wlasciwy, otrzymany):
    # Oblicz procent dokładności
	poprawnie = 0 # zainicjalizuj początkową wartość poprawności danych
	for i in range(len(wlasciwy)): # dla każdego z klasyfikatorów 
		if wlasciwy[i] == otrzymany[i]: # sprawdź czy otrzymany wynik pokrywa się z danymi uczącymi
			poprawnie += 1 # jeśli tak to zwiększ poprawność
	return poprawnie / float(len(wlasciwy)) * 100.0 # zwróć poprawność w procentach
 

def ocenAlgorytm(dane, iloscPrzedzialow, iloscWylosowanychWektorow, wspolczynnikUczenia, iloscEpok):
    # Ocenia algorytm używając podziału walidacji krzyżowej
    przedzialy = podzielDane(dane, iloscPrzedzialow) # utwórz przedziały
    wyniki = list() # zainicjalizuj pustą listę z wynikami
    for przedzial in przedzialy: # dla każdego z utworzonych przedziałów
        daneUczace = list(przedzialy) # utwórz kopie przedziałów
        daneUczace.remove(przedzial) # usuń aktualny przedział
        daneUczace = sum(daneUczace, [])
        daneTestowe = list() # zainicjalizuj pustą listę dla danych testowych
        for wektor in przedzial: # dla każdego wektora w przedziale
            wektorKopia = list(wektor) # utwórz jego kopie
            daneTestowe.append(wektorKopia) # dodaj wektor do danych testowych
            wektorKopia[-1] = None # usuń klasyfikator
        otrzymany = lvq(daneUczace, daneTestowe, iloscWylosowanychWektorow, wspolczynnikUczenia, iloscEpok) # uzyskaj prognozy
        wlasciwy = [wektor[-1] for wektor in przedzial] # wyciągnij klasyfikatory
        precyzja = wskaznikPrecyzjiDopasowania(wlasciwy, otrzymany) # sprawdź poprawność klasyfikacji
        wyniki.append(precyzja) # dodaj wynik procentowy do wyników
    return wyniki

########
# Main #
########

# *** Przygotowanie danych ***
kolumny = [] # tutaj przetrzymywane są całe kolumny zmiennych (czyli dla jednej cechy, ułatwia to wyszukanie minimum oraz maksimum)
for linia in plikZrodlowy:  # dla każdej linii w pliku z danymi
    temp = rzutujListeNaTyp(linia.replace('\n', '').split(';'), float)  # rozdziel ciąg znaków na znakach separacji (;) oraz utwórz z nich wektor liczbowy
    P.append(temp[0:kolumnaT])  # dodaj wektor cech do zbioru P
    T.append(temp[kolumnaT])  # dodaj identyfikator klasy do zbioru T
    i = 0
    for liczba in temp[0:kolumnaT]:  # każdą liczbę w wektorze
        if(len(kolumny) != kolumnaT):
            kolumny.append([liczba])
        else:
            kolumny[i].append(liczba)  # dodaj do kolumny dla odpowiedniej cechy
        i = 1 + i

minP = []
maxP = []
Pn = [0] * len(P)  # zainicjuj pustą liste dla znormalizowanych danych
for kolumna in kolumny:  # dla każdej cechy
    minP.append(min(kolumna))  # znajdź minimum
    maxP.append(max(kolumna))  # oraz maksimum

aktualnyWiersz = 0
for wektor in P:  # dla każdego wektora w zbiorze P
    aktualnaKolumna = 0
    for liczba in wektor:  # dla każdej liczby w wektorze
        znormalizowana = ((1 - (-1)) * (liczba - minP[aktualnaKolumna]) / (maxP[aktualnaKolumna] - minP[aktualnaKolumna]) + (-1))  # znormalizuj ją do przedziału <-1,1>
        if(aktualnaKolumna == 0):
            Pn[aktualnyWiersz] = [round(znormalizowana, 6)]  # utwórz listę dla znormalizowanych danych dla aktualnego wektora
        else:
            Pn[aktualnyWiersz].append(round(znormalizowana, 6))  # dodaj znormalizowane dane dla aktualnego wektora
        aktualnaKolumna = 1 + aktualnaKolumna
    aktualnyWiersz = 1 + aktualnyWiersz
zbiorDanych = [0] * len(Pn)  # inicjalizacja pustej listy zbioru danych
for i in range(len(Pn)):
    zbiorDanych[i] = Pn[i] + [T[i]]  # łączenie znormalizowanego zbioru cech ze zbiorem klasyfikatorów

# *** Ewaluacja algorytmu ***
seed(1)
iloscPrzedzialow = 5
wspolczynnikUczenia = 0.3
iloscEpok = 50
iloscWylosowanychWektorow = 20
wyniki = ocenAlgorytm(zbiorDanych, iloscPrzedzialow, iloscWylosowanychWektorow, wspolczynnikUczenia, iloscEpok)
print('wyniki: %s' % wyniki)
print('Średnia precyzji dopasowania: %.3f%%' % (sum(wyniki)/float(len(wyniki))))
