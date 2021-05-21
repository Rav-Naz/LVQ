from random import seed
from random import randrange
from copy import deepcopy

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import datetime


####################
# Zmienne globalne #
####################

plikZrodlowy = open("iris.data", "r")  # wczytanie do zmiennej pliku z danymi
P = list()  # zbiór wejściowy (tu znajdują się cechy obiektu)
T = list()  # zbiór identyfikatorów (tu znajdują się identyfikatory, po których klasyfikowane są obiekty)
kolumnaT = 4  # indeks kolumny T
zbiorDanych = list()
czyNormalizowacDane = True

################
# Algorytm LVQ #
################


def lvq(daneUczace, daneTestowe, iloscNeuronow, wspUczenia, iloscEpok):
    zestawDanychZmodyfikowanych = forward(daneUczace, iloscNeuronow,
                          wspUczenia, iloscEpok)  # trenuj zestaw danych
    prognozy = list()  # zainicjalizuj pustą listę prognoz
    for wektor in daneTestowe:  # dla każdego wektora w danych testowych
        output = prognoza(zestawDanychZmodyfikowanych, wektor)  # dopasuj klasyfikator
        prognozy.append(output)  # dodaj wynik do listy prognoz
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
    for i in range(len(wektor1)-1):  # dla każdej cechy w wektorze utworz sume
        dystans += (wektor1[i] - wektor2[i])**2
    return dystans**(1/2)


def dopasujNajlepszyWektor(wylosowaneNeurony, wektor_podstawowy):
    # wyszukuje najlepiej dopasowany wektor ze zbioru danych dla podanego wektora podstawowego
    dystanse = list()  # zainicjuj pustą listę
    for wektor in wylosowaneNeurony:  # dla kazdego wektora w podanym zbiorze
        # oblicz odległość do podanego wektora
        dist = odlegloscEuklidesowa(wektor, wektor_podstawowy)
        dystanse.append((wektor, dist))  # i dodaj obliczną wartość do listy
    # posortuj rosnąco listę z dystansami
    dystanse.sort(key=lambda tup: tup[1])
    # wybierz pierwszy element z listy (z najmniejszą odległością)
    return dystanse[0][0]


def losowyWektorZDanych(daneUczace):
    # dla podanego zbioru danych losuje jego losowy wektor
    iloscRekordow = len(daneUczace)  # ile rekordow jest w zbiorze 
    return daneUczace[randrange(iloscRekordow)] # wylosuj wektor w przedziale


def forward(dane, iloscNeuronow, wspUczenia, epoki):
    # procedura uczenia zestawu wektorów ze zbioru danych
    wylosowaneNeurony = [losowyWektorZDanych(dane) for i in range(iloscNeuronow)] # wylosowanie z listy zbiorów podanej ilości neuronow
    for epoch in range(epoki):  # dla podanej ilości epok
        # aktualny współczynnik uczenia
        rate = wspUczenia * (1.0-(epoch/float(epoki)))
        for wektor in dane:  # dla każdego wektora w podanym zbiorze danych
            # dopasuj najlepszy wektor z wylosowanego zbioru neuronów
            bmu = dopasujNajlepszyWektor(wylosowaneNeurony, wektor)
            for i in range(len(wektor)-1):  # dla każdej cechy
                error = wektor[i] - bmu[i]  # oblicz różnicę cech
                if bmu[-1] == wektor[-1]: # jeśli najbliższy neuron dla aktualnego wektora ma taką samą klasę
                    bmu[i] += rate * error # to przybliż neuron do wektora
                else:
                    bmu[i] -= rate * error  # to oddal neuron od wektora
    return wylosowaneNeurony


def prognoza(zestawDanych, wektor):
    # Dokonaj prognozy z wektorami danych z zestawu danych
    # wyszukaj z zestawu danych nalepiej dopasowany wektor do podanego
    bmu = dopasujNajlepszyWektor(zestawDanych, wektor)
    return bmu[-1]


def podzielDane(dane, iloscPrzedzialow):
    # Podziel dane na k przedziałów
    podzialDanych = list()  # zainicjalizuj pustą liste podzialow
    kopiaDanych = list(dane)  # utwórz kopie danych
    # ustal szerokosc przedzialu (np. iloscPrzedzialow = 5; => 150/5 = 30 elementowe przedzialy)
    szerokoscPrzedzialu = int(len(dane) / iloscPrzedzialow)
    for i in range(iloscPrzedzialow):  # dla każdego przedziału
        przedzial = list()  # zainicjalizuj pustą listę
        # dopóki dlugość listy jest mniejsza niż ustalona szerokość przedziału
        while len(przedzial) < szerokoscPrzedzialu:
            # wylosuj wektor z zestawu danych
            indeks = randrange(len(kopiaDanych))
            # dodaj go do aktualnego przedziału i usuń ze zbioru do losowania
            przedzial.append(kopiaDanych.pop(indeks))
        podzialDanych.append(przedzial)  # dodaj przedział do listy przedziałów
    return podzialDanych


def wskaznikPrecyzjiDopasowania(wlasciwy, otrzymany):
    # Oblicz procent dokładności
    blad = 0 # zainicjalizuj początkową wartość błędu średniokwadratowego
    poprawnie = 0  # zainicjalizuj początkową wartość poprawności danych
    for i in range(len(wlasciwy)):  # dla każdego z klasyfikatorów
        blad += (otrzymany[i] - wlasciwy[i])**2 # oblicz różnicę klasyfikatorów 
        if wlasciwy[i] == abs(round(float(otrzymany[i]))): # sprawdź czy otrzymany wynik pokrywa się z danymi uczącymi
            poprawnie += 1  # jeśli tak to zwiększ poprawność
    return [poprawnie / float(len(wlasciwy)) * 100, blad / float(len(wlasciwy))]


def ocenAlgorytm(dane, iloscPrzedzialow, iloscNeuronow, wspolczynnikUczenia, iloscEpok):
    # Ocenia algorytm używając podziału walidacji krzyżowej
    przedzialy = podzielDane(dane, iloscPrzedzialow)  # utwórz przedziały
    wyniki = list()  # zainicjalizuj pustą listę z wynikami procentowymi
    bledy = list()  # zainicjalizuj pustą listę z błędami śreniokwadratowymi
    for przedzial in przedzialy:  # dla każdego z utworzonych przedziałów
        daneUczace = list(przedzialy)  # utwórz kopie przedziałów
        daneUczace.remove(przedzial)  # usuń aktualny przedział
        daneUczace = sum(daneUczace, [])
        daneTestowe = list()  # zainicjalizuj pustą listę dla danych testowych
        for wektor in przedzial:  # dla każdego wektora w przedziale
            wektorKopia = list(wektor)  # utwórz jego kopie
            daneTestowe.append(wektorKopia)  # dodaj wektor do danych testowych
            # wektorKopia[-1] = None # usuń klasyfikator
        otrzymany = lvq(daneUczace, daneTestowe, iloscNeuronow,
                        wspolczynnikUczenia, iloscEpok)  # uzyskaj prognozy
        wlasciwy = [wektor[-1]for wektor in przedzial]  # wyciągnij klasyfikatory
        precyzja = wskaznikPrecyzjiDopasowania(wlasciwy, otrzymany)  # sprawdź poprawność klasyfikacji
        wyniki.append(precyzja[0])  # dodaj wynik procentowy do wyników
        bledy.append(precyzja[1])  # dodaj wynik procentowy do błędów
    return [wyniki,bledy]


def rysujGraf(x,y,PK, MSE):
    x, y = np.meshgrid(x, y) # utwórz przestrzeń XY
    fig, (ax, bx) = plt.subplots(1,2, subplot_kw=dict(projection='3d')) # zainicjalizuj osie
    ls = LightSource(270,45) # ustaw kąt padania światła
    rgb1 = ls.shade(PK, cmap=cm.brg, vert_exag=0.1, blend_mode='soft') # określ kolory wykresu
    rgb2 = ls.shade(MSE, cmap=cm.summer, vert_exag=0.1, blend_mode='soft') # określ kolory wykresu
    ax.plot_surface(x, y, PK, rstride=1, cstride=1, facecolors=rgb1,
                        linewidth=0, antialiased=False, shade=False) # utwórz wykres
    bx.plot_surface(x, y, MSE, rstride=1, cstride=1, facecolors=rgb2,
                        linewidth=0, antialiased=False, shade=False)
    ax.view_init(30, -135) # ustaw początkowy punkt widzenia
    bx.view_init(30, -135) # ustaw początkowy punkt widzenia
    ax.set_xlabel('współczynnik uczenia') # tytuł osi x wykresu PK
    ax.set_ylabel('liczba neuronów') # tytuł osi y wykresu PK
    ax.set_zlabel('poprawność klasyfikacji') # tytuł osi z wykresu PK
    bx.set_xlabel('współczynnik uczenia') # tytuł osi x wykresu MSE
    bx.set_ylabel('liczba neuronów') # tytuł osi y wykresu MSE
    bx.set_zlabel('błąd średniokwadratowy') # tytuł osi z wykresu MSE
    ax.set_title('PK(S1, lr)') # tytuł wykresu PK
    bx.set_title('MSE(S1, lr)') # tytuł wykresu MSE
    ax.set_zlim(0, 100) #  określ przedział osi Z dla PK od 0 do 100
    ax.zaxis.set_major_formatter('{x:.00f}%')
    plt.show() # wyświetl graf

########
# Main #
########


# *** Przygotowanie danych ***
# tutaj przetrzymywane są całe kolumny zmiennych (czyli dla jednej cechy, ułatwia to wyszukanie minimum oraz maksimum)
kolumny = []
for linia in plikZrodlowy:  # dla każdej linii w pliku z danymi
    # rozdziel ciąg znaków na znakach separacji (;) oraz utwórz z nich wektor liczbowy
    temp = rzutujListeNaTyp(linia.replace('\n', '').split(';'), float)
    P.append(temp[0:kolumnaT])  # dodaj wektor cech do zbioru P
    T.append(temp[kolumnaT])  # dodaj identyfikator klasy do zbioru T
    i = 0
    for liczba in temp[0:kolumnaT]:  # każdą liczbę w wektorze
        if(len(kolumny) != kolumnaT):
            kolumny.append([liczba])
        else:
            # dodaj do kolumny dla odpowiedniej cechy
            kolumny[i].append(liczba)
        i = 1 + i

minP = []
maxP = []
Pn = [0] * len(P)  # zainicjuj pustą liste dla znormalizowanych danych
for kolumna in kolumny:  # dla każdej cechy
    minP.append(min(kolumna))  # znajdź minimum
    maxP.append(max(kolumna))  # oraz maksimum

if czyNormalizowacDane:
    aktualnyWiersz = 0
    for wektor in P:  # dla każdego wektora w zbiorze P
        aktualnaKolumna = 0
        for liczba in wektor:  # dla każdej liczby w wektorze
            znormalizowana = ((1 - (-1)) * (liczba - minP[aktualnaKolumna]) / (
                maxP[aktualnaKolumna] - minP[aktualnaKolumna]) + (-1))  # znormalizuj ją do przedziału <-1,1>
            if(aktualnaKolumna == 0):
                # utwórz listę dla znormalizowanych danych dla aktualnego wektora
                Pn[aktualnyWiersz] = [round(znormalizowana, 6)]
            else:
                # dodaj znormalizowane dane dla aktualnego wektora
                Pn[aktualnyWiersz].append(round(znormalizowana, 6))
            aktualnaKolumna = 1 + aktualnaKolumna
        aktualnyWiersz = 1 + aktualnyWiersz
zbiorDanych = [0] * len(Pn)  # inicjalizacja pustej listy zbioru danych
for i in range(len(Pn)):
    # łączenie znormalizowanego zbioru cech ze zbiorem klasyfikatorów
    zbiorDanych[i] = (Pn[i] if czyNormalizowacDane else P[i]) + [T[i]]


# *** Ewaluacja algorytmu ***
seed(1)
iloscPrzedzialow = 5 # ilość przedziałów na które zostanie podzielony zbiór danych (potrzebne do walidacji krzyżowej)
wspolczynnikUczenia = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] # lista współczynników uczenia do przetestowania
iloscEpok = 2 # ilość powtórzeń przez które uczone są dane
iloscNeuronow = range(10, 51, 10) # lista ilości neuronów do przetestowania

najlepszeKombinacjePK = list() # inicjalizacja listy z najlepszymi wynikami
najlepszeKombinacjeMSE = list() # inicjalizacja listy z najlepszymi wynikami
najlepszaDokladnosc = float('-inf') # zmienna do określenia najlepszej precyzji dopasowania danych
najmniejszyBlad = float('inf') # zmienna do określenia najmniejszego błedu średniokwardatowego

zPK = list() # zmienna przetrzymująca obliczone parametry dla wykresu PK
zMSE = list() # zmienna przetrzymująca obliczone parametry dla wykresu MSE
start = datetime.datetime.now() # rozpoczęcie pomiaru czasu wykonania skryptu
for neuron in iloscNeuronow: # dla każdej ilości neuronów w liście
    wierszProcentow = list() # zainicjalizuj pusty wektor dla obliczonych danych
    wierszBledow = list() # zainicjalizuj pusty wektor dla obliczonych danych
    for lr in wspolczynnikUczenia: # dla każdego współczynnika uczenia
        poczatek = datetime.datetime.now() # rozpocznij pomiar czasu dla danej konfiguracji
        copy = deepcopy(zbiorDanych) # utwórz kopię zbioru danych
        wyniki = ocenAlgorytm(copy, iloscPrzedzialow, neuron, lr, iloscEpok) # uzyskaj wyniki poprawności klasyfikacji dla każdego przedziału
        koniec = datetime.datetime.now() # zakończ pomiar czasu dla danej konfiguracji
        czasWykonania = round((koniec - poczatek).total_seconds() * 1000) # oblicz czas obliczania danej konfiguracji
        sredniaProcentowa = round((sum(wyniki[0])/float(len(wyniki[0]))),5) # wyciągnij średnią z uzyskanych wyników poprawności klasyfikacji
        sredniaBledu = round((sum(wyniki[1])/float(len(wyniki[1]))),5) # wyciągnij średnią z uzyskanych wyników błędu śreniokwadratowego (MSE)
        wierszProcentow.append(sredniaProcentowa) # do wektora obliczonych srednich dodaj aktualną wartość PK
        wierszBledow.append(sredniaBledu) # do wektora obliczonych srednich dodaj aktualną wartość MSE
        print('S1:', neuron, ', lr:', lr, ', {0: .3f}%'.format(sredniaProcentowa), ', MSE {0: .3f}'.format(sredniaBledu), ', czas wykonania: ', czasWykonania, 'ms') # wypisz wyniki aktualnej konfiguracji
        if sredniaProcentowa > najlepszaDokladnosc: # jeśli aktualny wynik PK jest lepszy od poprzeniego
            najlepszeKombinacjePK = list() # wyczyść liste najlepszych konfiguracji PK
            najlepszaDokladnosc = sredniaProcentowa # przypisz nową najlepszą wartość
            najlepszeKombinacjePK.append([neuron,lr, sredniaProcentowa, sredniaBledu, czasWykonania]) # dodaj aktualną konfigurację do listy najlepszych konfiguracji PK
        elif len(najlepszeKombinacjePK) > 0 and sredniaProcentowa == najlepszaDokladnosc: # jeśli wynik jest tak samo dobry
            najlepszeKombinacjePK.append([neuron,lr, sredniaProcentowa, sredniaBledu, czasWykonania]) # dodaj aktualną konfigurację do listy najlepszych konfiguracji PK
        
        if sredniaBledu < najmniejszyBlad: # jeśli aktualny wynik MSE jest lepszy od poprzeniego
            najlepszeKombinacjeMSE = list() # wyczyść liste najlepszych konfiguracji MSE
            najmniejszyBlad = sredniaBledu # przypisz nową najlepszą wartość
            najlepszeKombinacjeMSE.append([neuron,lr, sredniaProcentowa, sredniaBledu, czasWykonania]) # dodaj aktualną konfigurację do listy najlepszych konfiguracji MSE
        elif len(najlepszeKombinacjeMSE) > 0 and sredniaBledu == najmniejszyBlad: # jeśli wynik jest tak samo dobry
            najlepszeKombinacjeMSE.append([neuron,lr, sredniaProcentowa, sredniaBledu, czasWykonania]) # dodaj aktualną konfigurację do listy najlepszych konfiguracji MSE

    zPK.append(wierszProcentow) # dodaj wektor obliczonych danych do listy dla wykresu PK
    zMSE.append(wierszBledow) # dodaj wektor obliczonych danych do listy dla wykresu MSE
stop = datetime.datetime.now() # zakończ pomiar czasu wykonania skryptu
czasWykonania = round((stop - start).total_seconds() * 1000) # oblicz czas wykonywania skryptu

# *** Podsumowanie wyników ***
print('Czas wykonania skryptu: ', czasWykonania, 'ms') # wypisz czas wykonywania skryptu

najlepszeKombinacjePK.sort(key=lambda tup: tup[4]) # posortuj nalepsze kombinacje PK względem czasu wykonania
najlepszeKombinacjeMSE.sort(key=lambda tup: tup[4]) # posortuj nalepsze kombinacje MSE względem czasu wykonania

print('najlepsze PK', najlepszeKombinacjePK) # wyświetl najlepsze kofiguracje PK
print('najlepsze MSE', najlepszeKombinacjeMSE) # wyświetl najlepsze kofiguracje MSE


zPK = np.array(zPK) # oś Z wykresu PK
zMSE = np.array(zMSE) # oś Z wykresu PK
x = np.array(wspolczynnikUczenia) # oś X wykresów
y = np.array(iloscNeuronow) # oś Y wykresów

rysujGraf(x,y,zPK,zMSE) # narysuj wykres powierzchniowy 3D