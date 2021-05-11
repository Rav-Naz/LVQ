####################
# Zmienne globalne #
####################

plikZrodlowy = open("iris.data", "r") # wczytanie do zmiennej pliku z danymi
P = [] # zbiór wejściowy (tu znajdują się cechy obiektu)
T = [] # zbiór identyfikatorów (tu znajdują się identyfikatory, po których klasyfikowane są obiekty)
kolumnaT = 4 # indeks kolumny T

###########
# Funkcje #
###########

def rzutujListeNaTyp(lista, typ):
    # zmienia typ danych dla całej listy a następnie zwraca ją
    return list(map(typ, lista))

################
# Algorytm LVQ #
################

# *** Przygotowanie danych ***

kolumny = [] # tutaj przetrzymywane są całe kolumny zmiennych (czyli dla jednej cechy, ułatwia to wyszukanie minimum oraz maksimum)
for linia in plikZrodlowy: # dla każdej linii w pliku z danymi
    temp = rzutujListeNaTyp(linia.replace('\n', '').split(';'), float) # rozdziel ciąg znaków na znakach separacji (;) oraz utwórz z nich wektor liczbowy
    P.append(temp[0:kolumnaT]) # dodaj wektor cech do zbioru P
    T.append(temp[kolumnaT]) # dodaj identyfikator klasy do zbioru T
    i = 0
    for liczba in temp[0:kolumnaT]: # każdą liczbę w wektorze
        if(len(kolumny) != kolumnaT):
            kolumny.append([liczba]) 
        else:
            kolumny[i].append(liczba) # dodaj do kolumny dla odpowiedniej cechy
        i = 1 + i

minP = []
maxP = []
Pn = [0] * len(P) # zainicjuj pustą liste dla znormalizowanych danych
for kolumna in kolumny: # dla każdej cechy
    minP.append(min(kolumna)) # znajdź minimum
    maxP.append(max(kolumna)) # oraz maksimum

aktualnyWiersz = 0
for wektor in P: # dla każdego wektora w zbiorze P
    aktualnaKolumna = 0
    for liczba in wektor: # dla każdej liczby w wektorze
        znormalizowana = ((1 - (-1)) * (liczba - minP[aktualnaKolumna]) / (maxP[aktualnaKolumna] - minP[aktualnaKolumna]) + (-1)) # znormalizuj ją do przedziału <-1,1>
        if(aktualnaKolumna == 0):
            Pn[aktualnyWiersz] = [znormalizowana] # utwórz listę dla znormalizowanych danych dla aktualnego wektora
        else:
            Pn[aktualnyWiersz].append(znormalizowana) # dodaj znormalizowane dane dla aktualnego wektora
        aktualnaKolumna = 1 + aktualnaKolumna
    aktualnyWiersz = 1 + aktualnyWiersz
print(Pn)


