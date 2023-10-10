import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Ouvrir un fichier texte en mode lecture
nom_fichier = "/Users/bagafoufabrice/Downloads/Pb1.txt"

try:
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()

except FileNotFoundError:
    print(f"Le fichier {nom_fichier} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite : {str(e)}")

villes = []
for i in range(1,len(lignes)-1):
    a = lignes[i].split(' ')
    a[1] = int(a[1][:-1])
    a[0] = int(a[0])
    villes.append(a)

num_villes = len(villes)


#Calcul de toutes les distances (pas a les recalculer a chaque fois)
def calcul_distance(ville1, ville2):
    x1, y1 = ville1
    x2, y2 = ville2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

distances = [[calcul_distance(villes[i], villes[j]) for j in range(num_villes)] for i in range(num_villes)]

#print(villes)

# Générer des données aléatoires en 2D
''''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(villes)  # Remplacez X par vos données
    silhouette_avg = silhouette_score(villes, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

best_k = silhouette_scores.index(max(silhouette_scores))
print('best k =',best_k)

# Créer un objet KMeans avec 2 clusters
kmeans = KMeans(n_clusters=15)

# Effectuer le clustering sur les données
kmeans.fit(villes)

# Obtenir les centres de cluster
centroids = kmeans.cluster_centers_

# Obtenir les étiquettes de cluster pour chaque point
labels = kmeans.labels_

x, y = zip(*villes)

# Afficher les points de données colorés en fonction de leur cluster
plt.scatter(x, y, c=labels)
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.legend()
plt.title('Clustering avec K-Means')
plt.show()

'''
max_dist = 250
#python Current_research.py
parcour_final =[]

def recherche_chemin(parcours,distance,i,ville_restantes,distance_max_globale):
    global num_villes
    global max_dist
    global parcour_final
    #print(distance_max_globale)
    if len(parcours) == num_villes:
        #print(max(distance))
        print('une solution',sum(distance))
        max_dist = max(distance)
        parcour_final = parcours
        print(parcours)
        return True
    if len(parcours) != num_villes :
        for j in ville_restantes:
            if distances[i][j]<distance_max_globale :

                distance.append(distances[i][j])
                parcours.append(j)
                ville_restantes.remove(j)
                if recherche_chemin(parcours,distance,j,ville_restantes,distance_max_globale):
                    return True
                distance.remove(distances[i][j])
                parcours.remove(j)
                ville_restantes.append(j)
            #elif distances[i][j] >= distance_max_globale :
                #continue

    return False
distance = []


#recherche_chemin([],[],0,[i for i in range(num_villes)],max_dist)



def plot_resul(data):
    x, y = zip(*data)

    # Tracez les points
    plt.plot(x, y, marker='o', linestyle='-')

    # Ajoutez des étiquettes aux axes
    plt.xlabel('Axe X')
    plt.ylabel('Axe Y')

    # Affichez le graphe
    plt.show()



def recherche_chemin_distance_totale(parcours,distance,i,ville_restantes,distance_max_globale):
    global num_villes
    global max_dist
    global parcour_final
    global distances
    #print('nombre de distance', len(distance))
    #print(distance_max_globale)
    if len(parcours) == num_villes:
        #print(max(distance))
        print('une solution',sum(distance))
        print('nombre de distance',len(distance))
        max_dist = sum(distance)
        parcour_final = parcours
        #print(parcours)
        return True
    if len(parcours) != num_villes :
        for j in ville_restantes:
            #print(sum(distance),distance_max_globale)
            #print('distance',distances[i][j],i,j)
            if (sum(distance)+distances[i][j])<distance_max_globale :
                print(len(distance))
                distance.append(distances[i][j])
                parcours.append(j)
                ville_restantes.remove(j)
                if recherche_chemin_distance_totale(parcours,distance,j,ville_restantes,distance_max_globale):
                    return True
                distance.remove(distances[i][j])
                parcours.remove(j)
                ville_restantes.append(j)
            #elif distances[i][j] >= distance_max_globale :
                #continue

    return False

compt = 0
#print('max_dist',max_dist)

while True :
    print('max dist', max_dist)
    depart = 0
    liste_villes_restantes = [i for i in range(num_villes)]
    liste_villes_restantes.remove(0)
    #print(liste_villes_restantes)
    recherche_chemin_distance_totale([depart], [], depart,liste_villes_restantes,  max_dist)
    compt+=1
    print(compt)
    if compt == 1000:
        break
    #print(max_dist)

ville_ordonne = []
for i in parcour_final:
    ville_ordonne.append(villes[i])

plot_resul(ville_ordonne)