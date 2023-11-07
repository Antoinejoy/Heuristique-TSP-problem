import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
def permutation(config,i,j):
    config_inter = config.copy()
    config_inter[i] = config[j]
    config_inter[j] = config[i]
    return config_inter
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
        #print(parcours)
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
                #print(len(distance))
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
def plusproche_voisin(donne,depart,point_restant):
    min = 10000
    for i in range(len(donne)):
        if i != depart and i in point_restant :
            dist = calcul_distance(donne[depart],donne[i])
            if dist<min :
                min=dist
                indice = i
    return min,indice
def glouton(donne):
    depart = int(np.random.rand()*len(donne)//1)
    #calcul plus proche voisin
    prems = depart
    liste_entiers = list(range(1000))
    liste_entiers.remove(depart)
    compt = 0
    config = [donne[depart]]
    distance = 0
    while compt<len(donne)-1:
        min,indice = plusproche_voisin(donne,depart,liste_entiers)
        #print(min)
        distance += min
        compt += 1
        liste_entiers.remove(indice)
        depart = indice
        config.append(donne[indice])
    distance += calcul_distance(donne[prems],donne[depart])
    #print(distance)
    return config,distance
def glouton_depart_arrive(donne,depart,arrive):
    liste_entiers = list(range(len(donne)))
    liste_entiers.remove(depart)
    liste_entiers.remove(arrive)
    compt = 0
    config = [donne[depart]]
    distance = 0
    while compt < len(donne) - 2:
        min, indice = plusproche_voisin(donne, depart, liste_entiers)
        distance += min
        compt += 1
        liste_entiers.remove(indice)
        depart = indice
        config.append(donne[indice])

    #distance += calcul_distance(donne[prems], donne[depart])

    # Assurez-vous que la dernière étape est le point d'arrivée spécifié
    config.append(donne[arrive])
    #distance += calcul_distance(donne[depart], donne[arrive])
    return config,distance
def swap_descente_complete(solution,distance):
    compt = 0
    #print(len(solution))
    while compt < len(solution):
        for i in range(len(solution)):
            #print(i)
            if i != compt :
                #permutation
                config_tempo = permutation(solution, compt, i)
                nouvelle_distance=0
                for i in range(len(config_tempo)-1):
                    nouvelle_distance+=calcul_distance(config_tempo[i],config_tempo[i+1])
                nouvelle_distance += calcul_distance(config_tempo[len(config_tempo)-1], config_tempo[0])
                #Si la solution est la bonne
                if nouvelle_distance < distance :
                    #print(nouvelle_distance)
                    compt = 0
                    solution = config_tempo
                    distance = nouvelle_distance
                    break
        compt+=1
        #print(compt)
    print('distance finale : ',distance)
    #print('solution finale : ', solution)
    return distance,solution
def swap_descente_complete_garder_premier_dernier(solution, distance):
    compt = 1
    # print(len(solution))
    while compt < len(solution) - 1:
        for i in range(len(solution) - 1):
            # print(i)
            if i != compt and i != compt - 1:
                # permutation
                config_tempo = permutation(solution, compt, i)
                nouvelle_distance = 0
                for i in range(len(config_tempo) - 1):
                    nouvelle_distance += calcul_distance(config_tempo[i], config_tempo[i + 1])
                nouvelle_distance += calcul_distance(config_tempo[len(config_tempo) - 1], config_tempo[0])
                # Si la solution est la bonne
                if nouvelle_distance < distance:
                    # print(nouvelle_distance)
                    compt = 1
                    solution = config_tempo
                    distance = nouvelle_distance
                    break
        compt += 1
        # print(compt)
    print('distance finale : ', distance)
    # print('solution finale : ', solution)
    return distance, solution
def swap_descente_best_sol(solution,distance):
    compt = 1
    #print(len(solution))
    best_distance_temp = 100000
    best_config = solution
    while compt < len(solution)-1:
        for i in range(1,len(solution)-1):
            #print(i)
            if i != compt :
                nouvelle_distance = distance

                #on enleve la distance des anciens
                if compt==len(solution)-1:
                    dist1_av = calcul_distance(solution[compt], solution[0])
                if i == len(solution)-1 :
                    dist3_av = calcul_distance(solution[i], solution[0])
                if compt < len(solution)-1 :
                    dist1_av = calcul_distance(solution[compt],solution[compt + 1])
                if i < len(solution)-1 :
                    dist3_av = calcul_distance(solution[i], solution[i+1])

                dist2_av = calcul_distance(solution[compt], solution[compt - 1])
                dist4_av = calcul_distance(solution[i], solution[i - 1])

                somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

                #permutation
                config_tempo = permutation(solution, compt, i)

                #ajout des nouvelles
                if compt==len(solution)-1:
                    dist1 = calcul_distance(config_tempo[compt], config_tempo[0])
                if i == len(solution)-1:
                    dist3 = calcul_distance(config_tempo[i], config_tempo[0])
                if compt<len(solution)-1 :
                    dist1 = calcul_distance(config_tempo[compt],config_tempo[compt + 1])
                if i<len(solution)-1 :
                    dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])

                dist2 = calcul_distance(config_tempo[compt], config_tempo[compt - 1])
                dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

                somme_des_dists_2 = dist1 + dist2 + dist3 + dist4
                #print('swap compt,i : ',compt,i)
                #print('delta',somme_des_dists_1,somme_des_dists_2)

                nouvelle_distance = nouvelle_distance - somme_des_dists_1 + somme_des_dists_2
                #print('nouvelle distance',nouvelle_distance)
                #Si la solution est la bonne
                if nouvelle_distance < best_distance_temp :
                    #print(nouvelle_distance)
                    best_config = config_tempo
                    best_distance_temp = nouvelle_distance
                    compt = 1
                    break
        compt += 1
        #print(compt)
        if compt == len(solution)-1:
            if best_distance_temp == distance :
                #permutation aléatoire
                ran = np.random.rand(2)*len(solution)//1
                x = int(ran[0])
                y = int(ran[1])
                solution = permutation(solution,x,y)
                compt = len(solution)
                #compt = 0
                break

            compt = 1
            distance = best_distance_temp
            print('nouvelle distance', best_distance_temp)
            solution = best_config

    print('distance finale : ', distance)
    #print('solution finale : ', solution)
    return distance,solution
def plot_resul(data):
    x, y = zip(*data)

    # Tracez les points
    plt.plot(x, y, marker='o', linestyle='-')

    # Ajoutez des étiquettes aux axes
    plt.xlabel('Axe X')
    plt.ylabel('Axe Y')

    # Affichez le graphe
    plt.show()
def cherche_point(liste,point):
    index_trouve = None
    for index, element in enumerate(liste):
        if element == point:
            index_trouve = index
            break
    return index_trouve
def swap_premier_dernier(solution,distance):
    compt = 1
    #print(len(solution))

    best_config = solution
    while compt < len(solution)-1:
        for i in range(1,len(solution)-1):
            #print(i)
            if i != compt :
                #on enleve la distance des anciens

                dist1_av = calcul_distance(solution[compt],solution[compt + 1])
                dist3_av = calcul_distance(solution[i], solution[i+1])
                dist2_av = calcul_distance(solution[compt], solution[compt - 1])
                dist4_av = calcul_distance(solution[i], solution[i - 1])

                somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

                #permutation
                config_tempo = permutation(solution, compt, i)

                #ajout des nouvelles

                dist1 = calcul_distance(config_tempo[compt],config_tempo[compt + 1])
                dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])
                dist2 = calcul_distance(config_tempo[compt], config_tempo[compt - 1])
                dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

                somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

                nouvelle_distance = distance - somme_des_dists_1 + somme_des_dists_2

                #Si la solution est la bonne
                if nouvelle_distance < distance :
                    solution = config_tempo
                    distance = nouvelle_distance
                    compt = 1
                    break
        compt += 1

    return distance,solution
def Depart_arrive(grp_av,groupe_apr):
    distance_min = float('inf')
    point1_plus_proche = None
    point2_plus_proche = None
    for point1 in grp_av:
        for point2 in groupe_apr:
            dist = calcul_distance(point1, point2)
            if dist < distance_min:
                distance_min = dist
                point1_plus_proche = point1
                point2_plus_proche = point2
    return [point1_plus_proche,point2_plus_proche]
def calcul_distance(ville1, ville2):
    x1, y1 = ville1
    x2, y2 = ville2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Ouvrir un fichier texte en mode lecture
nom_fichier = "/Users/bagafoufabrice/Downloads/Pb3.txt"

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
distances = [[calcul_distance(villes[i], villes[j]) for j in range(num_villes)] for i in range(num_villes)]

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(villes)  # Remplacez X par vos données
    silhouette_avg = silhouette_score(villes, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

best_k = silhouette_scores.index(max(silhouette_scores))
print('best k =',best_k)

N_cluster = len(villes)//20
N_cluster = 10
# Créer un objet KMeans
kmeans = KMeans(n_clusters=N_cluster)

# Effectuer le clustering sur les données
kmeans.fit(villes)

# Obtenir les centres de cluster
centroids = kmeans.cluster_centers_

# Obtenir les étiquettes de cluster pour chaque point
labels = kmeans.labels_

x, y = zip(*villes)

# Afficher les points de données colorés en fonction de leur cluster
plt.scatter(x, y, c=labels)
plt.legend()
plt.title('Clustering avec K-Means')
#plt.show()

#creation groupe de points
groupe = [[] for i in range(N_cluster)]
for i in range(len(labels)) :
    #print(labels[i])
    L = labels[i]
    groupe[L].append(villes[i])
    #print(groupe[labels[i]])
#print(groupe)
compt = 0
for i in range(len(groupe)):
    print('longueur des groupes = ',len(groupe[i]))
    compt+=len(groupe[i])
print('longueur total = ',compt)
#print('centre =',len(centroids ))

dist_data = []

# Ordre de reliance des clusters
for i in range(10):
    sol, dist = glouton(centroids)
    if i==0:
        min = dist
        sol_opti = sol
    elif i!=0:
        if dist<min:
            min = dist
            sol_opti = sol
    dist = round(dist,2)
    dist_data.append(dist)
    #print(dist)
print(min)

#faire plusieurs swap pour trouver la meilleure sol
'''for i in range (N_cluster):
    end_distance,end_solution = swap_descente_complete(sol_opti,min)
    if i==0 :
        new_min = end_distance
        new_sol = end_solution
        sol_opti = end_solution
        min = end_distance
    elif i!= 0 :
        if end_distance<new_min :
            new_min=end_distance
            new_sol=end_solution
            sol_opti = end_solution
            min = end_distance

end_solution = new_min
end_distance = new_min'''

end_distance,end_solution = swap_descente_complete(sol_opti,min)

#ordre
'''groupe_ordonne = groupe.copy()
for i in range(len(groupe)):
    valeur = centroids[i]
    for indice, sous_liste in enumerate(end_solution):
        print('indice et sous liste',indice,sous_liste)
        #print((sous_liste == valeur)[0])
        if (sous_liste == valeur)[0] and (sous_liste == valeur)[1]:
            print('PERMUTATION',i,indice)
            groupe_ordonne = permutation(groupe,i,indice)
print('BONNE PERMUTAION',groupe != groupe_ordonne)'''

#ordre2

groupe_ordonne = [0]*len(groupe)
end_sol_temp = [list(arr) for arr in end_solution]
#print(len(end_sol_temp))
#print('centre :' ,centroids[0]==end_sol_temp[0])
for i in range(len(end_sol_temp)):
    centre = centroids[i]
    for indice, sous_liste in enumerate(end_sol_temp):
        # print((sous_liste == valeur)[0])
        if (sous_liste == centre)[0] and (sous_liste == centre)[1]:
            #print('True')
            index = indice
    #index = cherche_point(end_sol_temp,centre)
    groupe_ordonne[index] = groupe[i]


plot_resul(end_solution)
#print('GROUPE ',groupe_ordonne)

#txouver les deux points[arrive,depart] de chaque groupe
liste_depart_arrive = []
for i in range(len(groupe_ordonne)-1):
    liste_inter = Depart_arrive(groupe_ordonne[i],groupe_ordonne[i+1])
    liste_depart_arrive.append(liste_inter)
liste_depart_arrive.append(Depart_arrive(groupe_ordonne[len(groupe_ordonne)-1],groupe_ordonne[0]))

print('liste depart arrive', liste_depart_arrive)

#Étude des groupes
dist_total = 0
nombres_villes1 = 0
nombres_villes2 = 0
Sol_gp = []
print('nombre de villes du groupe ',sum([len(i) for i in groupe_ordonne]))
for i in range(len(groupe_ordonne)):
    depart = liste_depart_arrive[i-1][1]
    arrive = liste_depart_arrive[i][0]
    depart = cherche_point(groupe_ordonne[i],depart)
    arrive = cherche_point(groupe_ordonne[i],arrive)
    sol_gp1, dist_gp1 = glouton_depart_arrive(groupe_ordonne[i],depart,arrive)
    nombres_villes2 += len(sol_gp1)
    #print(sol_gp1[0],sol_gp1[-1])
    end_distance_gp1, end_solution_gp1 = swap_premier_dernier(sol_gp1, dist_gp1)
    nombres_villes1+=len(end_solution_gp1)
    dist_total+=end_distance_gp1
    #print(end_solution_gp1[0], end_solution_gp1[-1])
    Sol_gp = Sol_gp + end_solution_gp1
print('final distance',dist_total)
print(len(Sol_gp))
plot_resul(Sol_gp)

#plot_resul(ville_ordonne)