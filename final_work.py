import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def permutation(config, i, j):
    """
    :param config: la liste a permuter
    :param i: indice de permutation
    :param j: indice de permutation
    :return: la liste premuté
    """

    config_inter = config.copy()
    config_inter[i] = config[j]
    config_inter[j] = config[i]
    return config_inter


def calcul_distance(ville1, ville2):
    # print('ville 1 :',ville1)
    """
    :param ville1:
    :param ville2:
    :return: distance euclidienne entre deux villes
    """

    x1, y1 = ville1
    x2, y2 = ville2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def plusproche_voisin(donne, depart, point_restant):
    """
    :param donne: vecteurs des coordonnées des villes
    :param depart: point de recherche
    :param point_restant: indices des villes pas encore visitées
    :return: la distance entre les deux points,indice du plus proche voisin
    """

    min = 10000
    for i in range(len(donne)):
        if i != depart and i in point_restant:
            dist = calcul_distance(donne[depart], donne[i])
            if dist < min:
                min = dist
                indice = i

    return min, indice


def glouton(donne):
    depart = int(np.random.rand() * len(donne) // 1)
    # calcul plus proche voisin
    prems = depart
    liste_entiers = list(range(1000))
    liste_entiers.remove(depart)
    compt = 0
    config = [donne[depart]]
    distance = 0
    while compt < len(donne) - 1:
        min, indice = plusproche_voisin(donne, depart, liste_entiers)
        # print(min)
        distance += min
        compt += 1
        liste_entiers.remove(indice)
        depart = indice
        config.append(donne[indice])
    distance += calcul_distance(donne[prems], donne[depart])
    # print(distance)
    return config, distance


def glouton_depart_arrive(donne, depart, arrive):
    """
    :param donne: vecteurs des villes
    :param depart: point de depart du glouton
    :param arrive: point d'arrivé du glouton
    :return: la solution, la distance totale
    """

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

    config.append(donne[arrive])
    return config, distance


def swap_descente_complete(solution, distance):
    compt = 0
    while compt < len(solution):
        for i in range(len(solution)):
            # print(i)
            if i != compt:
                # permutation
                config_tempo = permutation(solution, compt, i)
                nouvelle_distance = 0
                for i in range(len(config_tempo) - 1):
                    nouvelle_distance += calcul_distance(
                        config_tempo[i], config_tempo[i + 1]
                    )
                nouvelle_distance += calcul_distance(
                    config_tempo[len(config_tempo) - 1], config_tempo[0]
                )
                # Si la solution est la bonne
                if nouvelle_distance < distance:
                    # print(nouvelle_distance)
                    compt = 0
                    solution = config_tempo
                    distance = nouvelle_distance
                    break
        compt += 1
    return distance, solution


def swap_premier_dernier(solution, distance):
    """
    :param solution: vecteur de donnés
    :param distance: distance total de la solution
    :return: solution & distance total de la solution
    """
    compt = 1
    swap_reussi = False
    while compt < len(solution) - 1:
        for i in range(1, len(solution) - 1):
            if i != compt:
                # on enleve la distance des anciens
                dist1_av = calcul_distance(solution[compt], solution[compt + 1])
                dist3_av = calcul_distance(solution[i], solution[i + 1])
                dist2_av = calcul_distance(solution[compt], solution[compt - 1])
                dist4_av = calcul_distance(solution[i], solution[i - 1])

                somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

                # permutation
                config_tempo = permutation(solution, compt, i)

                # ajout des nouvelles
                dist1 = calcul_distance(config_tempo[compt], config_tempo[compt + 1])
                dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])
                dist2 = calcul_distance(config_tempo[compt], config_tempo[compt - 1])
                dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

                somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

                nouvelle_distance = distance - somme_des_dists_1 + somme_des_dists_2

                # Si la solution est la bonne
                if nouvelle_distance < distance:
                    # print('swap reussi, progression : ',distance-nouvelle_distance)
                    solution = config_tempo
                    distance = nouvelle_distance
                    swap_reussi = True
                    compt = 1

        compt += 1
        if compt == (len(solution) - 1) and not swap_reussi:
            ran = np.random.rand(2) * (len(solution) - 2) // 1 + 1
            x = int(ran[0])
            y = int(ran[1])
            dist1_av = calcul_distance(solution[x], solution[x + 1])
            dist3_av = calcul_distance(solution[y], solution[y + 1])
            dist2_av = calcul_distance(solution[x], solution[x - 1])
            dist4_av = calcul_distance(solution[y], solution[y - 1])
            somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

            solution = permutation(solution, x, y)

            # ajout des nouvelles
            dist1 = calcul_distance(config_tempo[x], config_tempo[x + 1])
            dist3 = calcul_distance(config_tempo[y], config_tempo[y + 1])
            dist2 = calcul_distance(config_tempo[x], config_tempo[x - 1])
            dist4 = calcul_distance(config_tempo[y], config_tempo[y - 1])

            somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

            distance = distance - somme_des_dists_1 + somme_des_dists_2
            swap_reussi = True
            compt = 1

    return distance, solution


def plot_resul(data, labels, inital, liste_solution):
    """
    :param data: donnée a plotter
    :return: plot de la donnée relié
    """
    x, y = zip(*data)
    x_init, y_init = zip(*inital)
    rd = int(random.uniform(0, len(liste_solution) - 1))
    rd1 = int(random.uniform(0, len(liste_solution) - 1))
    # Tracez les points
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(x, y, marker="o", linestyle="-")
    ax[0][1].scatter(x_init, y_init, c=labels)
    ax[0][1].legend()
    x1, y1 = zip(*liste_solution[rd1])
    ax[1][0].plot(x1, y1, marker="o", linestyle="-")
    # exemple de plot
    x2, y2 = zip(*liste_solution[rd])
    ax[1][1].plot(x2, y2, marker="o", linestyle="-")
    # Ajoutez des étiquettes aux axes
    # Affichez le graphe
    plt.show()


def import_fichier(num):
    """
    :param num: numero du fichier pb a telecharger
    :return num_villes : nombre de villes
            villes : vecteur de coordonnées des villes
            distances : matrice des distances entre deux villes
    """

    nom_fichier = f"./Pb{num}.txt"

    try:
        with open(nom_fichier, "r") as fichier:
            lignes = fichier.readlines()

    except FileNotFoundError:
        print(f"Le fichier {nom_fichier} n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")

    villes = []
    for i in range(1, len(lignes) - 1):
        a = lignes[i].split(" ")
        a[1] = int(a[1][:-1])
        a[0] = int(a[0])
        villes.append(a)

    num_villes = len(villes)
    distances = [
        [calcul_distance(villes[i], villes[j]) for j in range(num_villes)]
        for i in range(num_villes)
    ]

    return num_villes, villes, distances


def clusturing(K, donne):
    """
    :param K: nombre de clusters
    : param donne : donne des villes a clusturer
    :return:
    centroids : centre des clusters
    labels : appartenance aux groupes
    groupe : liste de groupe des points clusturés
    """

    kmeans = KMeans(n_clusters=K)
    # Effectuer le clustering sur les données
    kmeans.fit(donne)
    # Obtenir les centres de cluster
    centroids = kmeans.cluster_centers_
    # Obtenir les étiquettes de cluster pour chaque point
    labels = kmeans.labels_

    groupe = [[] for i in range(K)]
    for i in range(len(labels)):
        L = labels[i]
        groupe[L].append(donne[i])

    return centroids, labels, groupe


def organisation_groupes(centroids, groupe):
    """
    :param centroids: liste des centres des clusters
    :param groupe: liste des groupes de coordonné
    :return: liste des groupes ordonnés
    """
    dist_data = []
    # Ordre de reliance des clusters
    for i in range(10):
        sol, dist = glouton(centroids)
        if i == 0:
            min = dist
            sol_opti = sol
        elif i != 0:
            if dist < min:
                min = dist
                sol_opti = sol
        dist = round(dist, 2)
        dist_data.append(dist)

    end_distance, end_solution = swap_descente_complete(sol_opti, min)

    groupe_ordonne = [0] * len(groupe)
    end_sol_temp = [list(arr) for arr in end_solution]
    for i in range(len(end_sol_temp)):
        centre = centroids[i]
        for indice, sous_liste in enumerate(end_sol_temp):
            if (sous_liste == centre)[0] and (sous_liste == centre)[1]:
                index = indice
        groupe_ordonne[index] = groupe[i]

    return groupe_ordonne


def Depart_arrive(grp_av, groupe_apr):
    """
    :param grp_av: groupe 1
    :param groupe_apr: groupe 2
    :return: deux points els proches de chaque groupe
    """

    distance_min = float("inf")
    point1_plus_proche = None
    point2_plus_proche = None
    for point1 in grp_av:
        for point2 in groupe_apr:
            dist = calcul_distance(point1, point2)
            if dist < distance_min:
                distance_min = dist
                point1_plus_proche = point1
                point2_plus_proche = point2
    return [point1_plus_proche, point2_plus_proche]


def liste_depart_arrive(groupe):
    """
    :param groupe: groupes des points ordonnée
    :return: liste des points a connecté entre groupes
    """

    liste_depart_arrive = []
    for i in range(len(groupe) - 1):
        liste_inter = Depart_arrive(groupe[i], groupe[i + 1])
        liste_depart_arrive.append(liste_inter)
    liste_depart_arrive.append(Depart_arrive(groupe[len(groupe) - 1], groupe[0]))

    return liste_depart_arrive


def cherche_point(liste, point):
    """
    :param liste:
    :param point:
    :return: trouve l'indice du point dans la liste
    """

    index_trouve = None
    for index, element in enumerate(liste):
        if element == point:
            index_trouve = index
            break
    return index_trouve


def optimisation_des_groupes(groupe_ordonne, liste_depart_arriv):
    """
    :param groupe_ordonne: groupe ordonnés
    :param liste_depart_arriv: points a relier entre groupes
    :return: liste ordonnées des points a parcourir
    """
    Sol_gp = []
    liste_sol = []
    dist_total = 0
    for i in range(len(groupe_ordonne)):
        depart = liste_depart_arriv[i - 1][1]
        arrive = liste_depart_arriv[i][0]
        depart = cherche_point(groupe_ordonne[i], depart)
        arrive = cherche_point(groupe_ordonne[i], arrive)
        sol_gp1, dist_gp1 = glouton_depart_arrive(groupe_ordonne[i], depart, arrive)
        # print(sol_gp1[0],sol_gp1[-1])
        end_distance_gp1, end_solution_gp1 = swap_premier_dernier(sol_gp1, dist_gp1)
        liste_sol.append(end_solution_gp1)
        dist_total += end_distance_gp1
        # print(end_solution_gp1[0], end_solution_gp1[-1])
        Sol_gp = Sol_gp + end_solution_gp1
    return Sol_gp, dist_total, liste_sol


best_cluster_score = 0
best_cluster = 0
best_scores = []
for i in range(1, 6):
    nombre_de_villes, data, distances = import_fichier(i)

    for k in range(4, i * 6):
        centroids, labels, groupe = clusturing(k, data)
        groupe_ordonne = organisation_groupes(centroids, groupe)
        liste_depart = liste_depart_arrive(groupe_ordonne)
        # print(liste_depart)
        try:
            sol, distance, liste_sol = optimisation_des_groupes(
                groupe_ordonne, liste_depart
            )
            if k == 4:
                best_cluster_score = distance
                best_cluster = k
            else:
                if distance < best_cluster_score:
                    best_cluster_score = distance
                    best_cluster = k
            print(f"resultat pour {k} clusters : ", distance)
        except:
            continue
    best_scores.append([best_cluster_score, best_cluster])
    print(f"pb {i} best cluster = {best_cluster} with score = {best_cluster_score}")
print(best_scores)
# print('distance finale = ', distance)
plot_resul(sol, labels, data, liste_sol)
