import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Constantes as C
import time

# points a ne pas prendre
Liste_point_bani = []


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
    while compt < len(donne) - 1:
        min, indice = plusproche_voisin(donne, depart, liste_entiers)

        compt += 1
        liste_entiers.remove(indice)
        depart = indice
        config.append(donne[indice])

    dist_verif = [
        calcul_distance(config[i], config[i + 1]) for i in range(len(config) - 1)
    ]
    distance = sum(dist_verif)

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
        # distance += min
        compt += 1
        liste_entiers.remove(indice)
        depart = indice
        config.append(donne[indice])

    config.append(donne[arrive])

    dist_verif = [
        calcul_distance(config[i], config[i + 1]) for i in range(len(config) - 1)
    ]
    return config, sum(dist_verif)


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
            ran = (np.random.rand(2) * (len(solution) - 3) // 1) + 1

            x = int(ran[0])
            y = int(ran[1])
            dist1_av = calcul_distance(solution[x], solution[x + 1])
            dist3_av = calcul_distance(solution[y], solution[y + 1])
            dist2_av = calcul_distance(solution[x], solution[x - 1])
            dist4_av = calcul_distance(solution[y], solution[y - 1])

            somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

            solution = permutation(solution, x, y)

            # ajout des nouvelles
            dist1 = calcul_distance(solution[x], solution[x + 1])
            dist3 = calcul_distance(solution[y], solution[y + 1])
            dist2 = calcul_distance(solution[x], solution[x - 1])
            dist4 = calcul_distance(solution[y], solution[y - 1])

            somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

            distance = distance - somme_des_dists_1 + somme_des_dists_2

            swap_reussi = True
            compt = 1

    dist_verif = [
        calcul_distance(solution[i], solution[i + 1]) for i in range(len(solution) - 1)
    ]

    # print('distance verifié swap ', sum(dist_verif))
    # print('distance calculé swap ', distance )
    return sum(dist_verif), solution


def swap_force_max(solution, distance, indice_max):
    """
    :param solution:
    :param distance:
    :param indice_max:
    :return:
    """
    swap_reussi = False
    compteur = 0
    while not swap_reussi and compteur < len(solution):
        # print('bon i')
        i = np.random.rand(1) * (len(solution) - 3) // 1 + 1
        if int(i[0]) == 0:
            i = [1]
        i = int(i[0])
        # print(indice_max,i,len(solution))
        dist1_av = calcul_distance(solution[indice_max], solution[indice_max + 1])
        dist3_av = calcul_distance(solution[i], solution[i + 1])
        dist2_av = calcul_distance(solution[indice_max], solution[indice_max - 1])
        dist4_av = calcul_distance(solution[i], solution[i - 1])

        somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

        # permutation
        config_tempo = permutation(solution, indice_max, i)

        dist1 = calcul_distance(config_tempo[indice_max], config_tempo[indice_max + 1])
        dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])
        dist2 = calcul_distance(config_tempo[indice_max], config_tempo[indice_max - 1])
        dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

        somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

        nouvelle_distance = distance - somme_des_dists_1 + somme_des_dists_2

        distance_swap, sol_swap = swap_premier_dernier(config_tempo, nouvelle_distance)

        if distance_swap < distance:
            print("swap opti max reussi, ecart = ", distance - distance_swap)
            distance = distance_swap
            swap_reussi = True
            solution = sol_swap
        compteur += 1

    dist_verif = [
        calcul_distance(solution[i], solution[i + 1]) for i in range(len(solution) - 1)
    ]
    sum_dist = sum(dist_verif)

    return sum_dist, solution


def swap_forced_deg3(solution, distance, indice_max):
    swap_reussi = False
    compteur = 0
    indice_max = np.random.rand(1) * (len(solution) - 3) // 1 + 1
    if int(indice_max[0]) == 0:
        indice_max = [1]
    indice_max = int(indice_max[0])
    while not swap_reussi and compteur < len(solution):
        # print('bon i')
        i = np.random.rand(1) * (len(solution) - 3) // 1 + 1
        if int(i[0]) == 0:
            i = [1]
        i = int(i[0])
        # print(indice_max,i,len(solution))
        dist1_av = calcul_distance(solution[indice_max], solution[indice_max + 1])
        dist3_av = calcul_distance(solution[i], solution[i + 1])
        dist2_av = calcul_distance(solution[indice_max], solution[indice_max - 1])
        dist4_av = calcul_distance(solution[i], solution[i - 1])

        somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

        # permutation
        config_tempo = permutation(solution, indice_max, i)

        dist1 = calcul_distance(config_tempo[indice_max], config_tempo[indice_max + 1])
        dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])
        dist2 = calcul_distance(config_tempo[indice_max], config_tempo[indice_max - 1])
        dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

        somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

        nouvelle_distance = distance - somme_des_dists_1 + somme_des_dists_2

        # deg 2
        while True:
            i = np.random.rand(1) * (len(config_tempo) - 3) // 1 + 1
            if int(i[0]) == 0:
                i = [1]
            i = int(i[0])

            j = np.random.rand(1) * (len(config_tempo) - 3) // 1 + 1
            if int(j[0]) == 0:
                j = [1]
            j = int(j[0])

            if i != j:
                break

        dist1_av = calcul_distance(solution[j], solution[j + 1])
        dist3_av = calcul_distance(solution[i], solution[i + 1])
        dist2_av = calcul_distance(solution[j], solution[j - 1])
        dist4_av = calcul_distance(solution[i], solution[i - 1])

        somme_des_dists_1 = dist1_av + dist2_av + dist3_av + dist4_av

        config_tempo = permutation(config_tempo, j, i)

        dist1 = calcul_distance(config_tempo[j], config_tempo[j + 1])
        dist3 = calcul_distance(config_tempo[i], config_tempo[i + 1])
        dist2 = calcul_distance(config_tempo[j], config_tempo[j - 1])
        dist4 = calcul_distance(config_tempo[i], config_tempo[i - 1])

        somme_des_dists_2 = dist1 + dist2 + dist3 + dist4

        nouvelle_distance = nouvelle_distance - somme_des_dists_1 + somme_des_dists_2

        distance_swap, sol_swap = swap_premier_dernier(config_tempo, nouvelle_distance)

        if distance_swap < distance:
            print("swap deg 3 reussi, ecart = ", distance - distance_swap)
            distance = distance_swap
            swap_reussi = True
            solution = sol_swap
        compteur += 1

    dist_verif = [
        calcul_distance(solution[i], solution[i + 1]) for i in range(len(solution) - 1)
    ]
    sum_dist = sum(dist_verif)

    return sum_dist, solution


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


def open_instance_path(path):
    """
    :param path: path of the instance
    :return:
    """
    try:
        with open(path, "r") as fichier:
            lignes = fichier.readlines()

    except FileNotFoundError:
        print(f"Le fichier n'a pas été trouvé.")
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
<<<<<<< HEAD
    # x, y = zip(*end_solution)
    # # Tracez les points
    # plt.plot(x, y, marker="o", linestyle="-")
    # plt.show()
=======
    #x, y = zip(*end_solution)
    # Tracez les points
    #plt.plot(x, y, marker='o', linestyle='-')
    #plt.show()
>>>>>>> 0e741c6c095baf9819df61528bad73907bb62038
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
    global Liste_point_bani
    """
    :param grp_av: groupe 1
    :param groupe_apr: groupe 2
    :return: deux points els proches de chaque groupe
    """

    print("nombre de point banni ", len(Liste_point_bani))
    distance_min = 10000
    point1_plus_proche = None
    point2_plus_proche = None
    for point1 in grp_av:
        if point1 in Liste_point_bani:
            continue
        for point2 in groupe_apr:
            if point2 in Liste_point_bani:
                continue
            dist = calcul_distance(point1, point2)
            if dist < distance_min:
                distance_min = dist

                point1_plus_proche = point1
                point2_plus_proche = point2
    Liste_point_bani.append(point1_plus_proche)
    Liste_point_bani.append(point2_plus_proche)
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

        end_distance_gp1, end_solution_gp1 = swap_premier_dernier(sol_gp1, dist_gp1)

        distances = [
            calcul_distance(end_solution_gp1[i], end_solution_gp1[i + 1])
            for i in range(len(end_solution_gp1) - 1)
        ]

        index_max = distances.index(max(distances))
        if index_max == 0:
            index_max = 1
        if index_max == len(distances) - 1:
            index_max = index_max - 2

        compt = 0
        while (
            distances[index_max]
            > (C.FORCED_MAX_SWAP_PRECISION * sum(distances) / len(distances))
            and compt < C.FORCED_MAX_SWAP_TRY
        ):
            previous_dist = end_distance_gp1

            end_distance_gp1, end_solution_gp1 = swap_forced_deg3(
                end_solution_gp1, end_distance_gp1, index_max
            )
            distances = [
                calcul_distance(end_solution_gp1[i], end_solution_gp1[i + 1])
                for i in range(len(end_solution_gp1) - 1)
            ]

            index_max = distances.index(max(distances))
            if index_max == 0:
                index_max = 1
            if index_max == len(distances) - 1:
                index_max = index_max - 2

            # Si le swap deg 3 fonctionne on optimise le max
            if previous_dist > end_distance_gp1:
                # print('opti du max')
                end_distance_gp1, end_solution_gp1 = swap_force_max(
                    end_solution_gp1, end_distance_gp1, index_max
                )
                distances = [
                    calcul_distance(end_solution_gp1[i], end_solution_gp1[i + 1])
                    for i in range(len(end_solution_gp1) - 1)
                ]
                index_max = distances.index(max(distances))

                if index_max == 0:
                    index_max = 1
                if index_max == len(distances) - 1:
                    index_max = index_max - 2

            compt = compt + 1
            # print('compteur : ', compt)

        if compt == C.FORCED_MAX_SWAP_TRY:
            print("au bout des essaies")

        liste_sol.append(end_solution_gp1)
        dist_total += end_distance_gp1

        Sol_gp = Sol_gp + end_solution_gp1
    return Sol_gp, dist_total, liste_sol


best_cluster_score = 0
best_cluster = 0
best_scores = []
list_score = []


nombre_de_villes, data, distances = import_fichier(3)
for k in range(C.MIN_CLUSTER_RANGE, C.MAX_CLUSTER_RANGE):
    start_time = time.time()
    centroids, labels, groupe = clusturing(k, data)

    groupe_ordonne = organisation_groupes(centroids, groupe)
    liste_depart = liste_depart_arrive(groupe_ordonne)

    try:
        sol, distance, liste_sol = optimisation_des_groupes(
            groupe_ordonne, liste_depart
        )
        dist_verif = [calcul_distance(sol[i], sol[i + 1]) for i in range(len(sol) - 1)]
        end_time = time.time()
        list_score.append([sum(dist_verif), k, end_time - start_time])
        Liste_point_bani = []
    except Exception as e:
        print(e)
        continue

print(list_score)
plot_resul(sol, labels, data, liste_sol)
