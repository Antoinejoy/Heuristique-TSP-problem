import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Ouvrir un fichier texte en mode lecture

nom_fichier = "./Pb3.txt"

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


def calcul_distance(ville1, ville2):
    x1, y1 = ville1
    x2, y2 = ville2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


distances = [
    [calcul_distance(villes[i], villes[j]) for j in range(num_villes)]
    for i in range(num_villes)
]

# Initialisation des niveaux de phéromones
pheromones = [[1.0 for _ in range(num_villes)] for _ in range(num_villes)]

# Paramètres de l'algorithme
nombre_fourmis = 30
iterations = 200
taux_evaporation = 0.1
alpha = 1
beta = 1


# Fonction pour choisir la prochaine ville à visiter pour une fourmi
def choisir_ville_suivante(ville_actuelle, villes_non_visitees):
    probabilities = []
    total_prob = 0.0

    for ville in villes_non_visitees:
        pheromone = pheromones[ville_actuelle][ville]
        distance = distances[ville_actuelle][ville]
        prob = (pheromone**alpha) * ((1.0 / distance) ** beta)
        probabilities.append(prob)
        total_prob += prob

    norm_probs = [prob / total_prob for prob in probabilities]
    # print(villes_non_visitees,norm_probs)
    choix = random.choices(villes_non_visitees, k=1, weights=norm_probs)
    return choix[0]


for _ in range(iterations):
    print(_)
    for ant in range(nombre_fourmis):
        ville_depart = random.randint(0, num_villes - 1)
        chemin = [ville_depart]
        villes_non_visitees = [a for a in range(num_villes)]
        villes_non_visitees.remove(ville_depart)

        while villes_non_visitees:
            prochaine_ville = choisir_ville_suivante(chemin[-1], villes_non_visitees)
            chemin.append(prochaine_ville)
            villes_non_visitees.remove(prochaine_ville)

        # Calculez la longueur du chemin
        longueur_chemin = sum(
            distances[chemin[i]][chemin[i + 1]] for i in range(num_villes - 1)
        )
        longueur_chemin += distances[chemin[-1]][chemin[0]]  # Fermeture du cycle

        # Mettez à jour les niveaux de phéromones
        for i in range(num_villes - 1):
            pheromones[chemin[i]][chemin[i + 1]] = (1 - taux_evaporation) * pheromones[
                chemin[i]
            ][chemin[i + 1]] + (1.0 / longueur_chemin)

    # evaporation
    for i in range(num_villes):
        for j in range(num_villes):
            pheromones[i][j] *= taux_evaporation

# selection meilleure solution
meilleur_chemin = None
meilleure_distance = float("inf")

print("selection")

for ant in range(nombre_fourmis):
    chemin = [ville_depart]
    villes_non_visitees = [a for a in range(num_villes)]
    villes_non_visitees.remove(ville_depart)

    while villes_non_visitees:
        prochaine_ville = choisir_ville_suivante(chemin[-1], villes_non_visitees)
        chemin.append(prochaine_ville)
        villes_non_visitees.remove(prochaine_ville)

    longueur_chemin = sum(
        distances[chemin[i]][chemin[i + 1]] for i in range(num_villes - 1)
    )
    longueur_chemin += distances[chemin[-1]][chemin[0]]

    if longueur_chemin < meilleure_distance:
        meilleur_chemin = chemin
        meilleure_distance = longueur_chemin

print("Meilleur chemin trouvé:", meilleur_chemin)
print("Longueur du meilleur chemin:", meilleure_distance)

# plot de la solutio


ville_ordonne = []
for i in meilleur_chemin:
    ville_ordonne.append(villes[i])

x, y = zip(*ville_ordonne)

# Tracez les points
plt.plot(x, y, marker="o", linestyle="-")

# Ajoutez des étiquettes aux axes
plt.xlabel("Axe X")
plt.ylabel("Axe Y")

# Affichez le graphe
plt.show()
