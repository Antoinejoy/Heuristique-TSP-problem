import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import time
# Ouvrir un fichier texte en mode lecture

nom_fichier = "/Users/bagafoufabrice/Downloads/Pb3.txt"

try:
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()

except FileNotFoundError:
    print(f"Le fichier {nom_fichier} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite : {str(e)}")


data = []
for i in range(1,len(lignes)-1):
    a = lignes[i].split(' ')
    a[1] = int(a[1][:-1])
    a[0] = int(a[0])
    data.append(a)


data = []
for i in range(1,len(lignes)-1):
    a = lignes[i].split(' ')
    a[1] = int(a[1][:-1])
    a[0] = int(a[0])
    data.append(a)

#print(data)

#distance entre deux points
def calcul_distance(a,b):
    return int(sqrt((a[1]-b[1])**2+(a[0]-b[0])**2))

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
    depart = int(np.random.rand()*len(data)//1)
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

#sol,dist = glouton(data)
#dist = round(dist,2)


def permutation(config,i,j):
    config_inter = config.copy()
    config_inter[i] = config[j]
    config_inter[j] = config[i]
    return config_inter

def swap_descente(solution,distance):
    compt = 0
    print(len(solution))
    while compt < len(solution):
        for i in range(len(solution)):
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
                if nouvelle_distance < distance :
                    #print(nouvelle_distance)
                    print('nouvelle distance', nouvelle_distance)
                    compt = 0
                    solution = config_tempo
                    distance = nouvelle_distance
                    break
        #print(compt)
        compt+=1
    print('distance finale : ', distance)
    print('solution finale : ', solution)
    return distance,solution

def swap_descente_best_sol(solution,distance):
    compt = 0
    print(len(solution))
    best_distance_temp = 100000
    best_config = solution
    while compt < len(solution):
        for i in range(len(solution)):
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
                    break
        compt += 1
        #print(compt)
        if compt == len(solution)-1:
            if best_distance_temp == distance :
                #permutation aléatoire
                ran = np.random.rand(2)*len(data)//1
                x = int(ran[0])
                y = int(ran[1])
                solution = permutation(solution,x,y)
                compt = len(solution)
                #compt = 0
                break

            compt = 0
            distance = best_distance_temp
            print('nouvelle distance', best_distance_temp)
            solution = best_config

            #print(compt)

    print('distance finale : ', distance)
    #print('solution finale : ', solution)
    return distance,solution

def swap_descente_complete(solution,distance):
    compt = 0
    print(len(solution))
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
                    print(nouvelle_distance)
                    compt = 0
                    solution = config_tempo
                    distance = nouvelle_distance
                    break
        compt+=1
        print(compt)
    print('distance finale : ',distance)
    print('solution finale : ', solution)
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

dist_data = []
#comparaison des points de départ
for i in range(1):
    sol, dist = glouton(data)
    if i==0:
        min = dist
        sol_opti = sol
    elif i!=0:
        if dist<min:
            min = dist
            sol_opti = sol
    dist = round(dist,2)
    dist_data.append(dist)
    print(dist)
print(min)

end_distance,end_solution = swap_descente_best_sol(sol_opti,min)
plot_resul(data)



















