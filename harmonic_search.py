import random
import math
import copy


def greedy(list):
    L = list.copy()
    total = 0
    u0 = L[0]
    pos = L[0]
    while len(L) != 0:
        distance_mini = float("inf")
        index = 0
        for i in range(len(L)):
            distance = math.sqrt((pos[0] - L[i][0]) ** 2 + (pos[1] - L[i][1]) ** 2)
            if distance < distance_mini:
                distance_mini = distance
                index = i
        total += distance_mini
        pos = L[index]
        L.pop(index)
    total += math.sqrt((pos[0] - u0[0]) ** 2 + (pos[1] - u0[1]) ** 2)
    return total, len(list)


def generate_random_solution(L):
    solution = L.copy()
    random.shuffle(solution)
    return solution


def harmony_search_tsp(
    city_list, max_iterations, harmony_memory_size, pitch_adjustment_rate
):
    best_solution = None
    best_solution_length = float("inf")
    harmony_memory = [
        generate_random_solution(city_list) for _ in range(harmony_memory_size)
    ]
    for iteration in range(max_iterations):
        candidate_solution = generate_random_solution(city_list)
        candidate_length, _ = greedy(candidate_solution)

        # maj si meilleure
        for i in range(harmony_memory_size):
            current_length, _ = greedy(harmony_memory[i])
            if candidate_length < current_length:
                harmony_memory[i] = candidate_solution.copy()
                break

        # Choisissez la meilleure solution dans la mémoire harmonique actuelle
        for solution in harmony_memory:
            current_length, _ = greedy(solution)
            if current_length < best_solution_length:
                best_solution_length = current_length
                best_solution = solution.copy()

        # Applique un ajustement de ton pour favoriser la diversité
        for i in range(len(candidate_solution)):
            if random.random() < pitch_adjustment_rate:
                random_index1 = random.randint(0, len(candidate_solution) - 1)
                random_index2 = random.randint(0, len(candidate_solution) - 1)
                candidate_solution[random_index1], candidate_solution[random_index2] = (
                    candidate_solution[random_index2],
                    candidate_solution[random_index1],
                )

    return best_solution, best_solution_length


path = "./pb3.txt"

# Ouvrir le fichier
with open(path, "r") as tf:
    lines = tf.readlines()[1:]

# remplir la liste
L = []
for line in lines:
    x, y = map(int, line.split())
    L.append([x, y])

city_list = L
max_iterations = 500
harmony_memory_size = 30
pitch_adjustment_rate = 0.1

best_solution, best_length = harmony_search_tsp(
    city_list, max_iterations, harmony_memory_size, pitch_adjustment_rate
)
print("Meilleure solution trouvée:", best_solution)
print("Longueur de la meilleure solution:", best_length)
