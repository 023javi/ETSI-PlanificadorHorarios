import numpy as np
import random

from fontTools.merge.util import first

# Ejemplo de dataset de entrada para el problema de asignación de horarios
dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}


def generate_random_array_int(alphabet, length):
    # Genera un array de enteros aleatorios de tamaño length
    # usando el alfabeto dado
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]

def generate_initial_population_timetabling(pop_size, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo

    #Estraer información del dataset
    courses = dataset['courses']
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    #Calcular alfabeto y longitud
    alphabet = list(range(n_days * n_hours_day))  # Rango de valores posibles
    length = sum(hours for _, hours in courses)  # Total de horas a planificar


    # Generar población inicial
    population = [generate_random_array_int(alphabet, length) for _ in range(pop_size)]

    return population





################################# NO TOCAR #################################
#                                                                          #
def print_timetabling_solution(solution, dataset):
    # Imprime una solución de timetabling
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crea una matriz de n_days x n_hours_day
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Llena la matriz con las asignaturas
    i = 0
    max_len = 6 # Longitud del título Día XX
    for course in courses:
        for _ in range(course[1]):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[day][hour].append(course[0])
            i += 1
            # Calcula la longitud máxima del nombre de las asignaturas
            # en una misma franja horaria
            max_len = max(max_len, len('/'.join(timetable[day][hour])))

    # Imprime la matriz con formato de tabla markdown
    print('|         |', end='')
    for i in range(n_days):
        print(f' Día {i+1:<2}{" "*(max_len-6)} |', end='')
    print()
    print('|---------|', end='')
    for i in range(n_days):
        print(f'-{"-"*max_len}-|', end='')
    print()
    for j in range(n_hours_day):
        print(f'| Hora {j+1:<2} |', end='')
        for i in range(n_days):
            s = '/'.join(timetable[i][j])
            print(f' {s}{" "*(max_len-len(s))}', end=' |')
        print()
#                                                                          #
################################# NO TOCAR #################################


# Ejemplo de uso de la función generar individuo con el dataset de ejemplo
candidate = generate_random_array_int(list(range(9)), 6)
print_timetabling_solution(candidate, dataset)

def timetable_matrix(solution, dataset):
    n_hours_day = dataset['n_hours_day']
    n_days = dataset['n_days']

    schedule = np.zeros((n_days, n_hours_day), dtype=int)

    for val in solution:
        day = val // n_hours_day
        hour = val % n_hours_day
        schedule[day, hour] += 1

    return schedule

print(timetable_matrix(candidate, dataset=dataset))
def calculate_c1(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    # Calcula la cantidad de asignaturas que se imparten en mismas franjas horarias
    schedule = timetable_matrix(solution, dataset)

    c1 = np.sum(schedule[schedule > 1] - 1)

    return int(c1)

print("c1: ", calculate_c1(candidate, dataset=dataset))

def calculate_c2(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    # Calcula la cantidad de horas por encima de 2 que se imparten
    # de una misma asignatura en un mismo día
    courses = dataset['courses']

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    # Crear un diccionario para contar asignaturas por día
    daily_count = [{} for _ in range(n_days)]

    #Asignar las horas de la solución al diccionario
    idx = 0
    for course, hours in courses:
        for _ in range(hours):
            day = solution[idx] // n_hours_day
            daily_count[day][course] = daily_count[day].get(course, 0) + 1
            idx += 1

    c2 = sum(max(0, count - 2) for day in daily_count for count in day.values())
    return c2

print("C2: ", calculate_c2(candidate, dataset=dataset))


def calculate_p1(solution, *args, **kwargs):
    """
    Calcula el número de huecos vacíos entre asignaturas.
    """
    dataset = kwargs['dataset']

    # Crear matriz de horarios [n_days][n_hours_day] para marcar asignaturas
    schedule = timetable_matrix(solution, dataset)

    # Contar huecos vacíos
    p1 = 0
    for day in schedule:
        if np.any(day):  # Si hay asignaturas en el día
            first = np.argmax(day > 0)  # Primera hora asignada
            last = len(day) - np.argmax(day[::-1] > 0)  # Última hora asignada
            p1 += np.sum(day[first:last] == 0)  # Contar huecos entre primera y última hora

    return int(p1)


print("p1: ", calculate_p1(candidate, dataset=dataset))


def calculate_p2(solution, *args, **kwargs):
    """
    Calcula el número de días utilizados en los horarios.
    """
    dataset = kwargs['dataset']
    schedule = timetable_matrix(solution, dataset)

    # Contar días con asignaturas
    p2 = np.sum(np.any(schedule, axis=1))
    return int(p2)

print("p2: ", calculate_p2(candidate, dataset=dataset))
def calculate_p3(solution, *args, **kwargs):
    """
    Calcula el número de asignaturas con horas no consecutivas en un mismo día.
    """
    dataset = kwargs['dataset']
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crear un diccionario para almacenar las horas asignadas a cada asignatura
    daily_assignments = [{} for _ in range(n_days)]

    idx = 0
    for course, hours in courses:
        for _ in range(hours):
            day = solution[idx] // n_hours_day
            hour = solution[idx] % n_hours_day
            daily_assignments[day].setdefault(course, []).append(hour)
            idx += 1

    # Contar asignaturas no consecutivas
    p3 = 0
    for day in daily_assignments:
        for hours in day.values():
            if len(hours) > 1 and any(b - a > 1 for a, b in zip(sorted(hours), sorted(hours)[1:])):
                p3 += 1

    return int(p3)


print("p3:", calculate_p3(candidate, dataset=dataset))

def fitness_timetabling(solution, *args, **kwargs):
    """
    Calcula el fitness de una solución de timetabling siguiendo las penalizaciones.
    """
    dataset = kwargs['dataset']
    c1 = calculate_c1(solution, dataset=dataset)
    c2 = calculate_c2(solution, dataset=dataset)
    p1 = calculate_p1(solution, dataset=dataset)
    p2 = calculate_p2(solution, dataset=dataset)
    p3 = calculate_p3(solution, dataset=dataset)

    # Fitness según la fórmula del enunciado
    if c1 > 0 or c2 > 0:
        return 0
    return 1 / (1 + p1 + p2 + p3)


print(fitness_timetabling(candidate, dataset=dataset))
def hours_per_subject(dataset):
    return [course[1] for course in dataset['courses']]


# Pistas:
# - Una función que devuelva la tabla de horarios de una solución
# - Una función que devuelva la cantidad de horas por día de cada asignatura
# - A través de args y kwargs se pueden pasar argumentos adicionales que vayamos a necesitar
#fitness_timetabling(candidate, dataset=dataset) # Devuelve la fitness del candidato de ejemplo


def tournament_selection(population, fitness, number_parents, *args, **kwargs):
    t = kwargs['tournament_size']  # Tamaño del torneo
    newpopulation = []

    for _ in range(number_parents):
        parent = select_one_parent(population, fitness, t)
        newpopulation.append(parent)

    return newpopulation

def select_one_parent(population, fitness, tournament_size):
    # Selecciona tournament_size individuos aleatoriamente
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    # Encuentra el individuo con el mejor fitness en el torneo
    best_index = selected_indices[np.argmin([fitness[i] for i in selected_indices])]
    return population[best_index]

# Pista:
# - Crear una función auxiliar que genere un padre a partir de una selección por torneo
# - Recuerda usar la misma librería de números aleatorios que en el resto del código


def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    # Verificar si se realiza el cruce basado en la probabilidad p_cross
    if np.random.rand() > p_cross:
        # Si no se realiza el cruce, los hijos son copias de los padres
        return parent1, parent2

    # Seleccionar un punto de cruce aleatorio
    crossover_point = np.random.randint(1, len(parent1))

    # Crear los hijos combinando las partes de los padres
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return child1, child2


def uniform_mutation(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo
    # Realiza la mutación gen a gen con una probabilidad p_mut
    # Obtener el alfabeto del dataset para aplicar la mutación
    sol = chromosome.copy()
    m = dataset['n_days']
    k = dataset['n_hours_day']
    for i in range(0,len(sol)):
        if(np.random.random() < p_mut):
            sol[i] = np.random.randint(0,m*k)
    return sol



def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    # Realiza la sustitución generacional de la población
    # Debe devolver tanto la nueva población como el fitness de la misma
    auxp = population.copy()
    auxf = fitness.copy()
    for i in range(len(offspring)):
        auxp.pop(0)
        auxp.append(offspring.pop(0))
        auxf.pop(0)
        auxf.append(fitness_offspring.pop(0))
    return auxp, auxf


def generation_stop(generation, fitness, *args, **kwargs):
    # Comprueba si se cumple el criterio de parada (máximo número de generaciones)
    max_gen=kwargs['max_gen']
    if generation == max_gen:
        return True
    return False


def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Inicializar población
    population = generate_population(pop_size, *args, **kwargs)
    fitness = [fitness_function(ind, *args, **kwargs) for ind in population]
    best_fitness = [max(fitness)]
    mean_fitness = [sum(fitness) / len(fitness)]
    generation = 1


    # Ciclo evolutivo
    while not stopping_criteria(generation, fitness, *args, **kwargs):
        auxP, auxF = population.copy(), fitness.copy()
        for _ in range(offspring_size // 2):
            parents = selection(population, fitness, 2, *args, **kwargs)
            offspring1, offspring2 = crossover(parents[0], parents[1], p_cross, *args, **kwargs)
            offspring1 = mutation(offspring1, p_mut, *args, **kwargs)
            offspring2 = mutation(offspring2, p_mut, *args, **kwargs)
            offspring_fitness = [
                fitness_function(offspring1, *args, **kwargs),
                fitness_function(offspring2, *args, **kwargs)
            ]
            auxP, auxF = environmental_selection(auxP, auxF, [offspring1, offspring2], offspring_fitness, *args, **kwargs)

        population, fitness = auxP.copy(), auxF.copy()
        best_fitness.append(max(fitness))
        mean_fitness.append(sum(fitness) / len(fitness))
        generation += 1



    return population, fitness, generation, best_fitness, mean_fitness




### Coloca aquí tus funciones propuestas para la generación de población inicial ###

### Coloca aquí tus funciones de fitness propuestas ###

### Coloca aquí tus funciones de selección propuestas ###

### Coloca aquí tus funciones de cruce propuestas ###

### Coloca aquí tus funciones de mutación propuestas ###

### Coloca aquí tus funciones de reemplazo propuestas ###

### Coloca aquí tus funciones de parada propuestas ###

################################# NO TOCAR #################################
#                                                                          #
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        return *res, end - start
    return wrapper
#                                                                          #
################################# NO TOCAR #################################

# Este codigo temporiza la ejecución de una función cualquiera

################################# NO TOCAR #################################
#                                                                          #
@timer
def run_ga(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
           selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Además del retorno de la función, se devuelve el tiempo de ejecución en segundos
    return genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                             selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs)
#                                                                          #
################################# NO TOCAR #################################

# Se deben probar los 6 datasets
dataset1 = {"n_courses" : 3,
            "n_days" : 3,
            "n_hours_day" : 3,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

dataset2 = {"n_courses" : 4,
            "n_days" : 3,
            "n_hours_day" : 4,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3), ("POO", 2)]}

dataset3 = {"n_courses" : 4,
            "n_days" : 4,
            "n_hours_day" : 4,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4)]}

dataset4 = {"n_courses" : 5,
            "n_days" : 4,
            "n_hours_day" : 6,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4)]}

dataset5 = {"n_courses" : 7,
            "n_days" : 4,
            "n_hours_day" : 8,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2)]}

dataset6 = {"n_courses" : 11,
            "n_days" : 5,
            "n_hours_day" : 12,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2), ("FC", 4), ("TSO", 2), ("AM", 4), ("LMD", 4)]}



def set_seed(seed):
    # Se debe fijar la semilla usada para generar números aleatorios
    # Con la librería random
    random.seed(seed)
    # Con la librería numpy
    np.random.seed(seed)

################################# NO TOCAR #################################
#                                                                          #
def best_solution(population, fitness):
    # Devuelve la mejor solución de la población
    return population[fitness.index(max(fitness))]

import matplotlib.pyplot as plt
def plot_fitness_evolution(best_fitness, mean_fitness):
    plt.plot(best_fitness, label='Best fitness')
    plt.plot(mean_fitness, label='Mean fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
#                                                                          #
################################# NO TOCAR #################################


from statistics import mean, median, stdev

def launch_experiment(seeds, dataset, generate_population, pop_size, fitness_function, c1, c2, p1, p2, p3, stopping_criteria,
                      offspring_size, selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    best_individuals = []
    best_inds_c1 = []
    best_inds_c2 = []
    best_inds_p1 = []
    best_inds_p2 = []
    best_inds_p3 = []
    best_inds_fitness = []
    best_fitnesses = []
    mean_fitnesses = []
    last_generations = []
    execution_times = []
    # Ejecutamos el algoritmo con cada semilla
    for seed in seeds:
        print(f"Running Genetic Algorithm with seed {seed}")
        set_seed(seed)
        population, fitness, generation, best_fitness, mean_fitness, execution_time = run_ga(generate_population, pop_size, fitness_function,stopping_criteria,
                                                                                             offspring_size, selection, crossover, p_cross, mutation, p_mut,
                                                                                             environmental_selection, dataset=dataset, *args, **kwargs)
        best_individual = best_solution(population, fitness)
        best_ind_c1 = c1(best_individual, dataset=dataset)
        best_ind_c2 = c2(best_individual, dataset=dataset)
        best_ind_p1 = p1(best_individual, dataset=dataset)
        best_ind_p2 = p2(best_individual, dataset=dataset)
        best_ind_p3 = p3(best_individual, dataset=dataset)
        best_ind_fitness = fitness_function(best_individual, dataset=dataset)
        best_individuals.append(best_individual)
        best_inds_c1.append(best_ind_c1)
        best_inds_c2.append(best_ind_c2)
        best_inds_p1.append(best_ind_p1)
        best_inds_p2.append(best_ind_p2)
        best_inds_p3.append(best_ind_p3)
        best_inds_fitness.append(best_ind_fitness)
        best_fitnesses.append(best_fitness)
        mean_fitnesses.append(mean_fitness)
        last_generations.append(generation)
        execution_times.append(execution_time)
    # Imprimimos la media y desviación típica de los resultados obtenidos
    print("Mean Best Fitness: " + str(mean(best_inds_fitness)) + " " + u"\u00B1" + " " + str(stdev(best_inds_fitness)))
    print("Mean C1: " + str(mean(best_inds_c1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c1)))
    print("Mean C2: " + str(mean(best_inds_c2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c2)))
    print("Mean P1: " + str(mean(best_inds_p1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p1)))
    print("Mean P2: " + str(mean(best_inds_p2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p2)))
    print("Mean P3: " + str(mean(best_inds_p3)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p3)))
    print("Mean Execution Time: " + str(mean(execution_times)) + " " + u"\u00B1" + " " + str(stdev(execution_times)))
    print("Mean Number of Generations: " + str(mean(last_generations)) + " " + u"\u00B1" + " " + str(stdev(last_generations)))
    # Mostramos la evolución de la fitness para la mejor ejecución
    print("Best execution fitness evolution:")
    best_execution = best_inds_fitness.index(max(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[best_execution], mean_fitnesses[best_execution])
    # Mostramos la evolución de la fitness para la ejecución mediana
    print("Median execution fitness evolution:")
    median_execution = best_inds_fitness.index(median(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[median_execution], mean_fitnesses[median_execution])
    # Mostramos la evolución de la fitness para la peor ejecución
    print("Worst execution fitness evolution:")
    worst_execution = best_inds_fitness.index(min(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[worst_execution], mean_fitnesses[worst_execution])

    return best_individuals, best_inds_fitness, best_fitnesses, mean_fitnesses, last_generations, execution_times


# Crear un conjunto de 31 semillas para los experimentos
seeds = [1234567890 + i*23 for i in range(31)] # Semillas de ejemplo, cambiar por las semillas que se quieran
bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset1, generate_initial_population_timetabling, 50, fitness_timetabling, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop, 50, tournament_selection, one_point_crossover, 0.8,
                  uniform_mutation, 0.1, generational_replacement, max_gen=50, tournament_size=2)

print("Mejor individuo del mejor: \n")
print(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))])
print("\nMejor individuo del mediano: \n")
print(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))])
print("\nMejor individuo del peor: \n")
print(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))])
# Recuerda también mostrar el horario de la mejor solución obtenida en los casos peor, mejor y mediano
