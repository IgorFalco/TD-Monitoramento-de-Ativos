import numpy as np

def objective_function_1(solution, dist_bases_assets):
    """
    Calcula f1: minimizar a soma das distâncias entre ativos e suas bases.
    
    Args:
        solution: Dicionário com a solução
        dist_bases_assets: Matriz de distâncias [ativo x base]
    """
    solution['f1'] = 0
    x = solution['x']
    num_assets, num_bases = x.shape

    for i in range(num_assets):
        for j in range(num_bases):
            if x[i, j] == 1:
                solution['f1'] += dist_bases_assets[i, j]

def objective_function_2(solution):
    """
    Calcula f2: minimizar o número de equipes contratadas.
    
    Args:
        solution: Dicionário com a solução
    """
    y = solution['y']
    solution['f2'] = np.sum(y)