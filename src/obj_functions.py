import numpy as np

def objective_function_1(solution, dist_bases_assets):
    """Calcula f1: soma das distâncias entre ativos e bases (minimizar)."""
    solution['f1'] = 0
    x = solution['x']
    num_assets, num_bases = x.shape

    for i in range(num_assets):
        for j in range(num_bases):
            if x[i, j] == 1:
                solution['f1'] += dist_bases_assets[i, j]

def objective_function_2(solution):
    """Calcula f2: número total de equipes (minimizar)."""
    y = solution['y']
    solution['f2'] = np.sum(y)

def objective_function_weighted(solution, dist_bases_assets, w1, w2,
                                f1_max=None, f2_max=8,
                                f1_min=None, f2_min=None):
    """Soma ponderada normalizada: F = w1*f1_norm + w2*f2_norm."""
    # Atualiza f1 e f2
    objective_function_1(solution, dist_bases_assets)
    objective_function_2(solution)

    # Normalização (min–max quando possível; senão, divide por máximo)
    eps = 1e-12
    # f1
    if f1_min is not None and f1_max is not None and (f1_max - f1_min) > eps:
        f1_norm = (solution['f1'] - f1_min) / max(eps, (f1_max - f1_min))
    else:
        if f1_max is None or f1_max <= eps:
            f1_max = solution['f1'] if solution['f1'] > 0 else 1.0
        f1_norm = solution['f1'] / f1_max
    # f2
    if f2_min is not None and f2_max is not None and (f2_max - f2_min) > eps:
        f2_norm = (solution['f2'] - f2_min) / max(eps, (f2_max - f2_min))
    else:
        denom = f2_max if f2_max else 1.0
        f2_norm = solution['f2'] / denom

    return w1 * f1_norm + w2 * f2_norm