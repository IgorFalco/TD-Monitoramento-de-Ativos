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

def objective_function_weighted(solution, dist_bases_assets, w1, w2, f1_max=None, f2_max=8):
    """
    Calcula função objetivo ponderada normalizada: F(x) = w1*f1_norm(x) + w2*f2_norm(x)
    
    Conforme literatura de otimização multiobjetivo (método da soma ponderada).
    Normaliza os objetivos para evitar que diferenças de magnitude dominem a função.
    
    Args:
        solution: Dicionário com a solução
        dist_bases_assets: Matriz de distâncias [ativo x base]
        w1: Peso para f1 (distância)
        w2: Peso para f2 (número de equipes)
        f1_max: Valor máximo para normalização de f1 (se None, usa f1 atual como referência)
        f2_max: Valor máximo para normalização de f2 (padrão: 8 equipes)
    
    Returns:
        Valor da função objetivo ponderada normalizada
    """
    # Calcula f1 e f2
    objective_function_1(solution, dist_bases_assets)
    objective_function_2(solution)
    
    # Normalização: divide cada objetivo pelo seu valor máximo
    # f1_norm = f1 / f1_max (distâncias normalizadas entre 0 e 1)
    # f2_norm = f2 / f2_max (equipes normalizadas entre 0 e 1)
    
    # Se f1_max não foi fornecido, usa o valor atual de f1 como referência
    # (isso pode ser melhorado passando o f1_max da solução inicial ou pior caso)
    if f1_max is None or f1_max == 0:
        f1_max = solution['f1'] if solution['f1'] > 0 else 1.0
    
    f1_norm = solution['f1'] / f1_max
    f2_norm = solution['f2'] / f2_max
    
    # Calcula função ponderada com valores normalizados
    return w1 * f1_norm + w2 * f2_norm