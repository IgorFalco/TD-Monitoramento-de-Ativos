import numpy as np

def one_team_per_base(solution):
    """Cada equipe deve ser alocada a exatamente uma base."""
    y = solution['y']
    return np.all(np.sum(y, axis=0) == 1)

def one_assets_per_base(solution):
    """Cada ativo deve ser atribuído a exatamente uma base."""
    x = solution['x']
    return np.all(np.sum(x, axis=1) == 1)

def at_least_one_team_per_base(solution):
    """Cada ativo só pode ser atribuído a uma base ocupada por pelo menos uma equipe."""
    x = solution['x']
    y = solution['y']
    num_assets, num_bases = x.shape

    for i in range(num_assets):
        for j in range(num_bases):
            if x[i, j] == 1:
                if np.sum(y[j, :]) == 0:
                    return False
    return True

def one_asset_per_team(solution):
    """Cada ativo deve ser atribuído a exatamente uma equipe."""
    h = solution['h']
    return np.all(np.sum(h, axis=1) == 1)

def asset_and_team_must_be_in_the_same_base(solution):
    """Cada ativo deve estar na mesma base que sua equipe responsável."""
    x = solution['x']
    y = solution['y']
    h = solution['h']
    num_assets, num_bases = x.shape

    for i in range(num_assets):
        for j in range(num_bases):
            if x[i, j] == 1:  # Se ativo i está na base j
                # Verifica se a base j tem alguma equipe
                if not np.any(y[j, :] == 1):
                    return False
                
                # Verifica se o ativo i está com uma equipe que ocupa a base j
                if not np.any(np.logical_and(y[j, :], h[i, :])):
                    return False
    return True

def min_assets_per_team(solution):
    """Cada equipe deve monitorar pelo menos 20% do total de ativos dividido pelo número de equipes."""
    h = solution['h']
    num_assets, num_teams = h.shape
    
    min_load = (0.2 * num_assets) / num_teams
    return np.all(np.sum(h, axis=0) >= min_load)

def max_team(solution):
    """O número máximo de equipes é 8."""
    h = solution['h']
    num_assets, num_teams = h.shape
    return num_teams <= 8

# Lista de todas as constraints para validação
constraints = [
    one_team_per_base,
    one_assets_per_base, 
    at_least_one_team_per_base,
    one_asset_per_team,
    asset_and_team_must_be_in_the_same_base,
    min_assets_per_team,
    max_team,
]