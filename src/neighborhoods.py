import numpy as np
import random
from constraints import constraints

def neighborhood_swap_assets(solution, dist_bases_assets):
    """Troca dois ativo        new_solution['x'][asset, base1] = 0
        new_solution['x'][asset, base2] = 1
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solutionpes diferentes."""
    from obj_functions import objective_function_1, objective_function_2
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_assets, num_teams = solution['h'].shape
    
    # Encontra equipes com múltiplos ativos
    team_assets = {}
    for k in range(num_teams):
        assets_k = np.where(solution['h'][:, k] == 1)[0]
        if len(assets_k) > 1:
            team_assets[k] = list(assets_k)
    
    if len(team_assets) < 2:
        return None
    
    # Seleciona equipes e ativos
    teams = list(team_assets.keys())
    team1, team2 = random.sample(teams, 2)
    asset1 = random.choice(team_assets[team1])
    asset2 = random.choice(team_assets[team2])
    
    # Bases das equipes
    base_team1 = np.where(solution['y'][:, team1] == 1)[0][0]
    base_team2 = np.where(solution['y'][:, team2] == 1)[0][0]
    
    # Executa swap
    new_solution['h'][asset1, team1] = 0
    new_solution['h'][asset2, team2] = 0
    new_solution['x'][asset1, base_team1] = 0
    new_solution['x'][asset2, base_team2] = 0
    
    new_solution['h'][asset1, team2] = 1
    new_solution['h'][asset2, team1] = 1
    new_solution['x'][asset1, base_team2] = 1
    new_solution['x'][asset2, base_team1] = 1
    
    if not validate_solution(new_solution):
        return None
    
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    return new_solution

def neighborhood_relocate_asset(solution, dist_bases_assets):
    """Realoca um ativo para uma equipe diferente."""
    from obj_functions import objective_function_1, objective_function_2
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_assets, num_teams = solution['h'].shape
    
    # Encontra equipes com mais de 1 ativo (para não deixar equipes vazias)
    teams_with_assets = []
    for k in range(num_teams):
        assets_k = np.where(solution['h'][:, k] == 1)[0]
        if len(assets_k) > 1:
            teams_with_assets.extend([(asset, k) for asset in assets_k])
    
    if len(teams_with_assets) == 0:
        return None
    
    # Seleciona um ativo aleatório
    asset, source_team = random.choice(teams_with_assets)
    
    # Seleciona uma equipe destino diferente
    target_teams = [k for k in range(num_teams) if k != source_team and np.sum(solution['y'][:, k]) > 0]
    if not target_teams:
        return None
    
    target_team = random.choice(target_teams)
    
    # Encontra as bases
    source_base = np.where(solution['y'][:, source_team] == 1)[0][0]
    target_base = np.where(solution['y'][:, target_team] == 1)[0][0]
    
    # Realoca o ativo
    new_solution['h'][asset, source_team] = 0
    new_solution['x'][asset, source_base] = 0
    new_solution['h'][asset, target_team] = 1
    new_solution['x'][asset, target_base] = 1
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solution

def neighborhood_swap_bases(solution, dist_bases_assets):
    """Troca bases entre duas equipes."""
    from obj_functions import objective_function_1, objective_function_2
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_bases, num_teams = solution['y'].shape
    
    # Encontra equipes ativas
    active_teams = []
    for k in range(num_teams):
        if np.sum(solution['y'][:, k]) > 0:
            active_teams.append(k)
    
    if len(active_teams) < 2:
        return None
    
    # Seleciona duas equipes
    team1, team2 = random.sample(active_teams, 2)
    
    # Encontra as bases das equipes
    base1 = np.where(solution['y'][:, team1] == 1)[0][0]
    base2 = np.where(solution['y'][:, team2] == 1)[0][0]
    
    # Troca as bases
    new_solution['y'][base1, team1] = 0
    new_solution['y'][base2, team2] = 0
    new_solution['y'][base1, team2] = 1
    new_solution['y'][base2, team1] = 1
    
    # Atualiza alocação dos ativos
    assets_team1 = np.where(solution['h'][:, team1] == 1)[0]
    assets_team2 = np.where(solution['h'][:, team2] == 1)[0]
    
    # Move ativos da equipe1 para a nova base (base2)
    for asset in assets_team1:
        new_solution['x'][asset, base1] = 0
        new_solution['x'][asset, base2] = 1
    
    # Move ativos da equipe2 para a nova base (base1)
    for asset in assets_team2:
        new_solution['x'][asset, base2] = 0
        new_solution['x'][asset, base1] = 1
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solution

def neighborhood_relocate_base(solution, dist_bases_assets):
    """Realoca uma equipe para uma base diferente."""
    from obj_functions import objective_function_1, objective_function_2
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_bases, num_teams = solution['y'].shape
    
    # Encontra equipes ativas e bases ocupadas
    active_teams = []
    occupied_bases = set()
    
    for k in range(num_teams):
        if np.sum(solution['y'][:, k]) > 0:
            active_teams.append(k)
            base_k = np.where(solution['y'][:, k] == 1)[0][0]
            occupied_bases.add(base_k)
    
    # Encontra bases livres
    free_bases = [j for j in range(num_bases) if j not in occupied_bases]
    
    if not active_teams or not free_bases:
        return None
    
    # Seleciona uma equipe e uma base livre
    team = random.choice(active_teams)
    new_base = random.choice(free_bases)
    current_base = np.where(solution['y'][:, team] == 1)[0][0]
    
    # Move a equipe para a nova base
    new_solution['y'][current_base, team] = 0
    new_solution['y'][new_base, team] = 1
    
    # Atualiza alocação dos ativos da equipe
    team_assets = np.where(solution['h'][:, team] == 1)[0]
    for asset in team_assets:
        new_solution['x'][asset, current_base] = 0
        new_solution['x'][asset, new_base] = 1
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solution

def neighborhood_or_opt(solution, dist_bases_assets, block_size=2):
    """Move um bloco de ativos de uma equipe para outra."""
    from obj_functions import objective_function_1, objective_function_2
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_assets, num_teams = solution['h'].shape
    
    # Encontra equipes com ativos suficientes
    source_teams = []
    for k in range(num_teams):
        assets_k = np.where(solution['h'][:, k] == 1)[0]
        if len(assets_k) > block_size:
            source_teams.append((k, list(assets_k)))
    
    if not source_teams:
        return None
    
    # Seleciona equipe origem e bloco de ativos
    source_team, source_assets = random.choice(source_teams)
    block_size_actual = min(block_size, len(source_assets) - 1)
    block_assets = random.sample(source_assets, block_size_actual)
    
    # Seleciona equipe destino
    target_teams = [k for k in range(num_teams) if k != source_team]
    if not target_teams:
        return None
    
    target_team = random.choice(target_teams)
    
    # Encontra bases
    source_base = np.where(solution['y'][:, source_team] == 1)[0][0]
    target_base = np.where(solution['y'][:, target_team] == 1)[0][0]
    
    # Move o bloco de ativos
    for asset in block_assets:
        new_solution['h'][asset, source_team] = 0
        new_solution['x'][asset, source_base] = 0
        new_solution['h'][asset, target_team] = 1
        new_solution['x'][asset, target_base] = 1
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solution

def neighborhood_change_team_number(solution, dist_bases_assets):
    """Aumenta ou diminui o número de equipes ativas."""
    from obj_functions import objective_function_1, objective_function_2
    import random
    
    new_solution = {
        'x': solution['x'].copy(),
        'y': solution['y'].copy(),
        'h': solution['h'].copy(),
        'f1': 0,
        'f2': 0
    }
    
    num_bases, num_teams = solution['y'].shape
    num_assets = solution['h'].shape[0]
    
    # Encontra equipes ativas
    active_teams = []
    for k in range(num_teams):
        if np.sum(solution['y'][:, k]) > 0:
            active_teams.append(k)
    
    # Decide se vai aumentar ou diminuir número de equipes
    # Só pode diminuir se tiver mais de 1 equipe ativa
    possible_actions = ['increase']
    if len(active_teams) > 1:
        possible_actions.append('decrease')
    
    action = random.choice(possible_actions)
    
    if action == 'increase':
        # Encontra bases ocupadas
        occupied_bases = set()
        for k in active_teams:
            base_k = np.where(solution['y'][:, k] == 1)[0][0]
            occupied_bases.add(base_k)
        
        # Encontra bases livres
        free_bases = [j for j in range(num_bases) if j not in occupied_bases]
        if not free_bases:
            return None  # Não há bases livres
        
        new_base = random.choice(free_bases)
        
        # Cria uma nova equipe (expande as matrizes)
        new_team = num_teams  # Próximo índice disponível
        
        # Expande a matriz y para incluir a nova equipe
        new_solution['y'] = np.column_stack([new_solution['y'], np.zeros((num_bases, 1))])
        new_solution['h'] = np.column_stack([new_solution['h'], np.zeros((num_assets, 1))])
        
        # Ativa a nova equipe na base selecionada
        new_solution['y'][new_base, new_team] = 1
        
        # Redistribui alguns ativos para a nova equipe
        # Seleciona uma equipe origem com ativos suficientes
        teams_with_assets = []
        for k in active_teams:
            assets_k = np.where(solution['h'][:, k] == 1)[0]
            if len(assets_k) > 1:  # Só considera equipes com mais de 1 ativo
                teams_with_assets.append((k, list(assets_k)))
        
        if teams_with_assets:
            source_team, source_assets = random.choice(teams_with_assets)
            source_base = np.where(solution['y'][:, source_team] == 1)[0][0]
            
            # Move alguns ativos para a nova equipe (entre 1 e metade dos ativos)
            max_move = len(source_assets) // 2 if len(source_assets) > 2 else 1
            num_assets_move = random.randint(1, max(1, max_move))
            assets_to_move = random.sample(source_assets, num_assets_move)
            
            for asset in assets_to_move:
                new_solution['h'][asset, source_team] = 0
                new_solution['x'][asset, source_base] = 0
                new_solution['h'][asset, new_team] = 1
                new_solution['x'][asset, new_base] = 1
    
    elif action == 'decrease' and len(active_teams) > 1:
        # Seleciona uma equipe para remover (preferencialmente com poucos ativos)
        teams_with_assets = []
        for k in active_teams:
            assets_k = np.where(solution['h'][:, k] == 1)[0]
            teams_with_assets.append((k, len(assets_k)))
        
        # Ordena por número de ativos (menor primeiro)
        teams_with_assets.sort(key=lambda x: x[1])
        team_to_remove = teams_with_assets[0][0]
        
        # Encontra ativos da equipe a ser removida
        assets_to_remove = np.where(solution['h'][:, team_to_remove] == 1)[0]
        base_to_remove = np.where(solution['y'][:, team_to_remove] == 1)[0][0]
        
        # Redistribui os ativos para outras equipes
        other_teams = [k for k in active_teams if k != team_to_remove]
        if other_teams and len(assets_to_remove) > 0:
            for asset in assets_to_remove:
                target_team = random.choice(other_teams)
                target_base = np.where(solution['y'][:, target_team] == 1)[0][0]
                
                new_solution['h'][asset, team_to_remove] = 0
                new_solution['x'][asset, base_to_remove] = 0
                new_solution['h'][asset, target_team] = 1
                new_solution['x'][asset, target_base] = 1
        
        # Remove a equipe das matrizes (remove a última coluna se for a última equipe)
        if team_to_remove == num_teams - 1:  # Se é a última equipe
            new_solution['y'] = new_solution['y'][:, :-1]  # Remove última coluna
            new_solution['h'] = new_solution['h'][:, :-1]  # Remove última coluna
        else:
            # Se não é a última, move a última equipe para o lugar da removida
            last_index = num_teams - 1
            # Copia dados da última equipe para a posição da equipe removida
            new_solution['y'][:, team_to_remove] = new_solution['y'][:, last_index]
            new_solution['h'][:, team_to_remove] = new_solution['h'][:, last_index]
            # Remove a última coluna
            new_solution['y'] = new_solution['y'][:, :-1]
            new_solution['h'] = new_solution['h'][:, :-1]
    
    else:
        return None  # Não foi possível fazer a operação
    
    # Verifica se a solução satisfaz todas as restrições
    if not validate_solution(new_solution):
        return None
    
    # Calcula objetivos
    objective_function_1(new_solution, dist_bases_assets)
    objective_function_2(new_solution)
    
    return new_solution

def validate_solution(solution):
    """Verifica se solução satisfaz todas as constraints."""
    try:
        for constraint in constraints:
            if not constraint(solution):
                return False
        return True
    except Exception:
        return False

def generate_complete_neighborhood(solution, neighborhood_func, dist_bases_assets, max_neighbors=50):
    """Gera vizinhança limitada usando função de vizinhança."""
    valid_neighbors = []
    attempts = 0
    max_attempts = max_neighbors * 3
    
    while len(valid_neighbors) < max_neighbors and attempts < max_attempts:
        attempts += 1
        neighbor = neighborhood_func(solution, dist_bases_assets)
        
        if neighbor is not None:
            valid_neighbors.append(neighbor)
    
    return valid_neighbors

# Lista das funções de vizinhança disponíveis
NEIGHBORHOODS = [
    neighborhood_swap_assets,
    neighborhood_relocate_asset,
    neighborhood_swap_bases,
    neighborhood_relocate_base,
    neighborhood_or_opt,
    neighborhood_change_team_number
]