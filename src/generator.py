import numpy as np
import pandas as pd
from constraints import constraints

def read_csv_data():
    """
    Lê os dados do arquivo CSV e constrói a matriz de distâncias.
    
    Returns:
        dist_bases_assets: Matriz numpy [ativos x bases] com as distâncias
    """
    df = pd.read_csv('probdata.csv', delimiter=';', header=None, decimal=',')
    
    df.columns = ['latitude_base', 'longitude_base', 'latitude_ativo', 'longitude_ativo', 'distancia']
    bases = df[['latitude_base', 'longitude_base']].drop_duplicates().reset_index(drop=True)
    assets = df[['latitude_ativo', 'longitude_ativo']].drop_duplicates().reset_index(drop=True)

    base_indices = {tuple(base): idx for idx, base in bases.iterrows()}
    asset_indices = {tuple(asset): idx for idx, asset in assets.iterrows()}
    
    dist_bases_assets = np.zeros((len(assets), len(bases)))
    
    # Preenche a matriz de distâncias
    for _, row in df.iterrows():
        asset_idx = asset_indices[(row['latitude_ativo'], row['longitude_ativo'])]
        base_idx = base_indices[(row['latitude_base'], row['longitude_base'])]
        dist_bases_assets[asset_idx, base_idx] = row['distancia']
    
    return dist_bases_assets

def generate_initial_solution(num_assets, num_bases, dist_bases_assets):
    """
    Gera uma solução inicial que GARANTE o cumprimento de todas as constraints.
    Usa uma abordagem step-by-step rigorosa com exatamente 5 equipes para balancear
    de forma igual os ativos
    
    Args:
        num_assets: Número de ativos
        num_bases: Número de bases
        dist_bases_assets: Matriz de distâncias entre ativos e bases
    
    Returns:
        solution: Dicionário com a solução inicial que respeita todas as constraints
    """
    from obj_functions import objective_function_1, objective_function_2
    
    # Usa exatamente 5 equipes
    num_teams = 5
    
    # Verifica se é viável satisfazer min_assets_per_team
    min_assets_per_team = (0.2 * num_assets) / num_teams
    
    # Inicializa solução
    solution = {
        'x': np.zeros((num_assets, num_bases)),
        'y': np.zeros((num_bases, num_teams)),
        'h': np.zeros((num_assets, num_teams)),
        'f1': 0,
        'f2': 0
    }
    
    # ETAPA 1: Alocar equipes às melhores bases
    used_bases = []
    
    # Calcula score de cada base (soma inversa das distâncias - bases mais centrais têm score maior)
    bases_score = []
    for j in range(num_bases):
        score = np.sum(1.0 / (dist_bases_assets[:, j] + 0.1))
        bases_score.append((j, score))
    
    # Ordena bases pelo score (melhores primeiro)
    bases_score.sort(key=lambda x: x[1], reverse=True)
    
    # Aloca cada equipe à melhor base disponível
    for k in range(num_teams):
        base_k = bases_score[k][0]
        solution['y'][base_k, k] = 1
        used_bases.append(base_k)
    
    # ETAPA 2: Distribuir ativos entre as equipes de forma equilibrada
    
    # Calcula distribuição target para cada equipe
    assets_per_team_target = []
    base_per_team = num_assets // num_teams
    extras = num_assets % num_teams
    min_guaranteed = max(int(min_assets_per_team), base_per_team)
    
    for k in range(num_teams):
        target = max(min_guaranteed, base_per_team + (1 if k < extras else 0))
        assets_per_team_target.append(target)
    
    print(f"Usando {num_teams} equipes com distribuição: {assets_per_team_target}")
    
    assigned_assets = set()
    
    # Atribui ativos mais próximos a cada equipe
    for k in range(num_teams):
        base_k = used_bases[k]
        target_count = assets_per_team_target[k]
        
        # Encontra ativos mais próximos desta base
        distances_base_k = []
        for i in range(num_assets):
            if i not in assigned_assets:
                dist = dist_bases_assets[i, base_k]
                distances_base_k.append((i, dist))
        
        distances_base_k.sort(key=lambda x: x[1])
        
        # Atribui os ativos mais próximos
        count = 0
        for asset_idx, _ in distances_base_k:
            if count >= target_count:
                break
            
            solution['h'][asset_idx, k] = 1
            solution['x'][asset_idx, base_k] = 1
            assigned_assets.add(asset_idx)
            count += 1
        
        print(f"Equipe {k+1} (Base {base_k+1}): {count} ativos atribuídos")
    
    # ETAPA 3: Atribuir ativos restantes à equipe com menos ativos
    remaining_assets = [i for i in range(num_assets) if i not in assigned_assets]
    
    for asset_idx in remaining_assets:
        # Encontra equipe com menos ativos
        min_assets = float('inf')
        chosen_team = 0
        
        for k in range(num_teams):
            count_assets = np.sum(solution['h'][:, k])
            if count_assets < min_assets:
                min_assets = count_assets
                chosen_team = k
        
        team_base = used_bases[chosen_team]
        solution['h'][asset_idx, chosen_team] = 1
        solution['x'][asset_idx, team_base] = 1
    
    # ETAPA 4: Verificar consistência interna
    constraint_ok = True
    
    for i in range(num_assets):
        for k in range(num_teams):
            if solution['h'][i, k] == 1:
                asset_base = np.where(solution['x'][i, :] == 1)[0][0]
                team_base = np.where(solution['y'][:, k] == 1)[0][0]
                
                if asset_base != team_base:
                    constraint_ok = False
                    print(f"ERRO: Ativo {i} na base {asset_base}, equipe {k} na base {team_base}")
                    break
        if not constraint_ok:
            break
    
    if not constraint_ok:
        print("ERRO: Inconsistência na atribuição ativo-equipe-base")
        return None
    
    # ETAPA 5: Validar todas as constraints
    for constraint in constraints:
        if not constraint(solution):
            print(f"ERRO: Constraint {constraint.__name__} não satisfeita")
            return None
    
    # ETAPA 6: Calcular objetivos e retornar solução válida
    objective_function_1(solution, dist_bases_assets)
    objective_function_2(solution)
    return solution
    
 