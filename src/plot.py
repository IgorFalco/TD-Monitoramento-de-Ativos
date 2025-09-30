import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

def plot_solution(solution, probdata_path='probdata.csv'):
    """
    Plota a solução do problema de monitoramento de ativos.
    
    Args:
        solution: Dicionário com a solução (x, y, h, f1, f2)
        probdata_path: Caminho para o arquivo de dados
    """
    
    # Lê os dados para obter coordenadas
    df = pd.read_csv(probdata_path, delimiter=';', header=None, decimal=',')
    df.columns = ['latitude_base', 'longitude_base', 'latitude_ativo', 'longitude_ativo', 'distancia']
    
    # Extrai bases e ativos únicos
    bases = df[['latitude_base', 'longitude_base']].drop_duplicates().reset_index(drop=True)
    ativos = df[['latitude_ativo', 'longitude_ativo']].drop_duplicates().reset_index(drop=True)
    
    # Cria o gráfico
    plt.figure(figsize=(12, 10))
    
    # Plota bases não ocupadas
    plt.scatter(bases['longitude_base'], bases['latitude_base'], 
               c='lightgray', s=100, marker='s', alpha=0.7, 
               label='Bases não ocupadas', edgecolors='black', linewidth=0.5)
    
    # Plota todos os ativos
    plt.scatter(ativos['longitude_ativo'], ativos['latitude_ativo'], 
               c='lightblue', s=60, marker='o', alpha=0.8,
               label='Ativos', edgecolors='darkblue', linewidth=0.5)
    
    # Cores para as equipes
    cores_equipes = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Para cada equipe ativa
    num_equipes = solution['y'].shape[1]
    for k in range(num_equipes):
        # Verifica se a equipe está ativa (tem alguma base)
        if np.sum(solution['y'][:, k]) > 0:
            # Encontra a base da equipe
            base_equipe = np.where(solution['y'][:, k] == 1)[0][0]
            
            # Encontra ativos da equipe
            ativos_equipe = np.where(solution['h'][:, k] == 1)[0]
            
            cor = cores_equipes[k % len(cores_equipes)]
            
            # Plota a base ocupada com cor da equipe
            plt.scatter(bases.iloc[base_equipe]['longitude_base'], 
                       bases.iloc[base_equipe]['latitude_base'],
                       c=cor, s=200, marker='s', 
                       label=f'Base Equipe {k+1}', edgecolors='black', linewidth=2)
            
            # Plota os ativos da equipe com a mesma cor
            for ativo_idx in ativos_equipe:
                plt.scatter(ativos.iloc[ativo_idx]['longitude_ativo'], 
                           ativos.iloc[ativo_idx]['latitude_ativo'],
                           c=cor, s=80, marker='o', alpha=0.9,
                           edgecolors='black', linewidth=1)
                
                # Desenha linha conectando base ao ativo
                plt.plot([bases.iloc[base_equipe]['longitude_base'], 
                         ativos.iloc[ativo_idx]['longitude_ativo']],
                        [bases.iloc[base_equipe]['latitude_base'], 
                         ativos.iloc[ativo_idx]['latitude_ativo']],
                        color=cor, alpha=0.6, linewidth=1.5, linestyle='-')
    
    # Configurações do gráfico
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Solução do Problema de Monitoramento de Ativos\n'
              f'Distância Total: {solution["f1"]:.2f} | Número de Equipes: {int(solution["f2"])}')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Mostra estatísticas
    bases_ocupadas = np.where(np.sum(solution['y'], axis=1) > 0)[0]
    
    print(f"\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Função Objetivo 1 (Distância Total): {solution['f1']:.2f}")
    print(f"Função Objetivo 2 (Número de Equipes): {int(solution['f2'])}")
    print(f"Bases Ocupadas: {len(bases_ocupadas)}")
    print(f"Total de Ativos: {len(ativos)}")
    print(f"Total de Bases: {len(bases)}")
    
    # Detalhes por equipe
    print(f"\n=== DETALHES POR EQUIPE ===")
    for k in range(num_equipes):
        if np.sum(solution['y'][:, k]) > 0:
            base_equipe = np.where(solution['y'][:, k] == 1)[0][0]
            ativos_equipe = np.sum(solution['h'][:, k])
            print(f"Equipe {k+1}: Base {base_equipe+1} - {int(ativos_equipe)} ativos")
    
    plt.show()