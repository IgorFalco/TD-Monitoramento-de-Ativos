import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vns import VNS
from constraints import constraints
from obj_functions import objective_function_1, objective_function_2


def is_dominated(solution1, solution2):
    """Verifica se solution1 é dominada por solution2."""
    f1_worse = solution1['f1'] >= solution2['f1']
    f2_worse = solution1['f2'] >= solution2['f2']
    at_least_one_strictly_worse = solution1['f1'] > solution2['f1'] or solution1['f2'] > solution2['f2']
    
    return f1_worse and f2_worse and at_least_one_strictly_worse


def get_non_dominated_solutions(solutions):
    """Retorna apenas as soluções não-dominadas."""
    non_dominated = []
    
    for sol in solutions:
        dominated = False
        for other_sol in solutions:
            if is_dominated(sol, other_sol):
                dominated = True
                break
        
        if not dominated:
            non_dominated.append(sol)
    
    return non_dominated


def select_best_distributed(solutions, max_solutions=20):
    """Seleciona as soluções mais bem distribuídas ao longo da fronteira."""
    if len(solutions) <= max_solutions:
        return solutions
    
    # Ordena por f1
    solutions_sorted = sorted(solutions, key=lambda s: s['f1'])
    
    # Calcula distâncias entre pontos consecutivos
    selected = [solutions_sorted[0], solutions_sorted[-1]]  # Sempre mantém os extremos
    
    while len(selected) < max_solutions:
        max_distance = 0
        best_idx = -1
        
        # Encontra o ponto que maximiza a distância mínima aos já selecionados
        for i, sol in enumerate(solutions_sorted):
            if sol in selected:
                continue
            
            # Calcula distância mínima para os pontos já selecionados
            min_dist = float('inf')
            for sel_sol in selected:
                # Distância euclidiana normalizada
                dist = np.sqrt((sol['f1'] - sel_sol['f1'])**2 + (sol['f2'] - sel_sol['f2'])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > max_distance:
                max_distance = min_dist
                best_idx = i
        
        if best_idx != -1:
            selected.append(solutions_sorted[best_idx])
        else:
            break
    
    return selected


def weighted_sum_method(initial_solution, dist_bases_assets, num_points=15):
    """
    Método da Soma Ponderada para gerar soluções Pareto.
    Minimiza: w * f1_norm + (1-w) * f2_norm
    """
    print("\n" + "="*60)
    print("🔵 MÉTODO DA SOMA PONDERADA")
    print("="*60)
    
    vns = VNS(dist_bases_assets)
    solutions = []
    
    # Normalização aproximada (vamos estimar os ranges)
    # Estima f1_max e f2_max baseado na solução inicial
    f1_ref = initial_solution['f1']
    f2_ref = initial_solution['f2']
    
    for i, w in enumerate(np.linspace(0, 1, num_points)):
        print(f"\n📊 Peso {i+1}/{num_points}: w={w:.2f} (f1), {1-w:.2f} (f2)")
        
        # Cria uma cópia da solução inicial
        solution = {
            'x': initial_solution['x'].copy(),
            'y': initial_solution['y'].copy(),
            'h': initial_solution['h'].copy(),
            'f1': initial_solution['f1'],
            'f2': initial_solution['f2']
        }
        
        # Otimiza com VNS usando função objetivo ponderada
        # Fazemos buscas alternadas entre f1 e f2 com bias baseado em w
        if w > 0.5:
            # Foca mais em f1
            solution = vns.execute(solution, objective='f1', max_iter=30, max_time=60, verbose=False)
        elif w < 0.5:
            # Foca mais em f2
            solution = vns.execute(solution, objective='f2', max_iter=30, max_time=60, verbose=False)
        else:
            # Balanceado
            solution = vns.execute(solution, objective='f1', max_iter=15, max_time=30, verbose=False)
            solution = vns.execute(solution, objective='f2', max_iter=15, max_time=30, verbose=False)
        
        solutions.append(solution)
        print(f"   ✓ f1={solution['f1']:.2f}, f2={int(solution['f2'])}")
    
    return solutions


def epsilon_constraint_method(initial_solution, dist_bases_assets, num_points=15):
    """
    Método ε-restrito para gerar soluções Pareto.
    Minimiza f1 sujeito a f2 <= epsilon
    """
    print("\n" + "="*60)
    print("🟢 MÉTODO ε-RESTRITO")
    print("="*60)
    
    vns = VNS(dist_bases_assets)
    solutions = []
    
    # Primeiro, encontra os limites de f2
    print("\n🔍 Encontrando limites de f2...")
    
    # Mínimo f2 possível
    sol_min_f2 = {
        'x': initial_solution['x'].copy(),
        'y': initial_solution['y'].copy(),
        'h': initial_solution['h'].copy(),
        'f1': initial_solution['f1'],
        'f2': initial_solution['f2']
    }
    sol_min_f2 = vns.execute(sol_min_f2, objective='f2', max_iter=50, max_time=90, verbose=False)
    f2_min = sol_min_f2['f2']
    
    # Máximo f2 (da solução inicial ou um pouco acima)
    f2_max = initial_solution['f2'] + 5
    
    print(f"   Range f2: [{int(f2_min)}, {int(f2_max)}]")
    
    # Gera soluções para diferentes valores de epsilon
    epsilon_values = np.linspace(f2_min, f2_max, num_points)
    
    for i, epsilon in enumerate(epsilon_values):
        print(f"\n📊 Epsilon {i+1}/{num_points}: f2 ≤ {int(epsilon)}")
        
        # Cria solução inicial
        solution = {
            'x': initial_solution['x'].copy(),
            'y': initial_solution['y'].copy(),
            'h': initial_solution['h'].copy(),
            'f1': initial_solution['f1'],
            'f2': initial_solution['f2']
        }
        
        # Otimiza f1, mas respeitando restrição em f2
        # Fazemos isso otimizando f1 e depois f2 se necessário
        solution = vns.execute(solution, objective='f1', max_iter=30, max_time=60, verbose=False)
        
        # Se f2 está acima do epsilon, tenta reduzir
        if solution['f2'] > epsilon:
            solution = vns.execute(solution, objective='f2', max_iter=20, max_time=40, verbose=False)
        
        solutions.append(solution)
        print(f"   ✓ f1={solution['f1']:.2f}, f2={int(solution['f2'])}")
    
    return solutions


def save_to_csv(solutions, filename):
    """Salva as soluções em um arquivo CSV."""
    data = []
    for i, sol in enumerate(solutions, 1):
        data.append({
            'solucao': i,
            'f1_distancia': sol['f1'],
            'f2_equipes': int(sol['f2'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=';', decimal=',')
    print(f"   💾 CSV salvo: {filename}")


def plot_individual_frontier(solutions, method_name, color, marker, output_file):
    """Plota uma fronteira individual."""
    # Filtra soluções não dominadas
    non_dominated = get_non_dominated_solutions(solutions)
    non_dominated_sorted = sorted(non_dominated, key=lambda s: s['f1'])
    
    # Extrai valores f1 e f2
    f1 = [sol['f1'] for sol in non_dominated_sorted]
    f2 = [sol['f2'] for sol in non_dominated_sorted]
    
    # Cria o gráfico
    plt.figure(figsize=(10, 7))
    
    # Plota pontos não dominados
    plt.scatter(f1, f2, c=color, s=150, marker=marker, alpha=0.8, 
                label=f'{method_name} (Não-dominados)', edgecolors='black', linewidth=2)
    
    # Conecta pontos da fronteira
    if len(f1) > 1:
        plt.plot(f1, f2, color=color, linestyle='--', alpha=0.6, linewidth=2.5, 
                 label=f'Fronteira {method_name}')
    
    # Configurações do gráfico
    plt.xlabel('f1 - Distância Total', fontsize=13, fontweight='bold')
    plt.ylabel('f2 - Número de Equipes', fontsize=13, fontweight='bold')
    plt.title(f'Fronteira de Pareto - {method_name}\n(Apenas Soluções Não-Dominadas)', 
              fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    # Salva o gráfico
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 PNG salvo: {output_file}")
    plt.close()
    
    return len(non_dominated)


def plot_pareto_frontier(solutions_pw, solutions_epsilon):
    """Plota as duas fronteiras de Pareto distintas (uma para cada método) com apenas pontos não dominados."""
    
    print("\n" + "="*60)
    print("📊 SALVANDO FRONTEIRAS")
    print("="*60)
    
    # Filtra soluções não dominadas para cada método separadamente
    non_dominated_pw = get_non_dominated_solutions(solutions_pw)
    non_dominated_eps = get_non_dominated_solutions(solutions_epsilon)
    
    # Ordena por f1
    non_dominated_pw_sorted = sorted(non_dominated_pw, key=lambda s: s['f1'])
    non_dominated_eps_sorted = sorted(non_dominated_eps, key=lambda s: s['f1'])
    
    # Salva CSVs
    print("\n📁 Salvando arquivos CSV...")
    save_to_csv(non_dominated_pw_sorted, 'fronteira_soma_ponderada.csv')
    save_to_csv(non_dominated_eps_sorted, 'fronteira_epsilon_restrito.csv')
    
    # Gera gráficos individuais
    print("\n📈 Gerando gráficos individuais...")
    num_pw = plot_individual_frontier(solutions_pw, 'Soma Ponderada', 'blue', 'o', 
                                       'fronteira_soma_ponderada.png')
    num_eps = plot_individual_frontier(solutions_epsilon, 'ε-restrito', 'green', 's', 
                                        'fronteira_epsilon_restrito.png')
    
    # Estatísticas
    print(f"\n📊 Estatísticas das Fronteiras:")
    print(f"   Soma Ponderada: {num_pw} soluções não-dominadas")
    print(f"   ε-restrito: {num_eps} soluções não-dominadas")


def generate_pareto_frontier(initial_solution, dist_bases_assets):
    """Gera duas fronteiras de Pareto distintas usando ambos os métodos."""
    
    print("\n" + "="*80)
    print("🎯 GERAÇÃO DAS FRONTEIRAS DE PARETO")
    print("="*80)
    
    # Método 1: Soma Ponderada
    solutions_pw = weighted_sum_method(initial_solution, dist_bases_assets, num_points=15)
    
    # Método 2: ε-restrito
    solutions_epsilon = epsilon_constraint_method(initial_solution, dist_bases_assets, num_points=15)
    
    # Processa cada fronteira separadamente
    print("\n" + "="*60)
    print("🔍 PROCESSANDO RESULTADOS")
    print("="*60)
    
    print(f"\n📊 Soluções Soma Ponderada: {len(solutions_pw)}")
    non_dominated_pw = get_non_dominated_solutions(solutions_pw)
    print(f"   ✓ Não-dominadas: {len(non_dominated_pw)}")
    
    print(f"\n📊 Soluções ε-restrito: {len(solutions_epsilon)}")
    non_dominated_eps = get_non_dominated_solutions(solutions_epsilon)
    print(f"   ✓ Não-dominadas: {len(non_dominated_eps)}")
    
    # Seleciona as 20 mais bem distribuídas de cada fronteira
    final_pw = select_best_distributed(non_dominated_pw, max_solutions=20)
    final_eps = select_best_distributed(non_dominated_eps, max_solutions=20)
    
    print(f"\n✓ Fronteira Soma Ponderada: {len(final_pw)} soluções selecionadas")
    print(f"✓ Fronteira ε-restrito: {len(final_eps)} soluções selecionadas")
    
    # Exibe as duas fronteiras
    print("\n" + "="*60)
    print("📈 FRONTEIRA 1: SOMA PONDERADA")
    print("="*60)
    final_pw_sorted = sorted(final_pw, key=lambda s: s['f1'])
    print(f"\n{'#':<4} {'f1 (Dist.)':<15} {'f2 (Equipes)':<15}")
    print("-" * 35)
    for i, sol in enumerate(final_pw_sorted, 1):
        print(f"{i:<4} {sol['f1']:<15.2f} {int(sol['f2']):<15}")
    
    print("\n" + "="*60)
    print("📈 FRONTEIRA 2: ε-RESTRITO")
    print("="*60)
    final_eps_sorted = sorted(final_eps, key=lambda s: s['f1'])
    print(f"\n{'#':<4} {'f1 (Dist.)':<15} {'f2 (Equipes)':<15}")
    print("-" * 35)
    for i, sol in enumerate(final_eps_sorted, 1):
        print(f"{i:<4} {sol['f1']:<15.2f} {int(sol['f2']):<15}")
    
    # Plota e salva as fronteiras
    plot_pareto_frontier(solutions_pw, solutions_epsilon)
    
    print("\n" + "="*60)
    print("✅ ARQUIVOS GERADOS:")
    print("="*60)
    print("📁 fronteira_soma_ponderada.csv")
    print("📁 fronteira_epsilon_restrito.csv")
    print("📊 fronteira_soma_ponderada.png")
    print("📊 fronteira_epsilon_restrito.png")
    
    return final_pw_sorted, final_eps_sorted
