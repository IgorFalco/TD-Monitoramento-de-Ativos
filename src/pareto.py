import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vns import VNS
from constraints import constraints
from obj_functions import objective_function_1, objective_function_2

# Pasta de saída para resultados
RESULTS_DIR = 'result'

def _ensure_results_dir():
    """Garante que a pasta de resultados exista."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


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
    """Soma Ponderada: minimiza F = w1*f1_norm + w2*f2_norm (w1+w2=1), com normalização dos objetivos."""
    print("\n" + "="*60)
    print("🔵 MÉTODO DA SOMA PONDERADA")
    print("="*60)
    
    vns = VNS(dist_bases_assets)
    solutions = []
    
    # Calcula valores de referência para normalização
    # f2 em equipes: domínio inteiro [1, 8]
    f2_min, f2_max = 1, 8
    # f1: usa solução inicial como f1_max e estima f1_min com uma execução curta focada em f1
    f1_max = initial_solution['f1']
    sol_tmp = {
        'x': initial_solution['x'].copy(),
        'y': initial_solution['y'].copy(),
        'h': initial_solution['h'].copy(),
        'f1': initial_solution['f1'],
        'f2': initial_solution['f2']
    }
    sol_best_f1 = vns.execute(sol_tmp, objective='f1', verbose=False)
    f1_min = sol_best_f1['f1']
    
    print(f"\n📏 Normalização:")
    print(f"   f1_min (distância) = {f1_min:.2f}")
    print(f"   f1_max (distância) = {f1_max:.2f}")
    print(f"   f2_min (equipes)   = {f2_min}")
    print(f"   f2_max (equipes)   = {f2_max}")
    
    # Normalização dos pesos: w1 + w2 = 1
    for i, w1 in enumerate(np.linspace(0, 1, num_points)):
        w2 = 1 - w1
        
        print(f"\n📊 Peso {i+1}/{num_points}: w1={w1:.3f}, w2={w2:.3f}")
        print(f"   Função: F(x) = {w1:.3f}*f1_norm + {w2:.3f}*f2_norm")
        
        # Cria uma cópia da solução inicial
        solution = {
            'x': initial_solution['x'].copy(),
            'y': initial_solution['y'].copy(),
            'h': initial_solution['h'].copy(),
            'f1': initial_solution['f1'],
            'f2': initial_solution['f2']
        }
        
        # Otimiza usando VNS com função objetivo ponderada normalizada
        solution = vns.execute(
            solution, 
            objective='weighted',
            w1=w1,
            w2=w2,
            f1_min=f1_min,
            f1_max=f1_max,
            f2_min=f2_min,
            f2_max=f2_max,
            verbose=False
        )
        # Guarda os pesos que geraram esta solução (para salvar no CSV)
        solution['w1'] = float(w1)
        solution['w2'] = float(w2)
        
        solutions.append(solution)
        print(f"   ✓ f1={solution['f1']:.2f}, f2={int(solution['f2'])}")
    
    return solutions


def epsilon_constraint_method(initial_solution, dist_bases_assets, num_points=8):
    """ε-Restrito: minimiza f1 sujeito a f2 ≤ ε (ε inteiro de 1 a 8)."""
    print("\n" + "="*60)
    print("🟢 MÉTODO ε-RESTRITO")
    print("="*60)
    
    vns = VNS(dist_bases_assets)
    solutions = []
    
    # f2 representa número de equipes: valores inteiros de 1 a 8
    f2_min = 1
    f2_max = 8
    
    print(f"\n🔍 Range de equipes (f2): [{f2_min}, {f2_max}]")
    
    # Gera valores inteiros de epsilon de 1 a 8
    epsilon_values = np.arange(f2_min, f2_max + 1, dtype=int)
    
    for i, epsilon in enumerate(epsilon_values, 1):
        print(f"\n📊 Epsilon {i}/{len(epsilon_values)}: f2 ≤ {int(epsilon)}")
        print(f"   Minimiza: f1(x) sujeito a f2(x) ≤ {int(epsilon)}")
        
        # Cria solução inicial
        solution = {
            'x': initial_solution['x'].copy(),
            'y': initial_solution['y'].copy(),
            'h': initial_solution['h'].copy(),
            'f1': initial_solution['f1'],
            'f2': initial_solution['f2']
        }
        
        # Primeiro, garante que f2 <= epsilon
        if solution['f2'] > epsilon:
            # Reduz f2 até o limite epsilon
            solution = vns.execute(
                solution, 
                objective='f2', 
                verbose=False
            )
        
        # Agora otimiza f1 respeitando a restrição f2 <= epsilon
        solution = vns.execute(
            solution, 
            objective='f1',
            epsilon=epsilon,  # Passa o limite para f2
            verbose=False
        )
        
        solutions.append(solution)
        constraint_ok = "✓" if solution['f2'] <= epsilon else "✗"
        print(f"   {constraint_ok} f1={solution['f1']:.2f}, f2={int(solution['f2'])} (limite: {int(epsilon)})")
    
    return solutions


def save_to_csv(solutions, filename):
    """Salva as soluções em um arquivo CSV."""
    # Garante a pasta do arquivo
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    # Verifica se existem pesos nas soluções
    include_weights = any(('w1' in sol and 'w2' in sol) for sol in solutions)
    data = []
    for i, sol in enumerate(solutions, 1):
        row = {
            'solucao': i,
            'f1_distancia': sol['f1'],
            'f2_equipes': int(sol['f2'])
        }
        if include_weights:
            row['peso_w1'] = float(sol.get('w1', np.nan))
            row['peso_w2'] = float(sol.get('w2', np.nan))
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=';', decimal=',')
    print(f"   💾 CSV salvo: {filename}")


def plot_individual_frontier(solutions, method_name, color, marker, output_file):
    """Plota uma fronteira individual."""
    # Garante a pasta do arquivo
    dirpath = os.path.dirname(output_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
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
    _ensure_results_dir()
    # Sufixo numérico aleatório para evitar conflitos em execuções paralelas
    run_id = f"{int(time.time())}{random.randint(1000, 9999)}"
    
    # Filtra soluções não dominadas para cada método separadamente
    non_dominated_pw = get_non_dominated_solutions(solutions_pw)
    non_dominated_eps = get_non_dominated_solutions(solutions_epsilon)
    
    # Ordena por f1
    non_dominated_pw_sorted = sorted(non_dominated_pw, key=lambda s: s['f1'])
    non_dominated_eps_sorted = sorted(non_dominated_eps, key=lambda s: s['f1'])
    
    # Salva CSVs
    print("\n📁 Salvando arquivos CSV...")
    csv_pw = os.path.join(RESULTS_DIR, f'fronteira_soma_ponderada_{run_id}.csv')
    csv_eps = os.path.join(RESULTS_DIR, f'fronteira_epsilon_restrito_{run_id}.csv')
    save_to_csv(non_dominated_pw_sorted, csv_pw)
    save_to_csv(non_dominated_eps_sorted, csv_eps)
    
    # Gera gráficos individuais
    print("\n📈 Gerando gráficos individuais...")
    png_pw = os.path.join(RESULTS_DIR, f'fronteira_soma_ponderada_{run_id}.png')
    png_eps = os.path.join(RESULTS_DIR, f'fronteira_epsilon_restrito_{run_id}.png')
    num_pw = plot_individual_frontier(solutions_pw, 'Soma Ponderada', 'blue', 'o', png_pw)
    num_eps = plot_individual_frontier(solutions_epsilon, 'ε-restrito', 'green', 's', png_eps)
    
    # Estatísticas
    print(f"\n📊 Estatísticas das Fronteiras:")
    print(f"   Soma Ponderada: {num_pw} soluções não-dominadas")
    print(f"   ε-restrito: {num_eps} soluções não-dominadas")

    # Resumo dos arquivos gerados com sufixo aleatório
    print("\n" + "="*60)
    print("✅ ARQUIVOS GERADOS:")
    print("="*60)
    print(f"📁 {csv_pw}")
    print(f"📁 {csv_eps}")
    print(f"📊 {png_pw}")
    print(f"📊 {png_eps}")


# def generate_pareto_frontier(initial_solution, dist_bases_assets):
#     """Gera duas fronteiras de Pareto distintas usando ambos os métodos."""
    
#     print("\n" + "="*80)
#     print("🎯 GERAÇÃO DAS FRONTEIRAS DE PARETO")
#     print("="*80)
    
#     # Método 1: Soma Ponderada
#     solutions_pw = weighted_sum_method(initial_solution, dist_bases_assets, num_points=15)
    
#     # Método 2: ε-restrito
#     solutions_epsilon = epsilon_constraint_method(initial_solution, dist_bases_assets, num_points=15)
    
#     # Processa cada fronteira separadamente
#     print("\n" + "="*60)
#     print("🔍 PROCESSANDO RESULTADOS")
#     print("="*60)
    
#     print(f"\n📊 Soluções Soma Ponderada: {len(solutions_pw)}")
#     non_dominated_pw = get_non_dominated_solutions(solutions_pw)
#     print(f"   ✓ Não-dominadas: {len(non_dominated_pw)}")
    
#     print(f"\n📊 Soluções ε-restrito: {len(solutions_epsilon)}")
#     non_dominated_eps = get_non_dominated_solutions(solutions_epsilon)
#     print(f"   ✓ Não-dominadas: {len(non_dominated_eps)}")
    
#     # Seleciona as 20 mais bem distribuídas de cada fronteira
#     final_pw = select_best_distributed(non_dominated_pw, max_solutions=20)
#     final_eps = select_best_distributed(non_dominated_eps, max_solutions=20)
    
#     print(f"\n✓ Fronteira Soma Ponderada: {len(final_pw)} soluções selecionadas")
#     print(f"✓ Fronteira ε-restrito: {len(final_eps)} soluções selecionadas")
    
#     # Exibe as duas fronteiras
#     print("\n" + "="*60)
#     print("📈 FRONTEIRA 1: SOMA PONDERADA")
#     print("="*60)
#     final_pw_sorted = sorted(final_pw, key=lambda s: s['f1'])
#     print(f"\n{'#':<4} {'f1 (Dist.)':<15} {'f2 (Equipes)':<15}")
#     print("-" * 35)
#     for i, sol in enumerate(final_pw_sorted, 1):
#         print(f"{i:<4} {sol['f1']:<15.2f} {int(sol['f2']):<15}")
    
#     print("\n" + "="*60)
#     print("📈 FRONTEIRA 2: ε-RESTRITO")
#     print("="*60)
#     final_eps_sorted = sorted(final_eps, key=lambda s: s['f1'])
#     print(f"\n{'#':<4} {'f1 (Dist.)':<15} {'f2 (Equipes)':<15}")
#     print("-" * 35)
#     for i, sol in enumerate(final_eps_sorted, 1):
#         print(f"{i:<4} {sol['f1']:<15.2f} {int(sol['f2']):<15}")
    
#     # Plota e salva as fronteiras (imprime nomes com sufixo aleatório internamente)
#     plot_pareto_frontier(solutions_pw, solutions_epsilon)
        
#     return final_pw_sorted, final_eps_sorted

def generate_pareto_frontier(initial_solution, dist_bases_assets):
    """Gera 5 execuções e overlays para cada método; ≤20 pontos por execução nos overlays."""
    print("\n" + "="*80)
    print("🎯 GERAÇÃO DAS FRONTEIRAS DE PARETO (5 EXECUÇÕES)")
    print("="*80)

    _ensure_results_dir()
    overall_run_id = f"{int(time.time())}{random.randint(1000, 9999)}"

    all_runs_pw = []
    all_runs_eps = []

    runs = 5
    for r in range(1, runs + 1):
        print(f"\n--- Execução {r}/{runs} ---")
        # Método 1: Soma Ponderada
        solutions_pw = weighted_sum_method(initial_solution, dist_bases_assets, num_points=15)
        # Método 2: ε-restrito
        solutions_epsilon = epsilon_constraint_method(initial_solution, dist_bases_assets, num_points=8)

        # Processa cada fronteira separadamente e salva arquivos individuais desta execução
        print("\n" + "="*60)
        print("🔍 PROCESSANDO RESULTADOS")
        print("="*60)
        print(f"\n📊 Soluções Soma Ponderada: {len(solutions_pw)}")
        non_dominated_pw = get_non_dominated_solutions(solutions_pw)
        print(f"   ✓ Não-dominadas: {len(non_dominated_pw)}")
        print(f"\n📊 Soluções ε-restrito: {len(solutions_epsilon)}")
        non_dominated_eps = get_non_dominated_solutions(solutions_epsilon)
        print(f"   ✓ Não-dominadas: {len(non_dominated_eps)}")
        # Seleciona 20 mais bem distribuídas de cada fronteira
        final_pw = select_best_distributed(non_dominated_pw, max_solutions=20)
        final_eps = select_best_distributed(non_dominated_eps, max_solutions=20)
        print(f"\n✓ Fronteira Soma Ponderada: {len(final_pw)} soluções selecionadas")
        print(f"✓ Fronteira ε-restrito: {len(final_eps)} soluções selecionadas")

        # Exibe soluções desta execução (opcional; mantém para consistência)
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

        # Salva CSVs/PNGs individuais desta execução
        plot_pareto_frontier(solutions_pw, solutions_epsilon)

        # Acumula para overlays
        all_runs_pw.append(solutions_pw)
        all_runs_eps.append(solutions_epsilon)

    # Gera overlays com ≤20 pontos por execução
    overlay_pw = os.path.join(RESULTS_DIR, f'fronteira_soma_ponderada_overlay_{overall_run_id}.png')
    overlay_eps = os.path.join(RESULTS_DIR, f'fronteira_epsilon_restrito_overlay_{overall_run_id}.png')

    # Garante a pasta
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot overlay para Soma Ponderada
    plt.figure(figsize=(10, 7))
    alphas = [0.9, 0.7, 0.5, 0.35, 0.25]
    colors = ['green', 'red', 'blue', 'yellow', 'orange']
    for idx, sols in enumerate(all_runs_pw, start=1):
        non_dominated = get_non_dominated_solutions(sols)
        selected = select_best_distributed(non_dominated, max_solutions=20)
        f1 = [s['f1'] for s in selected]
        f2 = [s['f2'] for s in selected]
        label = f"Execução {idx}"
        color = colors[(idx-1) % len(colors)]
        plt.scatter(f1, f2, c=color, s=120, marker='o', alpha=alphas[(idx-1) % len(alphas)],
                    label=label, edgecolors='black', linewidth=1.5)
        if len(f1) > 1:
            ordered = sorted(zip(f1, f2))
            plt.plot([p[0] for p in ordered], [p[1] for p in ordered], color=color, linestyle='--', alpha=0.4)
    plt.xlabel('f1 - Distância Total', fontsize=13, fontweight='bold')
    plt.ylabel('f2 - Número de Equipes', fontsize=13, fontweight='bold')
    plt.title('Fronteiras de Pareto - Soma Ponderada\n(5 execuções, ≤20 pts por execução)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(overlay_pw, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 PNG salvo (overlay Soma Ponderada): {overlay_pw}")

    # Plot overlay para ε-restrito
    plt.figure(figsize=(10, 7))
    for idx, sols in enumerate(all_runs_eps, start=1):
        non_dominated = get_non_dominated_solutions(sols)
        selected = select_best_distributed(non_dominated, max_solutions=20)
        f1 = [s['f1'] for s in selected]
        f2 = [s['f2'] for s in selected]
        label = f"Execução {idx}"
        color = colors[(idx-1) % len(colors)]
        plt.scatter(f1, f2, c=color, s=120, marker='s', alpha=alphas[(idx-1) % len(alphas)],
                    label=label, edgecolors='black', linewidth=1.5)
        if len(f1) > 1:
            ordered = sorted(zip(f1, f2))
            plt.plot([p[0] for p in ordered], [p[1] for p in ordered], color=color, linestyle='--', alpha=0.4)
    plt.xlabel('f1 - Distância Total', fontsize=13, fontweight='bold')
    plt.ylabel('f2 - Número de Equipes', fontsize=13, fontweight='bold')
    plt.title('Fronteiras de Pareto - ε-restrito\n(5 execuções, ≤20 pts por execução)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(overlay_eps, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 PNG salvo (overlay ε-restrito): {overlay_eps}")

    print("\n" + "="*60)
    print("✅ OVERLAYS GERADOS:")
    print("="*60)
    print(f"📊 {overlay_pw}")
    print(f"📊 {overlay_eps}")

    return all_runs_pw, all_runs_eps
