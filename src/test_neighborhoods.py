"""
Script para testar o impacto de cada estrutura de vizinhança.
Testa todas as combinações removendo 1, 2 e 3 vizinhanças.
Executa 5 vezes cada configuração e calcula a média dos resultados.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from generator import read_csv_data, generate_initial_solution
from vns import VNS
from neighborhoods import (
    neighborhood_swap_assets,
    neighborhood_relocate_asset,
    neighborhood_swap_bases,
    neighborhood_relocate_base,
    neighborhood_or_opt,
    neighborhood_change_team_number
)
import time

# Dicionário com todas as vizinhanças disponíveis
ALL_NEIGHBORHOODS = {
    'swap_assets': neighborhood_swap_assets,
    'relocate_asset': neighborhood_relocate_asset,
    'swap_bases': neighborhood_swap_bases,
    'relocate_base': neighborhood_relocate_base,
    'or_opt': neighborhood_or_opt,
    'change_team_number': neighborhood_change_team_number
}


def run_vns_with_neighborhoods(initial_solution, dist_bases_assets, neighborhoods, num_runs=5):
    """Executa VNS múltiplas vezes com conjunto específico de vizinhanças."""
    results = []
    
    for run in range(num_runs):
        # Cria uma cópia da solução inicial
        solution = {
            'x': initial_solution['x'].copy(),
            'y': initial_solution['y'].copy(),
            'h': initial_solution['h'].copy(),
            'f1': initial_solution['f1'],
            'f2': initial_solution['f2']
        }
        
        # Cria VNS e substitui as vizinhanças
        vns = VNS(dist_bases_assets)
        vns.neighborhoods = neighborhoods
        
        # Executa VNS
        start_time = time.time()
        optimized = vns.execute(solution, objective='f1', max_iter=30, max_time=60, verbose=False)
        execution_time = time.time() - start_time
        
        results.append({
            'run': run + 1,
            'f1': optimized['f1'],
            'f2': optimized['f2'],
            'time': execution_time,
            'improvement_f1': ((initial_solution['f1'] - optimized['f1']) / initial_solution['f1']) * 100
        })
    
    return results


def test_configurations():
    """Testa todas as combinações de vizinhanças."""
    
    print("="*80)
    print("🧪 TESTE DE ESTRUTURAS DE VIZINHANÇA")
    print("="*80)
    
    # Carrega dados
    print("\n📁 Carregando dados...")
    dist_bases_assets = read_csv_data()
    num_assets, num_bases = dist_bases_assets.shape
    print(f"✓ Ativos: {num_assets}, Bases: {num_bases}")
    
    # Gera solução inicial
    print("\n🔄 Gerando solução inicial...")
    initial_solution = generate_initial_solution(num_assets, num_bases, dist_bases_assets)
    print(f"✓ Solução inicial: f1={initial_solution['f1']:.2f}, f2={int(initial_solution['f2'])}")
    
    neighborhood_names = list(ALL_NEIGHBORHOODS.keys())
    all_test_results = []
    
    print("\n" + "="*80)
    print("📊 INICIANDO TESTES")
    print("="*80)
    
    # 1. Teste com todas as vizinhanças (baseline)
    print("\n🔵 BASELINE: Todas as 6 vizinhanças")
    print("-" * 60)
    all_neighborhoods = list(ALL_NEIGHBORHOODS.values())
    results = run_vns_with_neighborhoods(initial_solution, dist_bases_assets, all_neighborhoods, num_runs=5)
    
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_f2 = np.mean([r['f2'] for r in results])
    avg_improvement = np.mean([r['improvement_f1'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    std_f1 = np.std([r['f1'] for r in results])
    
    print(f"Média f1: {avg_f1:.2f} (±{std_f1:.2f})")
    print(f"Média f2: {avg_f2:.2f}")
    print(f"Melhoria média: {avg_improvement:.2f}%")
    print(f"Tempo médio: {avg_time:.2f}s")
    
    all_test_results.append({
        'config': 'BASELINE (6 vizinhanças)',
        'neighborhoods': ', '.join(neighborhood_names),
        'num_neighborhoods': 6,
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'avg_f2': avg_f2,
        'avg_improvement': avg_improvement,
        'avg_time': avg_time
    })
    
    # 2. Teste removendo 1 vizinhança por vez
    print("\n" + "="*80)
    print("🟡 TESTE: Removendo 1 vizinhança (6 configurações)")
    print("="*80)
    
    for i, name_to_remove in enumerate(neighborhood_names, 1):
        print(f"\n[{i}/6] Removendo: {name_to_remove}")
        print("-" * 60)
        
        # Cria lista sem essa vizinhança
        remaining_names = [n for n in neighborhood_names if n != name_to_remove]
        remaining_neighborhoods = [ALL_NEIGHBORHOODS[n] for n in remaining_names]
        
        results = run_vns_with_neighborhoods(initial_solution, dist_bases_assets, remaining_neighborhoods, num_runs=5)
        
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_f2 = np.mean([r['f2'] for r in results])
        avg_improvement = np.mean([r['improvement_f1'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        
        print(f"Média f1: {avg_f1:.2f} (±{std_f1:.2f})")
        print(f"Média f2: {avg_f2:.2f}")
        print(f"Melhoria média: {avg_improvement:.2f}%")
        print(f"Tempo médio: {avg_time:.2f}s")
        
        all_test_results.append({
            'config': f'Sem {name_to_remove}',
            'neighborhoods': ', '.join(remaining_names),
            'num_neighborhoods': 5,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_f2': avg_f2,
            'avg_improvement': avg_improvement,
            'avg_time': avg_time
        })
    
    # 3. Teste removendo 2 vizinhanças (15 combinações)
    print("\n" + "="*80)
    print("🟠 TESTE: Removendo 2 vizinhanças (15 configurações)")
    print("="*80)
    
    combinations_2 = list(combinations(neighborhood_names, 2))
    for i, names_to_remove in enumerate(combinations_2, 1):
        print(f"\n[{i}/15] Removendo: {', '.join(names_to_remove)}")
        print("-" * 60)
        
        # Cria lista sem essas vizinhanças
        remaining_names = [n for n in neighborhood_names if n not in names_to_remove]
        remaining_neighborhoods = [ALL_NEIGHBORHOODS[n] for n in remaining_names]
        
        results = run_vns_with_neighborhoods(initial_solution, dist_bases_assets, remaining_neighborhoods, num_runs=5)
        
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_f2 = np.mean([r['f2'] for r in results])
        avg_improvement = np.mean([r['improvement_f1'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        
        print(f"Média f1: {avg_f1:.2f} (±{std_f1:.2f})")
        print(f"Média f2: {avg_f2:.2f}")
        print(f"Melhoria média: {avg_improvement:.2f}%")
        print(f"Tempo médio: {avg_time:.2f}s")
        
        all_test_results.append({
            'config': f'Sem {", ".join(names_to_remove)}',
            'neighborhoods': ', '.join(remaining_names),
            'num_neighborhoods': 4,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_f2': avg_f2,
            'avg_improvement': avg_improvement,
            'avg_time': avg_time
        })
    
    # 4. Teste removendo 3 vizinhanças (20 combinações = mantendo apenas 3)
    print("\n" + "="*80)
    print("🔴 TESTE: Mantendo apenas 3 vizinhanças (20 configurações)")
    print("="*80)
    
    combinations_3_keep = list(combinations(neighborhood_names, 3))
    for i, names_to_keep in enumerate(combinations_3_keep, 1):
        print(f"\n[{i}/20] Mantendo: {', '.join(names_to_keep)}")
        print("-" * 60)
        
        # Cria lista com apenas essas vizinhanças
        remaining_neighborhoods = [ALL_NEIGHBORHOODS[n] for n in names_to_keep]
        
        results = run_vns_with_neighborhoods(initial_solution, dist_bases_assets, remaining_neighborhoods, num_runs=5)
        
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_f2 = np.mean([r['f2'] for r in results])
        avg_improvement = np.mean([r['improvement_f1'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        std_f1 = np.std([r['f1'] for r in results])
        
        print(f"Média f1: {avg_f1:.2f} (±{std_f1:.2f})")
        print(f"Média f2: {avg_f2:.2f}")
        print(f"Melhoria média: {avg_improvement:.2f}%")
        print(f"Tempo médio: {avg_time:.2f}s")
        
        all_test_results.append({
            'config': f'Apenas {", ".join(names_to_keep)}',
            'neighborhoods': ', '.join(names_to_keep),
            'num_neighborhoods': 3,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_f2': avg_f2,
            'avg_improvement': avg_improvement,
            'avg_time': avg_time
        })
    
    # Salva resultados
    print("\n" + "="*80)
    print("💾 SALVANDO RESULTADOS")
    print("="*80)
    
    df = pd.DataFrame(all_test_results)
    df = df.sort_values('avg_f1')
    df.to_csv('test_neighborhoods_results.csv', index=False, sep=';', decimal=',')
    print("✓ Resultados salvos em: test_neighborhoods_results.csv")
    
    # Análise de resultados
    print("\n" + "="*80)
    print("📊 ANÁLISE DOS RESULTADOS")
    print("="*80)
    
    baseline_f1 = all_test_results[0]['avg_f1']
    
    print("\n🏆 TOP 10 MELHORES CONFIGURAÇÕES:")
    print("-" * 80)
    print(f"{'Rank':<6} {'f1':<12} {'f2':<8} {'Melhoria':<12} {'Tempo':<10} {'Vizinhanças':<20}")
    print("-" * 80)
    
    for i, row in df.head(10).iterrows():
        print(f"{i+1:<6} {row['avg_f1']:<12.2f} {row['avg_f2']:<8.2f} {row['avg_improvement']:<12.2f}% "
              f"{row['avg_time']:<10.2f}s {row['num_neighborhoods']:<20}")
    
    print("\n🎯 MELHORES CONFIGURAÇÕES COM 3 VIZINHANÇAS:")
    print("-" * 80)
    df_3 = df[df['num_neighborhoods'] == 3].head(5)
    
    for idx, (i, row) in enumerate(df_3.iterrows(), 1):
        print(f"\n{idx}. {row['config']}")
        print(f"   f1: {row['avg_f1']:.2f} (±{row['std_f1']:.2f})")
        print(f"   f2: {row['avg_f2']:.2f}")
        print(f"   Melhoria: {row['avg_improvement']:.2f}%")
        print(f"   Tempo: {row['avg_time']:.2f}s")
        print(f"   Vizinhanças: {row['neighborhoods']}")
    
    print("\n❌ VIZINHANÇAS COM MENOR IMPACTO (aparecem menos no top 10 com 3 vizinhanças):")
    print("-" * 80)
    
    # Conta frequência de cada vizinhança no top 10 com 3 vizinhanças
    top_10_with_3 = df[df['num_neighborhoods'] == 3].head(10)
    neighborhood_count = {name: 0 for name in neighborhood_names}
    
    for _, row in top_10_with_3.iterrows():
        for name in neighborhood_names:
            if name in row['neighborhoods']:
                neighborhood_count[name] += 1
    
    sorted_neighborhoods = sorted(neighborhood_count.items(), key=lambda x: x[1])
    
    for name, count in sorted_neighborhoods:
        percentage = (count / 10) * 100
        print(f"   {name:<25} aparece {count}/10 vezes ({percentage:.0f}%)")
    
    print("\n✅ RECOMENDAÇÃO:")
    print("-" * 80)
    best_config = df[df['num_neighborhoods'] == 3].iloc[0]
    print(f"Melhor configuração com 3 vizinhanças:")
    print(f"   {best_config['config']}")
    print(f"   f1: {best_config['avg_f1']:.2f}")
    print(f"   Vizinhanças: {best_config['neighborhoods']}")
    
    bottom_3 = sorted_neighborhoods[:3]
    print(f"\nVizinhanças com menor impacto (podem ser removidas):")
    for name, count in bottom_3:
        print(f"   - {name}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO!")
    print("="*80)


if __name__ == "__main__":
    test_configurations()
