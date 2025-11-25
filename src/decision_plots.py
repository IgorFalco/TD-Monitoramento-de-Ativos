"""
Visualizações para a análise de decisão multicritério.
Inclui plots da fronteira Pareto com solução escolhida e detalhes da solução final.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_pareto_with_chosen(enriched_solutions, chosen_idx, output_file=None):
    """
    Plota fronteira Pareto com destaque para a solução escolhida.
    
    Args:
        enriched_solutions: DataFrame com todas as soluções avaliadas
        chosen_idx: índice da solução escolhida
        output_file: caminho para salvar a figura (se None, usa diretório do script)
    """
    if output_file is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'result', 'pareto_decisao_final.png')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Todas as soluções da fronteira
    ax.scatter(enriched_solutions['f1_distancia'], 
              enriched_solutions['f2_equipes'],
              c='lightblue', s=150, alpha=0.6, 
              edgecolors='navy', linewidth=1.5,
              label='Soluções Pareto', zorder=2)
    
    # Conecta pontos da fronteira
    sorted_idx = enriched_solutions.sort_values('f1_distancia').index
    ax.plot(enriched_solutions.loc[sorted_idx, 'f1_distancia'],
           enriched_solutions.loc[sorted_idx, 'f2_equipes'],
           'b--', alpha=0.4, linewidth=2, zorder=1)
    
    # Destaque para a solução escolhida
    chosen = enriched_solutions.iloc[chosen_idx]
    ax.scatter(chosen['f1_distancia'], chosen['f2_equipes'],
              c='red', s=400, marker='*', 
              edgecolors='darkred', linewidth=2,
              label='Solução Escolhida', zorder=3)
    
    # Anotação com detalhes
    ax.annotate(
        f"Solução {chosen_idx}\n"
        f"f1={chosen['f1_distancia']:.1f} km\n"
        f"f2={int(chosen['f2_equipes'])} equipes",
        xy=(chosen['f1_distancia'], chosen['f2_equipes']),
        xytext=(20, 20), textcoords='offset points',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                       color='red', lw=2)
    )
    
    # Configurações do gráfico
    ax.set_xlabel('f1 - Distância Total (km)', fontsize=13, fontweight='bold')
    ax.set_ylabel('f2 - Número de Equipes', fontsize=13, fontweight='bold')
    ax.set_title('Fronteira de Pareto - Decisão Multicritério\n'
                'Avaliação por AHP e PROMETHEE', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # Anotações dos extremos
    min_f1_idx = enriched_solutions['f1_distancia'].idxmin()
    min_f2_idx = enriched_solutions['f2_equipes'].idxmin()
    
    ax.annotate('Menor distância', 
               xy=(enriched_solutions.loc[min_f1_idx, 'f1_distancia'],
                   enriched_solutions.loc[min_f1_idx, 'f2_equipes']),
               xytext=(-60, -30), textcoords='offset points',
               fontsize=9, style='italic', color='navy',
               arrowprops=dict(arrowstyle='->', color='navy', lw=1))
    
    ax.annotate('Menos equipes', 
               xy=(enriched_solutions.loc[min_f2_idx, 'f1_distancia'],
                   enriched_solutions.loc[min_f2_idx, 'f2_equipes']),
               xytext=(20, -30), textcoords='offset points',
               fontsize=9, style='italic', color='navy',
               arrowprops=dict(arrowstyle='->', color='navy', lw=1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico salvo: {output_file}")
    plt.close()


def plot_chosen_solution_details(chosen_solution, chosen_idx, 
                                 output_file=None):
    """
    Plota características detalhadas da solução escolhida.
    
    Args:
        chosen_solution: Series com dados da solução escolhida
        chosen_idx: índice da solução
        output_file: caminho para salvar a figura (se None, usa diretório do script)
    """
    if output_file is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'result', 'solucao_final_detalhes.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Análise Detalhada da Solução Escolhida (ID: {chosen_idx})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Objetivos Principais (f1 e f2)
    ax1 = axes[0, 0]
    objectives = ['Distância\nTotal (km)', 'Número de\nEquipes']
    values = [chosen_solution['f1_distancia'], chosen_solution['f2_equipes']]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = ax1.bar(objectives, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Valor', fontsize=11, fontweight='bold')
    ax1.set_title('Objetivos de Otimização', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}' if val > 10 else f'{int(val)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Critérios Adicionais (0-1 scale)
    ax2 = axes[0, 1]
    additional = ['Robustez', 'Confiabilidade', 'Flexibilidade']
    values2 = [chosen_solution['robustez'], 
               chosen_solution['confiabilidade'],
               chosen_solution['flexibilidade']]
    colors2 = ['#2ecc71', '#9b59b6', '#f39c12']
    
    bars2 = ax2.barh(additional, values2, color=colors2, alpha=0.7, 
                     edgecolor='black', linewidth=2)
    ax2.set_xlabel('Score (0-1)', fontsize=11, fontweight='bold')
    ax2.set_title('Critérios Adicionais (Maximizar)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Adiciona valores nas barras
    for bar, val in zip(bars2, values2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}',
                ha='left', va='center', fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 3. Risco Operacional (1-10 scale, menor é melhor)
    ax3 = axes[1, 0]
    risk_value = chosen_solution['risco_operacional']
    
    # Barra horizontal com gradiente de cor (verde a vermelho)
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0, 1, 10))
    for i in range(10):
        ax3.barh(0, 1, left=i, color=colors_gradient[i], edgecolor='black', linewidth=0.5)
    
    # Marcador da posição do risco
    ax3.plot([risk_value, risk_value], [-0.3, 0.3], 'k-', linewidth=4)
    ax3.scatter(risk_value, 0, s=300, c='black', marker='v', zorder=5)
    
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xlabel('Nível de Risco', fontsize=11, fontweight='bold')
    ax3.set_title(f'Risco Operacional: {risk_value:.2f}/10 (Minimizar)', 
                 fontsize=12, fontweight='bold')
    ax3.set_yticks([])
    ax3.text(risk_value, -0.45, f'{risk_value:.2f}', ha='center', 
            fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9))
    
    # Labels nos extremos
    ax3.text(0.5, 0.35, 'Baixo Risco', ha='center', fontsize=9, color='green', fontweight='bold')
    ax3.text(9.5, 0.35, 'Alto Risco', ha='center', fontsize=9, color='red', fontweight='bold')
    
    # 4. Radar Chart com todos os critérios normalizados
    ax4 = axes[1, 1]
    
    # Normaliza critérios para 0-1
    criteria_names = ['Dist.\n(inv)', 'Equipes\n(inv)', 'Robustez', 
                     'Confiab.', 'Risco\n(inv)', 'Flexib.']
    
    # Para minimização, inverte (1 - norm); para maximização, usa direto
    # Normaliza usando min-max simples para visualização
    norm_values = [
        1 - (chosen_solution['f1_distancia'] / 4000),  # Inverte distância (assume max ~4000)
        1 - (chosen_solution['f2_equipes'] / 8),  # Inverte equipes (max 8)
        chosen_solution['robustez'],
        chosen_solution['confiabilidade'],
        1 - (chosen_solution['risco_operacional'] / 10),  # Inverte risco
        chosen_solution['flexibilidade']
    ]
    norm_values = [max(0, min(1, v)) for v in norm_values]  # Clip 0-1
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(criteria_names), endpoint=False).tolist()
    norm_values += norm_values[:1]  # Fecha o polígono
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, norm_values, 'o-', linewidth=2, color='#3498db', markersize=8)
    ax4.fill(angles, norm_values, alpha=0.25, color='#3498db')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(criteria_names, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax4.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
    ax4.set_title('Perfil Normalizado\n(valores próximos de 1 = melhor)', 
                 fontsize=12, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_file}")
    plt.close()


def plot_methods_comparison(comparison_df, output_file=None):
    """
    Compara rankings de AHP vs PROMETHEE.
    
    Args:
        comparison_df: DataFrame com rankings dos dois métodos
        output_file: caminho para salvar (se None, usa diretório do script)
    """
    if output_file is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'result', 'comparacao_metodos.png')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: Rank AHP vs PROMETHEE
    ax1 = axes[0]
    ax1.scatter(comparison_df['Rank_AHP'], comparison_df['Rank_PROMETHEE'],
               s=100, alpha=0.6, c=comparison_df['f2 (equipes)'], 
               cmap='viridis', edgecolors='black', linewidth=1)
    
    # Linha de concordância perfeita
    max_rank = max(comparison_df['Rank_AHP'].max(), comparison_df['Rank_PROMETHEE'].max())
    ax1.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, linewidth=2, 
            label='Concordância Perfeita')
    
    ax1.set_xlabel('Posição AHP', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Posição PROMETHEE', fontsize=11, fontweight='bold')
    ax1.set_title('Comparação de Rankings\nAHP vs PROMETHEE', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Histograma de diferenças
    ax2 = axes[1]
    rank_diff = comparison_df['Rank_AHP'] - comparison_df['Rank_PROMETHEE']
    ax2.hist(rank_diff, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Sem diferença')
    ax2.set_xlabel('Diferença de Ranking (AHP - PROMETHEE)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequência', fontsize=11, fontweight='bold')
    ax2.set_title('Distribuição das Diferenças\nentre Métodos', 
                 fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_file}")
    plt.close()


def plot_all_visualizations(results_dict):
    """
    Gera todas as visualizações da análise de decisão.
    
    Args:
        results_dict: retorno da função run_multicriteria_decision
    """
    print("\n" + "="*60)
    print("📊 GERANDO VISUALIZAÇÕES")
    print("="*60)
    
    enriched = results_dict['enriched_solutions']
    chosen_idx = results_dict['chosen_idx']
    chosen_sol = results_dict['chosen_solution']
    comparison = results_dict['comparison']
    
    # 1. Fronteira com solução escolhida
    plot_pareto_with_chosen(enriched, chosen_idx)
    
    # 2. Detalhes da solução final
    plot_chosen_solution_details(chosen_sol, chosen_idx)
    
    # 3. Comparação dos métodos
    plot_methods_comparison(comparison)
    
    print("\n✅ Todas as visualizações foram geradas em result/")


if __name__ == "__main__":
    # Exemplo com dados simulados
    from decision_pipeline import run_multicriteria_decision
    
    results = run_multicriteria_decision(
        results_dir='result',
        method='soma_ponderada',
        max_solutions=20
    )
    
    plot_all_visualizations(results)
