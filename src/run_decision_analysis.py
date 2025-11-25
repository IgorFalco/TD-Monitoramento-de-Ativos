"""
Script principal para executar a análise de decisão multicritério completa.
Integra consolidação de fronteiras, AHP, PROMETHEE e visualizações.
"""

from decision_pipeline import run_multicriteria_decision
from decision_plots import plot_all_visualizations
import pandas as pd


def main():
    """Executa pipeline completo de decisão multicritério."""
    
    print("\n" + "="*80)
    print("🎯 SISTEMA DE DECISÃO MULTICRITÉRIO")
    print("   Análise de Fronteiras Pareto via AHP e PROMETHEE")
    print("="*80)
    
    # Configurações
    # Caminho relativo ao diretório src (onde o script está)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(script_dir, 'result')
    METHOD = 'soma_ponderada'  # ou 'epsilon_restrito'
    MAX_SOLUTIONS = 20
    
    try:
        # 1. Executa análise de decisão
        print("\n🚀 Iniciando análise de decisão multicritério...")
        results = run_multicriteria_decision(
            results_dir=RESULTS_DIR,
            method=METHOD,
            max_solutions=MAX_SOLUTIONS
        )
        
        # 2. Gera visualizações
        print("\n📊 Gerando visualizações...")
        plot_all_visualizations(results)
        
        # 3. Salva relatório em CSV
        print("\n💾 Salvando relatórios...")
        
        # Soluções enriquecidas com critérios
        enriched_file = f'{RESULTS_DIR}/solucoes_enriquecidas.csv'
        results['enriched_solutions'].to_csv(enriched_file, index=False, sep=';', decimal=',')
        print(f"   ✓ {enriched_file}")
        
        # Comparação dos métodos
        comparison_file = f'{RESULTS_DIR}/comparacao_ahp_promethee.csv'
        results['comparison'].to_csv(comparison_file, index=False, sep=';', decimal=',')
        print(f"   ✓ {comparison_file}")
        
        # Rankings detalhados
        ahp_ranking_file = f'{RESULTS_DIR}/ranking_ahp.csv'
        results['ahp_result'].to_csv(ahp_ranking_file, index=False, sep=';', decimal=',')
        print(f"   ✓ {ahp_ranking_file}")
        
        promethee_ranking_file = f'{RESULTS_DIR}/ranking_promethee.csv'
        results['promethee_result'].to_csv(promethee_ranking_file, index=False, sep=';', decimal=',')
        print(f"   ✓ {promethee_ranking_file}")
        
        # Solução escolhida
        chosen_file = f'{RESULTS_DIR}/solucao_escolhida.txt'
        with open(chosen_file, 'w', encoding='utf-8') as f:
            chosen = results['chosen_solution']
            idx = results['chosen_idx']
            
            f.write("="*60 + "\n")
            f.write("SOLUÇÃO FINAL ESCOLHIDA - DECISÃO MULTICRITÉRIO\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Índice da Solução: {idx}\n\n")
            
            f.write("--- OBJETIVOS DE OTIMIZAÇÃO ---\n")
            f.write(f"f1 - Distância Total: {chosen['f1_distancia']:.2f} km\n")
            f.write(f"f2 - Número de Equipes: {int(chosen['f2_equipes'])}\n")
            
            if 'peso_w1' in chosen:
                f.write(f"\nPesos do Método Soma Ponderada:\n")
                f.write(f"  w1 (distância): {chosen['peso_w1']:.4f}\n")
                f.write(f"  w2 (equipes): {chosen['peso_w2']:.4f}\n")
            
            f.write("\n--- CRITÉRIOS ADICIONAIS ---\n")
            f.write(f"Robustez: {chosen['robustez']:.4f} (0-1, maior é melhor)\n")
            f.write(f"Confiabilidade: {chosen['confiabilidade']:.4f} (0-1, maior é melhor)\n")
            f.write(f"Risco Operacional: {chosen['risco_operacional']:.2f} (1-10, menor é melhor)\n")
            f.write(f"Flexibilidade: {chosen['flexibilidade']:.4f} (0-1, maior é melhor)\n")
            
            f.write("\n--- JUSTIFICATIVA ---\n")
            f.write("Esta solução foi escolhida após análise por dois métodos de decisão\n")
            f.write("multicritério (AHP e PROMETHEE II), considerando:\n")
            f.write("  - Minimização de distância e número de equipes\n")
            f.write("  - Maximização de robustez, confiabilidade e flexibilidade\n")
            f.write("  - Minimização de risco operacional\n")
            f.write("\nA priorização dos critérios favoreceu confiabilidade e robustez,\n")
            f.write("essenciais para operação em cenários com incertezas.\n")
            
            f.write("\n--- RANKINGS ---\n")
            ahp_pos = results['ahp_result'].loc[idx, 'Posição'] if idx in results['ahp_result'].index else 'N/A'
            prom_pos = results['promethee_result'].loc[idx, 'Posição'] if idx in results['promethee_result'].index else 'N/A'
            f.write(f"Posição no ranking AHP: {ahp_pos}\n")
            f.write(f"Posição no ranking PROMETHEE: {prom_pos}\n")
        
        print(f"   ✓ {chosen_file}")
        
        # 4. Resumo final
        print("\n" + "="*80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*80)
        print(f"\n📂 Arquivos gerados em: {RESULTS_DIR}/")
        print("\nArquivos principais:")
        print(f"  • pareto_decisao_final.png - Fronteira com solução escolhida")
        print(f"  • solucao_final_detalhes.png - Características da solução")
        print(f"  • comparacao_metodos.png - AHP vs PROMETHEE")
        print(f"  • solucao_escolhida.txt - Relatório da decisão")
        print(f"  • ranking_ahp.csv e ranking_promethee.csv - Rankings completos")
        
        print("\n🏆 Solução Final:")
        chosen = results['chosen_solution']
        print(f"  f1 (Distância): {chosen['f1_distancia']:.2f} km")
        print(f"  f2 (Equipes): {int(chosen['f2_equipes'])}")
        print(f"  Robustez: {chosen['robustez']:.3f}")
        print(f"  Confiabilidade: {chosen['confiabilidade']:.3f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Dica: Execute primeiro a geração das fronteiras Pareto (main.py modo 2)")
        print("   para gerar os arquivos CSV necessários em result/")
        
    except Exception as e:
        print(f"\n❌ Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()
