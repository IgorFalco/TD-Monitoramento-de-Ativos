from generator import read_csv_data, generate_initial_solution
from plot import plot_solution
from constraints import constraints
from vns import create_vns
import numpy as np

def main():
    """Função principal do sistema de monitoramento de ativos."""
    
    print("=" * 80)
    print("🏢 SISTEMA DE MONITORAMENTO DE ATIVOS")
    print("=" * 80)
    print("Carregando dados...")
    
    try:
        dist_bases_assets = read_csv_data()
        num_assets, num_bases = dist_bases_assets.shape
        
        print(f"✅ Dados carregados!")
        print(f"📊 Ativos: {num_assets}")
        print(f"🏭 Bases: {num_bases}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    print("\n" + "-" * 50)
    print("🔄 GERANDO SOLUÇÃO INICIAL...")
    print("-" * 50)
    
    try:
        solution = generate_initial_solution(num_assets, num_bases, dist_bases_assets)
        
        if solution is None:
            print("❌ Não foi possível gerar solução válida!")
            return
            
        print("✅ Solução inicial gerada!")
        
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return
    
    # Verifica constraints
    print("\nValidando constraints...")
    constraints_ok = True
    
    for i, constraint in enumerate(constraints):
        try:
            result = constraint(solution)
            constraint_name = constraint.__name__
            status = "✓" if result else "✗"
            print(f"{status} {constraint_name}: {'OK' if result else 'FALHOU'}")
            if not result:
                constraints_ok = False
        except Exception as e:
            print(f"✗ Erro ao verificar constraint {i}: {e}")
            constraints_ok = False
    
    print(f"\nStatus: {'✓ TODAS OK' if constraints_ok else '✗ FALHAS DETECTADAS'}")
    
    # Informações da solução inicial
    print(f"\n📈 SOLUÇÃO INICIAL:")
    print(f"  f1 (Distância): {solution['f1']:.2f}")
    print(f"  f2 (Equipes): {int(solution['f2'])}")
    
    # Detalhes das equipes
    num_equipes = solution['y'].shape[1]
    equipes_ativas = 0
    
    print(f"\n📋 DISTRIBUIÇÃO:")
    for k in range(num_equipes):
        if np.sum(solution['y'][:, k]) > 0:
            equipes_ativas += 1
            base_equipe = np.where(solution['y'][:, k] == 1)[0][0]
            ativos_equipe = int(np.sum(solution['h'][:, k]))
            print(f"  Equipe {k+1}: Base {base_equipe+1} → {ativos_equipe} ativos")
    
    print(f"\n📊 Equipes ativas: {equipes_ativas}")
    
    if not constraints_ok:
        print("⚠️  Constraints violadas. VNS não será executado.")
        return
    
    # OTIMIZAÇÃO VNS
    print("\n" + "=" * 80)
    print("🚀 INICIANDO OTIMIZAÇÃO VNS")
    print("=" * 80)
    
    try:
        # Cria o VNS
        vns = create_vns(dist_bases_assets)
        
        executar_vns = input("\nExecutar otimização VNS? (s/n): ").strip().lower()
        
        if executar_vns in ['s', 'sim', 'y', 'yes']:
            
            print("\nTipo de otimização:")
            print("1. f1 (Distância)")
            print("2. f2 (Número de Equipes)")
            
            opcao = input("Opção (1/2): ").strip()
            
            if opcao == '1':
                print("\n🎯 Otimizando f1...")
                solution_otimizada = vns.otimizar(solution, objetivo='f1', max_iter=50, max_time=180)
                
            elif opcao == '2':
                print("\n🎯 Otimizando f2...")
                solution_otimizada = vns.otimizar(solution, objetivo='f2', max_iter=50, max_time=180)

            else:
                print("Opção inválida.")
                solution_otimizada = solution
            
            # Comparação de resultados
            print("\n" + "=" * 50)
            print("📊 COMPARAÇÃO")
            print("=" * 50)
            
            print(f"INICIAL: f1={solution['f1']:.2f}, f2={int(solution['f2'])}")
            print(f"FINAL:   f1={solution_otimizada['f1']:.2f}, f2={int(solution_otimizada['f2'])}")
            
            # Melhorias
            melhoria_f1 = ((solution['f1'] - solution_otimizada['f1']) / solution['f1']) * 100
            melhoria_f2 = solution['f2'] - solution_otimizada['f2']
            
            print(f"\nMELHORIA:")
            print(f"  f1: {melhoria_f1:.2f}%")
            print(f"  f2: {int(melhoria_f2)} equipes")
            
            vns.print_history()
            solution = solution_otimizada
            
        else:
            print("VNS não executado.")
            
    except Exception as e:
        print(f"❌ Erro durante otimização VNS: {e}")
        print("Usando solução inicial para visualização.")
    
    # Visualização
    try:
        print("\n" + "-" * 50)
        resposta = input("Visualizar solução? (s/n): ").strip().lower()
        
        if resposta in ['s', 'sim', 'y', 'yes']:
            print("📈 Gerando visualização...")
            
            try:
                plot_solution(solution)
            except Exception as e:
                print(f"Erro no plot: {e}")
        else:
            print("Visualização dispensada.")
            
    except KeyboardInterrupt:
        print("\n👋 Interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro: {e}")
    
    print("\n" + "=" * 50)
    print("✅ FINALIZADO")
    print("=" * 50)

if __name__ == "__main__":
    main()
