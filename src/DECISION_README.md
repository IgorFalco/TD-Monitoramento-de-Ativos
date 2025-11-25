# Análise de Decisão Multicritério

## Visão Geral

Este módulo implementa a etapa de decisão multicritério do projeto, utilizando os métodos **AHP (Analytic Hierarchy Process)** e **PROMETHEE II** para escolher a melhor solução da fronteira de Pareto.

## Estrutura de Arquivos

```
src/
├── multi_criteria_decision.py   # Classe AHPAnalysis (já existente)
├── promethee.py                  # Classe PROMETHEEAnalysis (novo)
├── decision_pipeline.py          # Pipeline completo de decisão (novo)
├── decision_plots.py             # Visualizações (novo)
└── run_decision_analysis.py      # Script principal (novo)
```

## Como Executar

### 1. Pré-requisito
Certifique-se de ter executado a otimização multiobjetivo (modo 2 do `main.py`) para gerar os CSVs das fronteiras Pareto em `result/`.

### 2. Executar Análise de Decisão
```bash
python src/run_decision_analysis.py
```

## Critérios de Decisão

### Critérios Originais (da otimização)
1. **f1 - Distância Total (km)**: Minimizar
2. **f2 - Número de Equipes**: Minimizar

### Critérios Adicionais (gerados)
3. **Robustez** (0-1): Maximizar
   - Estabilidade da solução ante variações nos parâmetros
   - Soluções intermediárias tendem a ser mais robustas

4. **Confiabilidade** (0-1): Maximizar
   - Probabilidade de sucesso na operação
   - Favorece mais equipes e menores distâncias

5. **Risco Operacional** (1-10): Minimizar
   - Pontuação de risco de falhas
   - Aumenta com distâncias e diminui com mais equipes

6. **Flexibilidade** (0-1): Maximizar
   - Capacidade de adaptação a novos cenários
   - Proporcional ao número de equipes

**Nota**: Todos os critérios são conflitantes entre si!

## Métodos Aplicados

### AHP (Analytic Hierarchy Process)
- Matriz de comparação par-a-par (escala de Saaty 1-9)
- Cálculo de pesos via autovetor principal
- Verificação de consistência (CR < 0.10)
- Normalização dos critérios e soma ponderada

**Priorização**: Confiabilidade > Robustez > f2 > Risco > Flexibilidade > f1

### PROMETHEE II
- Funções de preferência por critério (usual, linear, level, gaussian)
- Comparações par-a-par com limiares de indiferença (q) e preferência (p)
- Fluxos de sobreclassificação (Φ+, Φ-, Φnet)
- Ranking completo baseado no fluxo líquido

**Parâmetros**:
- f1: Linear (q=50km, p=300km)
- f2: Usual (preferência binária)
- Robustez/Confiab./Flexib.: Linear (q=0.02, p=0.10)
- Risco: Linear (q=0.5, p=2.0)

## Resolução de Incomparabilidades

**Critério de Desempate**: Quando AHP e PROMETHEE divergem, escolhe-se a solução com menor ranking médio entre os dois métodos.

## Saídas Geradas

### CSVs
- `solucoes_enriquecidas.csv`: Todas as soluções com critérios adicionais
- `ranking_ahp.csv`: Ranking completo do AHP
- `ranking_promethee.csv`: Ranking completo do PROMETHEE
- `comparacao_ahp_promethee.csv`: Comparação lado a lado
- `solucao_escolhida.txt`: Relatório detalhado da solução final

### Visualizações
- `pareto_decisao_final.png`: Fronteira Pareto com solução escolhida destacada
- `solucao_final_detalhes.png`: 4 gráficos detalhando a solução escolhida
- `comparacao_metodos.png`: Comparação dos rankings AHP vs PROMETHEE

## Exemplo de Uso Programático

```python
from decision_pipeline import run_multicriteria_decision
from decision_plots import plot_all_visualizations

# Executa análise
results = run_multicriteria_decision(
    results_dir='result',
    method='soma_ponderada',
    max_solutions=20
)

# Acessa resultados
chosen_idx = results['chosen_idx']
chosen_solution = results['chosen_solution']
ahp_ranking = results['ahp_result']
promethee_ranking = results['promethee_result']

# Gera visualizações
plot_all_visualizations(results)
```

## Configurações Personalizáveis

### Matriz AHP
Edite `decision_pipeline.py`, função `run_multicriteria_decision()`, variável `ahp_matrix`.

### Funções de Preferência PROMETHEE
Edite `decision_pipeline.py`, função `run_multicriteria_decision()`, variável `promethee_config`.

### Pesos dos Critérios
Os pesos são calculados automaticamente pelo AHP baseado na matriz de comparação.

## Validações

- **Consistência AHP**: CR < 0.10 (caso contrário, aviso é exibido)
- **Não-dominância**: Apenas soluções não-dominadas são consideradas
- **Representatividade**: Até 20 soluções mais bem distribuídas

## Referências

- Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
- Brans, J. P., & Vincke, P. (1985). PROMETHEE: A new family of outranking methods. IFORS.
- Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms. Wiley.
