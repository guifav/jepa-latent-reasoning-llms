# Phase 1 spec: language JEPA core

## Objetivo da fase 1
Transformar a direção de pesquisa em um primeiro experimento executável, comparável e metodologicamente defensável.

## Documentos de governança desta fase
- `article_framing.md`
- `backbone_selection.md`
- `gsm8k_phase1_protocol.md`
- `nl2regex_phase1_protocol.md`
- `methodology_protocol.md`
- `benchmark_suite.md`
- `mcq_support_benchmarks_protocol.md`

## Decisão de benchmark da fase 1

A fase 1 não vai mais depender de **GSM8K sozinho**.

### Camada paper-core
1. **GSM8K**
   - principal benchmark real de raciocínio em linguagem;
   - continua sendo a prova central da tese de raciocínio latente.
2. **RegexEval (`s2e-lab/RegexEval`)**
   - benchmark estruturado de transferência linguagem → especificação formal;
   - continua sendo a prova de alinhamento semântico fora da aritmética.
3. **ARC-Challenge**
   - benchmark adicional de raciocínio múltipla-escolha;
   - acrescenta pressão de raciocínio sem depender só de matemática escolar.
4. **HellaSwag**
   - benchmark adicional de inferência plausível / continuação;
   - acrescenta commonsense e coerência inferencial.

### Camada de suporte
5. **MMLU**
   - benchmark de cobertura ampla;
   - entra como suporte de generalidade, não como headline principal.

### Benchmark adiado
6. **HumanEval**
   - fica para fase posterior, depois que o stack principal de linguagem estiver estável.

### Benchmarks não priorizados nesta fase
- **TruthfulQA**: relevante para factualidade/alinhamento, mas não para a tese central.
- **BIG-bench**: heterogêneo demais para uma claim limpa de fase 1.

### Benchmark sintético de apoio
**Symbolic arithmetic traces** continuam apenas como bancada diagnóstica, não como headline experimental.

## Comparação central da fase 1
Backbone compartilhado da fase 1:
- **checkpoint principal: `google/gemma-4-E2B`**;
- usar a variante **base, não instruction-tuned**, para reduzir confounds de alinhamento conversacional e manter a intervenção arquitetural mais limpa;
- manter o mesmo checkpoint em todos os modelos centrais.

Checkpoint confirmatório posterior, só se a fase 1 mostrar sinal claro:
- **`google/gemma-4-E4B`** como réplica de escala moderadamente maior.

Dois modelos sob mesmo budget:
1. **Coupled language JEPA** (baseline estilo LLM-JEPA)
2. **Decoupled JEPA reasoner** (baseline estilo JEPA-Reasoner)

O comparativo mínimo da fase 1 fica assim:
1. **Gemma-4-E2B LM puro**
2. **Gemma-4-E2B + LLM-JEPA acoplado**
3. **Gemma-4-E2B + JEPA-Reasoner desacoplado**

Se o coupled não ganhar do LM puro, ainda assim ele continua útil como ponto de comparação arquitetural.

---

# Modelo A — Coupled language JEPA baseline

## Backbone
- backbone base: **`google/gemma-4-E2B`**;
- mesmo backbone do LM puro para comparação justa;
- mesma tokenização, mesma largura base e mesmo orçamento de treino das comparações centrais.

## Entradas
Duas views do mesmo item, empacotadas em bloco.

## Loss
`L = L_ce + lambda_jepa * L_latent`

Onde:
- `L_ce`: next-token prediction normal;
- `L_latent`: soma controlada de distância cosseno e InfoNCE com negativos in-batch entre embedding previsto da view alvo e embeddings reais das views alvo.

## Primeira configuração sugerida
- predictor raso;
- `k=0` ou `k=1` primeiro;
- `lambda_jepa` pequeno/moderado em grid curto;
- `contrastive_temperature=0.07` como default inicial;
- medir margem positivo-vs-negativo para verificar se o sinal contrastivo não colapsou.

## O que medir
- ganho absoluto sobre LM puro;
- estabilidade do treino;
- geometria entre views;
- custo por step.

---

# Modelo B — Decoupled JEPA-Reasoner baseline

## Backbone base
- inicializar a partir do mesmo checkpoint **`google/gemma-4-E2B`** usado no LM puro / coupled;
- preservar a maior simetria possível entre embeddings e capacidade total para que a comparação permaneça limpa.


## Blocos
1. **Reasoner**
   - recebe embedding da entrada/contexto;
   - produz cadeia latente autoregressiva.
2. **Target encoder**
   - gera alvos latentes do próximo passo.
3. **Talker**
   - recebe trajetória latente pronta;
   - verbaliza resposta/solução.

## Losses
- `L_reasoner`: previsão latente do próximo passo;
- `L_talker`: cross-entropy para reconstrução/verbalização;
- treino em duas etapas mais alinhamento opcional:
  1. Reasoner;
  2. Talker congelando o Reasoner;
  3. stage 3 opcional de alinhamento Reasoner→Talker com LR baixo.

## Decisões iniciais
- usar versão determinística primeiro;
- usar normalização forte no latente;
- usar rollout curto no começo para manter controle;
- não fazer full-backbone joint training logo no primeiro experimento; o stage 3 inicial deve ser curto e conservador.

## O que medir
- accuracy final;
- robustez a corrupção de tokens;
- degradação por horizonte;
- dependência real do Talker em relação ao latente.

---

# Primeira matriz de ablação

## Bloco 1 — comparação base
1. LM puro em GSM8K subset
2. Coupled JEPA em GSM8K subset
3. Decoupled JEPA em GSM8K subset
4. Repetição do melhor par em RegexEval

## Bloco 2 — sensibilidade do coupled
4. Coupled com `lambda_jepa` baixo
5. Coupled com `lambda_jepa` médio
6. Coupled com predictor mínimo vs pequeno

## Bloco 3 — sensibilidade do decoupled
7. Decoupled com Talker pequeno
8. Decoupled com Talker mais forte
9. Decoupled com e sem corrupção de tokens de entrada
10. Decoupled com horizonte curto vs longo

## Bloco 4 — diagnósticos latentes
11. isotropia agregada
12. rank efetivo
13. norma média
14. separação entre classes/semânticas
15. linearidade entre views quando aplicável

---

# Critério para avançar à fase 2

Avançar com regularização estilo LeJEPA e depois Var-JEPA se pelo menos um destes ocorrer em benchmark real:
- o decoupled ganha do coupled em robustez ou horizon;
- o coupled ganha do LM puro com clareza;
- a geometria latente mostra estrutura útil mesmo quando o ganho bruto é modesto.

O benchmark sintético entra como diagnóstico complementar, não como critério principal de avanço.

Se nada disso ocorrer, a próxima etapa vira diagnóstico de dataset/view antes de complicar a arquitetura.

---

# Views e setup inicial por benchmark

## GSM8K subset controlado
### Formato base do item
- `question`
- `solution_rationale`
- `final_answer`

### Views iniciais
- **View A**: `question`
- **View B**: `solution_rationale`
- **View C**: `final_answer`

### Uso por modelo
- **LM puro**: `question -> final_answer` e, em corrida separada opcional, `question -> solution_rationale -> final_answer`
- **Coupled JEPA**: alinhar principalmente `question ↔ solution_rationale`; usar `final_answer` como alvo de geração
- **Decoupled JEPA**: o Reasoner aprende a cadeia latente condicionada na `question`; o Talker verbaliza `solution_rationale` e/ou `final_answer`

### Subset inicial
- usar subset controlado e balanceado por comprimento da solução;
- registrar buckets curtos, médios e longos para medir degradação por horizonte.
- subset já materializado em `/root/workspace/jepa/data/gsm8k/phase1/` com manifesto em `/root/workspace/jepa/data/gsm8k/phase1/manifest.json`.

## Benchmark real secundário
### Escolha atual
- **RegexEval** como segunda tarefa real da fase 1
- split local já materializado em `/root/workspace/jepa/data/regexeval/phase1/`
- manifesto congelado em `/root/workspace/jepa/data/regexeval/phase1/manifest.json`

### Views iniciais
- **View A**: `raw_prompt`
- **View B**: `refined_prompt`
- **View C**: `expression`
- **View D**: `matches/non_matches`

### Papel no projeto
- validar a hipótese de alignment multi-view do LLM-JEPA em um caso mais limpo que GSM8K;
- checar se o ganho do JEPA aparece com mais nitidez quando as views são mais bem definidas;
- medir sucesso semântico, não só surface-form exact match.

# Grade mínima de hiperparâmetros compartilhados

## Regras gerais
- mesmo orçamento aproximado de passos para LM puro, coupled e decoupled;
- mesmo tokenizer/backbone Gemma;
- mesma política de batch/otimização, salvo necessidade arquitetural explícita.

## Grid inicial do coupled
- `lambda_jepa`: baixo / médio
- predictor depth: mínimo / pequeno
- loss latente: cosseno primeiro

## Grid inicial do decoupled
- rollout latente: curto / médio
- Talker size: pequeno / médio
- normalização: forte por padrão

## Logs obrigatórios
- accuracy final
- accuracy por bucket de comprimento
- robustez a perturbação leve da pergunta/descrição
- latent loss
- norma média do latente
- rank efetivo / isotropia agregada
- custo por step

# Próximo passo imediato após este spec
1. materializar o subset inicial de GSM8K com buckets por comprimento;
2. escrever a spec formal do protocolo de split, tuning e avaliação de GSM8K;
3. materializar o split congelado de RegexEval e o avaliador semântico;
4. transformar esta spec em configs de treino comparáveis;
5. escrever o protocolo metodológico para reprodutibilidade, significância estatística e reporting;
6. escrever depois o spec do benchmark sintético apenas como bancada diagnóstica.

Status atual desses itens:
- RegexEval já materializado em `/root/workspace/jepa/data/regexeval/phase1/`;
- avaliador semântico já implementado em `/root/workspace/jepa/scripts/eval_regex_semantics.py`;
- corpora por view já materializados em:
  - `/root/workspace/jepa/data/gsm8k/phase1_views/`
  - `/root/workspace/jepa/data/regexeval/phase1_views/`
- configs executáveis de fase 1 já preparados em `/root/workspace/jepa/configs/phase1/runs/`;
- runner mínimo de validação já implementado em `/root/workspace/jepa/scripts/phase1_runner.py`.
