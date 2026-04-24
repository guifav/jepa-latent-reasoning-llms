# Research plan: language JEPA for latent reasoning

## Título de trabalho
Language JEPA for Latent Reasoning: decoupling, latent geometry, and uncertainty

## Decisão de foco
Documento complementar de framing do artigo:
- `article_framing.md`

Este plano fixa o núcleo do projeto em quatro papers-base:
- `2509.14252` — **LLM-JEPA**
- `2512.19171` — **JEPA-Reasoner**
- `2511.08544` — **LeJEPA**
- `2603.20111` — **Var-JEPA**

Os outros dois papers lidos nesta rodada entram como extensão futura:
- `2603.22281` — ThinkJEPA, para guidance externo/teacher semântico
- `2602.01456` — Rectified LpJEPA, para latentes esparsos

## Tese do projeto
A hipótese central é que um modelo de linguagem pode raciocinar de forma mais robusta se:
1. o raciocínio principal ocorrer em espaço latente, não dentro do loop token-a-token;
2. a verbalização for separada do raciocínio por um módulo Talker/decoder;
3. a estabilidade do espaço latente for controlada por um princípio explícito, e não só por truques de treino;
4. a incerteza do estado latente for modelada explicitamente, em vez de ficar implícita no ruído autoregressivo.

## Pergunta principal
Sob mesmo orçamento de dados/compute, e mantendo o mesmo backbone principal (**`google/gemma-4-E2B`**) entre os comparativos centrais, um modelo JEPA desacoplado para linguagem supera:
- um Gemma-4-E2B LM padrão;
- um Gemma-4-E2B + JEPA acoplado estilo LLM-JEPA;
em robustez, degradação de longo horizonte, geometria latente e calibração?

## Perguntas secundárias
1. O ganho principal vem do **desacoplamento** ou da **regularização latente**?
2. Regularização estilo **LeJEPA** melhora estabilidade em linguagem?
3. Formulação **variacional** estilo Var-JEPA produz incerteza útil para abstention/risk-coverage?
4. Em quais tarefas o desacoplamento ajuda de fato, e em quais ele atrapalha?
5. Que tipo de par/multi-view em linguagem dá mais ganho para JEPA?

## O que NÃO será foco na fase 1
- multimodalidade pesada;
- teacher externo grande estilo ThinkJEPA;
- latentes esparsos estilo Rectified LpJEPA;
- scaling para regime foundation-model real;
- RLHF, tool use e retrieval avançado.

A fase 1 precisa responder se a direção funciona em pequena/média escala, não vencer SOTA industrial.

---

# 1. Linha de modelos

## Baseline 0 — LM padrão
### Objetivo
Estabelecer referência mínima.

### Arquitetura
- backbone principal: **`google/gemma-4-E2B`**;
- causal LM loss padrão.

### Função
Medir o que já se ganha sem JEPA.

---

## Baseline 1 — LLM-JEPA acoplado
### Inspiração
`2509.14252`.

### Objetivo
Verificar se JEPA já ajuda em linguagem sem mudar a arquitetura de geração.

### Arquitetura
- backbone principal **`google/gemma-4-E2B`**, igual ao Baseline 0;
- perda total = cross-entropy + termo JEPA entre duas views do mesmo conteúdo;
- atenção em blocos/packing para evitar vazamento entre views;
- predictor leve ou amarrado ao backbone, como no paper.

### Variáveis críticas
- peso `lambda_jepa`;
- profundidade do predictor (`k=0` ou pequeno);
- tipo de view par (paráfrase, pergunta↔solução, descrição↔formalização, contexto↔continuação).

### Critério de sucesso
Ganhar do LM padrão com custo extra aceitável e sem quebrar geração.

---

## Modelo 2 — JEPA-Reasoner desacoplado
### Inspiração
`2512.19171`.

### Objetivo
Testar se separar Reasoner e Talker já produz ganho claro usando o mesmo backbone base (**`google/gemma-4-E2B`**) dos baselines acoplados.

### Arquitetura mínima
- **Reasoner**: Transformer latente autoregressivo;
- **Target encoder**: encoder/embedding alvo com EMA ou variante controlada;
- **Talker**: decoder que verbaliza a trajetória latente pronta;
- **rollout latente**: o Reasoner gera uma cadeia latente antes da saída textual final.

### Decisões concretas para a fase 1
- começar com estados latentes determinísticos, não variacionais;
- usar normalização forte (RMS + L2 / hypersphere) inspirada no paper;
- treinar Talker separado inicialmente, para não misturar causas cedo demais;
- manter tamanhos pequenos o bastante para ablação rápida.

### Pergunta experimental
O desacoplamento por si só já melhora robustez e/ou tasks de raciocínio?

---

## Modelo 3A — Desacoplado + regularização latente estilo LeJEPA
### Inspiração
`2511.08544`.

### Objetivo
Trocar “anti-colapso por costume” por geometria latente explícita.

### Ideia
Adicionar ao espaço do Reasoner uma regularização distribucional que empurre embeddings agregados para comportamento aproximadamente gaussiano isotrópico.

### Implementação fase 1
Em vez de clonar LeJEPA inteiro de saída, usar uma versão progressiva:
1. medir isotropia e colapso sem regularização extra;
2. adicionar um termo leve de regularização agregada inspirado em LeJEPA;
3. só depois testar uma forma mais fiel do mecanismo por projeções/sketches se fizer diferença.

### Hipótese
Parte da instabilidade do raciocínio latente em linguagem vem de geometria ruim do espaço, não só do objetivo de previsão.

---

## Modelo 3B — Desacoplado + formulação variacional estilo Var-JEPA
### Inspiração
`2603.20111`.

### Objetivo
Dar semântica probabilística ao estado latente e medir incerteza útil.

### Ideia
Modelar o estado latente do contexto e/ou do alvo como distribuições, com prior condicional aprendido para o alvo latente.

### Implementação fase 1
- começar com uma versão pequena, com gaussian latent states;
- usar KL por amostra e prior condicional simples;
- medir risk-coverage, selective accuracy e sensibilidade a inputs ambíguos;
- não tentar reconstrução rica demais no início.

### Hipótese
Se o raciocínio latente for realmente multimodal ou ambíguo, uma formulação puramente determinística vai esconder isso; a variacional pode expor e calibrar melhor esse comportamento.

---

# 2. Estratégia de dados

## Regra geral
Começar com tarefas em que “duas views do mesmo conteúdo” são naturais e auditáveis.

## Fase 1A — Benchmarks reais pequenos/médios
### Prioridade inicial
1. **GSM8K** ou subset controlado dele
2. **RegexEval** como benchmark real secundário de linguagem→regex
3. **Spider subset** ou benchmark mais simples de formalização
4. **Paráfrases / reformulações** para pré-treino JEPA auxiliar

### Por que isso vem primeiro
Porque a pergunta científica principal é sobre valor em linguagem real. Se o método não mostrar sinal em benchmark real, eu não quero me enganar com uma vitória só em ambiente sintético.

## Fase 1B — Dados sintéticos controlados
### Objetivo
Usar tarefas sintéticas como bancada diagnóstica para robustez, horizon stress test e inspeção do latente.

### Conjunto sugerido
1. **Arithmetic trace tasks**
   - entrada: problema simples de aritmética/composição;
   - views possíveis: enunciado ↔ solução estruturada, enunciado ↔ passos latentes simplificados.
2. **Regex/text transformation tasks**
   - inspirado no LLM-JEPA;
   - descrição ↔ regex / transformação formal.
3. **CFG / Dyck / compositional grammar tasks**
   - inspirado no JEPA-Reasoner;
   - bom para perturbação controlada e long horizon.
4. **Symbolic program traces**
   - pequenas execuções determinísticas com estados intermediários claros.

### Papel do sintético
Não será o eixo principal da evidência. Vai servir para:
- debugging rápido;
- testar perturbação controlada;
- entender falha de arquitetura vs falha de dataset;
- inspecionar trajetória latente quando o benchmark real estiver ambíguo.

### Regra de seleção
Usar tarefas onde:
- há razoável noção de views pareadas;
- o acerto final é fácil de medir;
- há espaço para perturbação controlada.

## Estrutura de datasets por papel
- **LLM-JEPA lane**: pares naturais de views.
- **JEPA-Reasoner lane**: tarefas com passos latentes e horizon relevante.
- **LeJEPA lane**: qualquer tarefa onde possamos medir colapso/isotropia.
- **Var-JEPA lane**: tarefas com ambiguidade/uncertainty ou selective prediction.

---

# 3. Plano experimental

## Etapa 0 — Infra e reprodutibilidade mínima
### Meta
Ter pipeline único para treino, avaliação e logging.

### Entregáveis
- config única por experimento;
- runners reproduzíveis;
- logging de losses, métricas e diagnósticos latentes;
- salvar checkpoints e tabelas comparáveis.

### Métricas mínimas sempre registradas
- loss total;
- CE loss;
- latent/JEPA loss;
- norma média dos latentes;
- espectro/covariância agregada;
- taxa de colapso / rank efetivo;
- accuracy final.

## Etapa 1 — Baselines acoplados
### Experimentos
1. LM padrão em 1 tarefa sintética + 1 tarefa real.
2. LLM-JEPA acoplado nas mesmas tarefas.
3. Grid pequeno de `lambda_jepa` e profundidade do predictor.

### Saída esperada
Descobrir se JEPA em linguagem já rende antes do desacoplamento.

## Etapa 2 — Desacoplamento puro
### Experimentos
1. JEPA-Reasoner pequeno com Talker separado.
2. Comparação direta com Baseline 1 sob mesmo budget.
3. Teste com e sem corrupção de tokens de entrada.
4. Teste com rollout mais longo.

### Saída esperada
Medir o “valor puro” do desacoplamento.

## Etapa 3 — Regularização geométrica
### Experimentos
1. Modelo 2 sem reg geométrica adicional.
2. Modelo 2 + regularização inspirada em LeJEPA.
3. Comparação de isotropia, rank efetivo, estabilidade e accuracy.

### Saída esperada
Saber se a geometria explícita melhora linguagem ou só visão.

## Etapa 4 — Formulação variacional
### Experimentos
1. Versão determinística vs variacional do modelo desacoplado.
2. Ambiguity test / selective prediction.
3. Risk-coverage, AUROC de erro, calibration.

### Saída esperada
Saber se a variacional traz valor real ou só complexidade.

---

# 4. Métricas e diagnósticos

## Métricas de tarefa
- exact match;
- accuracy final;
- execution accuracy quando aplicável;
- pass@1 em tasks programáticas simples.

## Métricas de robustez
- queda de accuracy sob corrupção de tokens;
- queda sob ruído latente;
- erro por comprimento / por horizonte;
- sensibilidade a parafraseamento e reformulação.

## Métricas de geometria latente
- rank efetivo;
- espectro de covariância;
- isotropia / desvio da identidade;
- estabilidade de norma;
- linearidade entre views;
- clustering por semântica da tarefa.

## Métricas de incerteza
- NLL/ELBO quando aplicável;
- ECE / calibration proxy;
- risk-coverage;
- selective accuracy;
- AUROC para detectar erro/resposta incerta.

## Sinais de falha a monitorar
- colapso de embeddings;
- Talker compensando demais o Reasoner;
- latente suave porém logicamente errado;
- ganho em CE sem ganho em estrutura latente;
- ganho sintético que não transfere para benchmark real.

---

# 5. Matriz de ablações

## Ablations obrigatórias
1. **LM padrão vs LLM-JEPA acoplado**
2. **LLM-JEPA vs JEPA-Reasoner desacoplado**
3. **Talker separado vs decoder compartilhado**
4. **normalização fraca vs normalização forte**
5. **sem reg geométrica vs com reg geométrica**
6. **determinístico vs variacional**
7. **dados naturais pareados vs dados sintéticos controlados**
8. **treino conjunto vs treino em estágios**
9. **sem perturbação vs perturbação de tokens / ruído latente**
10. **mesmo budget de compute** em todas as comparações centrais

## Ablations úteis mas não prioritárias
- profundidade do predictor;
- tamanho do Talker;
- EMA vs alternativas;
- tipo de loss angular/cosseno;
- tipo de prior condicional;
- usar passos latentes explícitos ou não.

---

# 6. Critérios de sucesso

## Sucesso mínimo
Pelo menos uma destas afirmações precisa ficar verdadeira com evidência boa:
1. O modelo desacoplado supera o acoplado com mesmo budget em pelo menos uma tarefa real e uma sintética.
2. A regularização geométrica melhora estabilidade sem destruir performance.
3. A variante variacional produz sinal de incerteza útil para seleção/abstenção.

## Sucesso forte
- o desacoplamento melhora robustez de forma consistente;
- a regularização explícita melhora a qualidade do espaço latente;
- a variacional melhora calibration sem perda severa de acurácia;
- as conclusões aparecem em mais de uma família de tarefa.

## Falha honesta útil
Mesmo se o desacoplamento não ganhar, o projeto ainda vale se mostrar claramente:
- quando JEPA em linguagem ajuda e quando não ajuda;
- se LeJEPA-style regularization transfere ou não para linguagem;
- se uncertainty modeling é real ou cosmético nesse contexto.

---

# 7. Riscos principais

## Risco 1 — faltam views boas em linguagem
### Mitigação
Começar em datasets onde as views são nativas, não inventadas artificialmente.

## Risco 2 — o Talker resolve demais e mascara o ganho do Reasoner
### Mitigação
Ablar o Talker agressivamente e medir dependência real do latente.

## Risco 3 — regularização geométrica de visão não transfere para linguagem
### Mitigação
Começar com reg leve e medir geometria antes de assumir benefício.

## Risco 4 — modelo variacional adiciona custo sem valor
### Mitigação
Entrar só depois de baseline desacoplado estável.

## Risco 5 — ganhos só aparecem em tarefas sintéticas
### Mitigação
Desde cedo manter pelo menos um benchmark real no loop.

---

# 8. Ordem de execução recomendada

## Sprint 1 — Coupled lane
- implementar **Gemma-4-E2B LM padrão**;
- implementar **Gemma-4-E2B + LLM-JEPA acoplado**;
- usar **GSM8K** como benchmark real principal e **RegexEval** como benchmark real secundário estruturado;
- rodar primeiro comparativo limpo;
- deixar 1 tarefa sintética só como bancada diagnóstica, não como eixo principal.

## Sprint 2 — Decoupled lane
- implementar **Gemma-4-E2B + Reasoner/Talker**;
- repetir comparação com mesmo budget;
- rodar perturbação de tokens e rollout longo.

## Sprint 3 — Geometry lane
- adicionar reg inspirada em LeJEPA;
- medir isotropia, rank, estabilidade e accuracy.

## Sprint 4 — Uncertainty lane
- introduzir latent states variacionais;
- medir risk-coverage e selective accuracy.

---

# 9. Entregáveis esperados

## Artefatos científicos
- tabela comparativa dos quatro modelos centrais;
- suite pequena de benchmarks/diagnósticos para language JEPA;
- análise de geometria latente;
- análise de robustez;
- análise de calibração/incerteza.

## Artefatos de projeto
- configs reproduzíveis;
- scripts de treino/avaliação;
- notas de leitura dos papers-base;
- relatório parcial por sprint.

---

# 10. Próximo passo concreto

## Passo escolhido agora
Começar pelo núcleo que eu recomendei:
1. **fixar o par de tarefas** da fase 1;
2. **desenhar a especificação mínima dos modelos acoplado e desacoplado**;
3. **definir a primeira matriz de ablações obrigatórias**.

## Decisão operacional
A prioridade imediata não é ler mais 20 papers. É fechar:
- benchmark sintético inicial,
- benchmark real inicial,
- arquitetura mínima do LLM-JEPA baseline,
- arquitetura mínima do JEPA-Reasoner baseline,
- logging dos diagnósticos latentes.

Se essa etapa ficar boa, o restante do projeto deixa de ser vago e vira execução.
