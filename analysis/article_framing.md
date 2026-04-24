# Article framing for phase 1

## Scientific core
This project is not primarily about improving generic text generation.
It is about testing whether **JEPA-style latent reasoning architectures improve real language reasoning under controlled budget**.

## Central scientific hypothesis
A language model will reason more robustly if:
1. the main reasoning trajectory is formed in latent space rather than through token-by-token verbalization;
2. generation is separated from reasoning through a Talker/decoder;
3. the latent space is explicitly regularized instead of relying only on standard LM training dynamics.

## Primary real use case
**Grade-school mathematical reasoning in natural language**

### Operational benchmark
- **GSM8K**

### Why this is the main use case
Because it is the cleanest first test of the actual thesis:
- the input is natural language;
- the task requires multi-step reasoning;
- the dataset naturally exposes three views:
  - question
  - rationale/solution
  - final answer
- the evaluation is simple enough to keep the paper disciplined.

### What the model is supposed to do here
Given a question in natural language:
- the LM baseline predicts the answer directly;
- the coupled JEPA baseline aligns question and rationale while still generating normally;
- the decoupled JEPA model forms a latent reasoning trajectory first and then verbalizes.

### The paper question this use case answers
Does latent-space JEPA reasoning improve real reasoning quality, robustness, and long-horizon behavior compared with a standard LM and a coupled JEPA baseline?

## Secondary real use case
**Natural language to formal specification**

### Operational benchmark
- **RegexEval**

### Why this is secondary, not primary
Regex generation is not the main scientific target of the project. It is valuable because it isolates a different strength of JEPA:
- alignment between natural-language intent and structured symbolic output;
- multi-view correspondence is cleaner than in GSM8K;
- semantic evaluation is possible via match / non-match examples.

### What the model is supposed to do here
Given a natural-language requirement:
- predict a regex that satisfies the semantics of the description;
- for coupled JEPA, align prompt and formal output;
- for decoupled JEPA, test whether latent reasoning supports symbolic construction.

### The paper question this use case answers
Does the JEPA effect transfer beyond arithmetic reasoning into a cleaner language-to-formal alignment setting?

## Additional paper-core benchmarks
These do not replace the two use cases above, but they prevent the paper from collapsing into a single-benchmark story.

### ARC-Challenge
- role: additional reasoning benchmark
- purpose: test whether any gain survives in non-arithmetic multiple-choice reasoning

### HellaSwag
- role: commonsense / continuation benchmark
- purpose: test whether the effect survives in plausibility and inference-heavy continuation selection

## Support benchmark
### MMLU
- role: broader coverage check
- purpose: verify whether the effect generalizes across subjects without making MMLU the core thesis carrier

## What is NOT the main use case
The phase-1 paper is **not** mainly about:
- regex generation as an application product;
- chat alignment quality;
- multimodal reasoning;
- tool use;
- agent behavior;
- scaling to frontier model size.

Those may matter later, but they are not the article's core claim.

## Target paper claim
A careful version of the intended phase-1 claim is:

> Under matched backbone and training budget, a decoupled JEPA-style language model can outperform both a standard LM baseline and a coupled LLM-JEPA baseline on real reasoning tasks, while also showing stronger robustness and more structured latent behavior.

## Minimum evidence required to support that claim
At least one real benchmark must show:
1. repeatable improvement over the LM baseline;
2. repeatable improvement over the coupled JEPA baseline;
3. support from a robustness metric;
4. support from a latent diagnostic.

If only one of those holds, the paper claim must be weakened accordingly.

## Recommended paper narrative
1. **Problem**: token-level autoregressive reasoning entangles thought and expression.
2. **Hypothesis**: latent reasoning plus decoupled verbalization should be more robust.
3. **Method**: compare Gemma LM, coupled LLM-JEPA, and decoupled JEPA-Reasoner under matched budget.
4. **Primary evidence**: GSM8K.
5. **Transfer evidence**: RegexEval.
6. **Analysis**: robustness, bucketed horizon performance, latent diagnostics.
7. **Conclusion**: whether JEPA helps language reasoning, and under which conditions.

## Practical interpretation
If the project works, the real product-level lesson is not “use JEPA for regex.”
The real lesson is:
- **JEPA may be a better substrate for structured reasoning in language**.

RegexEval is there to stress-test alignment and formalization, not to define the entire program.
