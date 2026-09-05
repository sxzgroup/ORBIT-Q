# Figure contract

The paper update follows the visual and metric contracts of
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105).

1. Updated Fig. 1c preserves the agent-by-framework matrix, places all
   configurations in one continuous layout without an old/new separator, and
   includes nine added TensorCircuit-NG campaigns through PR #29.
2. Updated Fig. 2b retains all thirteen model points and the previous non-Astra
   coordinates. Astra uses eleven paired measurements plus the approved
   50 s estimated Task 08 expert reference. Both axes remain linear
   with extra padding left of zero failure and below unit runtime. No footer,
   reference-reuse marker, or numeric runtime annotation is added to the figure.
3. Updated Fig. 4 preserves the paper's 2 × 3 wall-time, token, and efficiency
   layout. All token bars use the same total-token encoding because the source
   tables do not publish components for every configuration.
4. The expert-optimization overview reports paired end-to-end measurements;
   exact task reductions are distinguished visually; Task 08 remains a valid
   optimized implementation with a small measured point estimate.
5. Fable 5 and DeepSeek max remain excluded. Published outcomes are preserved
   per campaign: Task 08 is failed for the eight pre-Astra configurations and
   passed for Astra. Astra retains its automatic result; no blanket failure
   override or retroactive adjudication is applied.
6. Figure labels omit the repeated `high` qualifier. The accompanying text
   states that Sol, Terra, Luna, DeepSeek V4 Flash/Pro, Grok 4.5/4.6 and Astra use high
   thinking effort; the separately named Sol ultra configuration uses ultra.
7. Astra uses purple in model-specific encodings. Passed/failed and total-token
   bars retain their common colors; the matrix retains its count scale.
8. The Astra paired runtime figure supplies the Astra ratio in Fig. 2b: it uses
   same-host remeasurements for eleven tasks and the estimated Task 08 reference.
   The raw Task 08 OOM record is preserved separately; the replacement is
   not reported as a completed measurement. Its source is given in the data
   and accompanying text, not as an added figure footer.
   Its right panel reports expert/Astra speedup, not Astra/expert runtime.

Vector PDF is the publication deliverable; PNG copies are non-publication
previews for inline GitHub/PR display only.
Agent wall time, token use, artifact-runtime ratio, and expert-implementation
speedup remain separate quantities.
