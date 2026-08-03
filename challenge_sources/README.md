# Local challenge-source overrides

The task generator normally reads the TensorCircuit challenge suite from
`../tensorcircuit/examples/challenge_suite`. Files placed under
`challenge_sources/challenge-XX/` take precedence for that challenge.

These overrides keep benchmark-design fixes reproducible in this repository
while retaining the external TensorCircuit suite as the default source for
unchanged tasks.
