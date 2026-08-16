\
# Authorization Dataset v0

Minimal programmatic dataset generator for studying whether prompt-injection
training learns attack-specific shortcuts or contextual authorization.

## What this version generates

Four SFT training conditions:

- `attack_heavy`: explicit attacks are common; external imperative text is strongly correlated with unauthorized actions.
- `diverse_attack`: more styles/mechanisms, but the same shortcut correlation largely remains.
- `authorization_balanced`: reference, authorized, and unauthorized examples are emitted as matched counterfactual triplets. All members share the task context and fact; the reference has no embedded instruction, while the other two differ only in explicit delegation status and target.
- `capability_only`: the byte-identical capability-rehearsal rows shared by the other three conditions, with no hierarchy examples.

Seven evaluation splits:

- `iid`
- `lexical_ood` (uses phrase/template families excluded from training)
- `mechanism_ood` (holds out fake reasoning, policy citation, instruction redefinition, and completion-gate attacks from training)
- `auth_recombination` (uses a fixed source × action matrix excluded from training, with both authorization labels)
- `benign_control`

Additional shared evaluation splits:

- `auth_recombination_natural`: the held-out source × action combinations expressed with ordinary user-request wording rather than policy templates.
- `authorization_policy_ood`: matched authorization triplets using authorization phrasings excluded from every training condition.

Every row is JSONL and contains the prompt, structured target, factors, and metadata.
Instruction text is composed from independently sampled action-phrase, style,
lexical-family, and mechanism-wrapper factors. Paired examples additionally
record their policy-template family and counterfactual triplet ID in metadata.

The default training generation produces 3,000 regime-specific hierarchy rows
and appends 1,000 shared capability-rehearsal rows to every regime file. These
simple QA, extraction, JSON-formatting, and tool-call rows are byte-identical
across the three authorization files. The `capability_only` file contains
exactly those rows. Set `--n-capability 0` to disable them. The
`mechanism_ood` split also holds out a small closed-domain
extraction/classification subset in which untrusted data contains a conflicting
instruction; use `--closed-domain-eval-fraction 0` to disable that subset.

## Quick start

Requires only Python 3.9+.

Generate a 1k-example training dataset for the balanced regime and the shared
evaluation suite:

```bash
python generate.py \
  --regime authorization_balanced \
  --n-train 1000 \
  --n-eval-each 100 \
  --seed 0
```

Generate 1k training examples for all four conditions and one shared evaluation
suite:

```bash
python generate.py \
  --all-regimes \
  --n-train 1000 \
  --n-eval-each 100 \
  --seed 0
```

For the repository's canonical 3,000 hierarchy-row + 1,000 shared-capability-row / 200-eval layout:

```bash
python generate.py --all-regimes --n-train 3000 --n-capability 1000 --n-eval-each 200 --seed 0
```

Regenerate the small, tracked preview set (20 balanced-training examples and
10 examples for each eval split). This intentionally replaces `data/preview/`:

```bash
python generate.py --preview --seed 0
```

Inspect:

```bash
python inspect_dataset.py data/generated/train_authorization_balanced.jsonl
python inspect_dataset.py data/generated/eval_auth_recombination.jsonl
```

The resulting layout keeps experimental conditions separate from the shared
measurement suite:

```text
data/generated/
  train_attack_heavy.jsonl
  train_diverse_attack.jsonl
  train_authorization_balanced.jsonl
  train_capability_only.jsonl
  eval_iid.jsonl
  eval_lexical_ood.jsonl
  eval_mechanism_ood.jsonl
  eval_auth_recombination.jsonl
  eval_auth_recombination_natural.jsonl
  eval_authorization_policy_ood.jsonl
  eval_benign_control.jsonl
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Suggested immediate next steps

1. Inspect ~100 generated rows manually. The templates are intentionally simple.
2. Expand the factorization before increasing dataset size:
   - finer scoped delegation;
   - identical source text under counterfactual authorization;
   - source × action combinations held out explicitly from train;
   - realistic tool-call schemas.
3. The formatting/capability-only control is `train_capability_only.jsonl`: it
   contains the same 1,000 byte-identical shared capability rows included in
   every hierarchy regime, and no authorization or prompt-injection rows.
4. Add further real-world task families after the generated labels look correct.

## Current limitations

This is deliberately a v0 generator, not a benchmark:

- synthetic language is template-heavy;
- source/action coverage remains synthetic and small, even though `auth_recombination` now enforces a fixed holdout matrix;
- factual answer targets echo a simple sentence rather than testing sophisticated summarization;
- no real tool execution occurs.

Those are useful next implementation targets once the basic factorization is validated.
