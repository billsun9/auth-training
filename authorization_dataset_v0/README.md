\
# Authorization Dataset v0

Minimal programmatic dataset generator for studying whether prompt-injection
training learns attack-specific shortcuts or contextual authorization.

## What this version generates

Three SFT training distributions:

- `attack_heavy`: explicit attacks are common; external imperative text is strongly correlated with unauthorized actions.
- `diverse_attack`: more styles/mechanisms, but the same shortcut correlation largely remains.
- `authorization_balanced`: authorized and unauthorized external instructions are balanced and, except for an odd-size remainder, emitted as matched counterfactual pairs. Pair members share source text, action, arguments, fact, style, and mechanism; only an explicit delegation policy and target differ.

Five evaluation splits:

- `iid`
- `lexical_ood` (uses phrase/template families excluded from training)
- `mechanism_ood` (holds out fake reasoning, policy citation, instruction redefinition, and completion-gate attacks from training)
- `auth_recombination` (uses a fixed source × action matrix excluded from training, with both authorization labels)
- `benign_control`

Every row is JSONL and contains the prompt, structured target, factors, and metadata.
Instruction text is composed from independently sampled action-phrase, style,
lexical-family, and mechanism-wrapper factors. Paired examples additionally
record their policy-template family and counterfactual pair ID in metadata.

## Quick start

Requires only Python 3.9+.

Generate a 1k-example training dataset for the balanced regime:

```bash
python generate.py \
  --regime authorization_balanced \
  --n-train 1000 \
  --n-eval-each 100 \
  --seed 0
```

Generate 1k training examples for all three regimes:

```bash
python generate.py \
  --all-regimes \
  --n-train 1000 \
  --n-eval-each 100 \
  --seed 0
```

Inspect:

```bash
python inspect_dataset.py data/generated/authorization_balanced/train.jsonl
python inspect_dataset.py data/generated/authorization_balanced/auth_recombination.jsonl
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
3. Create a formatting-only/benign SFT control.
4. Add a Hugging Face SFT script only after the generated labels look correct.

## Current limitations

This is deliberately a v0 generator, not a benchmark:

- synthetic language is template-heavy;
- source/action coverage remains synthetic and small, even though `auth_recombination` now enforces a fixed holdout matrix;
- factual answer targets echo a simple sentence rather than testing sophisticated summarization;
- no real tool execution occurs.

Those are useful next implementation targets once the basic factorization is validated.
