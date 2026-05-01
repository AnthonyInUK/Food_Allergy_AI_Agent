# Manual Labeling Guide (Food Agent)

## Goal
Build a small human-labeled set to measure real end-to-end answer quality.

## Labeling Unit
One row = one user question and one model answer.

## Fields
- `id`: unique case id
- `question`: user query
- `expected_key_facts`: key facts expected in answer (short phrase list)
- `model_answer`: model output text
- `label`: one of `correct` / `partial` / `wrong` / `proper_refusal`
- `error_type`: one of:
  - `none`
  - `retrieval_miss`
  - `hallucination`
  - `format_or_language`
  - `incomplete`
  - `timeout_or_iteration`
  - `other`
- `notes`: short reviewer note

## Label Rules
- `correct`: all key facts are present and no material contradiction.
- `partial`: some key facts are correct, but important details missing.
- `wrong`: major factual errors or contradiction with known data.
- `proper_refusal`: answer clearly states uncertainty / missing data without fabricating.

## Metrics You Can Compute
- Strict accuracy = `correct / total`
- Useful accuracy = `(correct + proper_refusal) / total`
- Partial rate = `partial / total`
- Hallucination rate = `hallucination / total`

## Recommended Setup
- Start with 50 cases.
- Two reviewers if possible; disagreeing rows go to adjudication.
