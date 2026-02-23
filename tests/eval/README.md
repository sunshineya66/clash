# Search Quality Eval Harness

Measures retrieval quality using query-answer pairs from real usage.

## Metrics

- **Precision@5**: What fraction of top-5 results are relevant?
- **MRR (Mean Reciprocal Rank)**: How high is the first relevant result?
- **Hit Rate@5**: Does at least one relevant doc appear in top 5?

## Running

```bash
pytest tests/eval/ -v              # Run eval suite
pytest tests/eval/ -v --tb=short   # With failure details
```

## Adding test cases

Edit `tests/eval/cases.json`. Each case needs:

```json
{
  "query": "search query text",
  "expected_paths": ["path/to/expected-doc.md"],
  "category": null
}
```

The `expected_paths` list should contain file paths (or substrings) that a correct search should return.
