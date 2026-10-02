---
name: jeeves
description: Classify text with Jeeves, a local 9B decision model on the user's PC, returning calibrated probabilities for choice, yes/no and score questions. Use when the user says to use Jeeves, or for bulk labelling, routing or scoring of many items where calibrated probabilities matter. Not for one-off questions you can answer yourself.
---

# Jeeves

Jeeves runs on the user's PC, reachable over Tailscale from any machine:

`http://pc.example.com:6871` (fallback `http://100.64.0.1:6871`)

## Check it is up

```bash
curl -s --max-time 5 http://pc.example.com:6871/v1/models
```

No answer means it is not running. Do not start it yourself: it needs the whole GPU and stops the user's main model. Ask the user to run, on the PC: `uv run lct down swift-qwen && uv run lct up jeeves`.

## Ask

```bash
curl -s http://pc.example.com:6871/v1/systemone -H 'content-type: application/json' -d '{
  "state": "<the text to judge>",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team handles this?",
               "criteria": {"billing": "Payment problems", "shipping": "Delivery problems"}},
    "urgent": {"type": "noul", "instructions": "Does this need a human now?"},
    "mood":   {"type": "score", "instructions": "How upset is the writer?", "criteria": ["Calm", "Upset", "Furious"]}
  },
  "options": {"think": false}
}'
```

- `choice` → `choice`, `probabilities` per option. `noul` (yes/no) → `noul`, the probability of yes. `score` → `score` (expected value), `probabilities` per level.
- `"think": false`: ~1 s per request. `"max_think": 512`: reasons first, more accurate, ~20 s per question. Start without thinking; use thinking for hard or low-confidence items.
- Several questions in one request share one read of `state`.

## Bulk

Write a script that sends one request per item, one at a time (the server runs one request at a time), and saves the raw answers next to the input. Report the distribution and the low-confidence items (for example, top probability below 0.7) rather than every row.
