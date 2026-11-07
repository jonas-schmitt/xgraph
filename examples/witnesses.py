"""CPU: inspect an executed path's endpoints, caps and entity-ID witness."""

import json

from toy_kg import graph, served_tokenizer

from xgraph import SpanSyntax, build_kg_index
from xgraph.programs import MatchedExecutor

tokenizer = served_tokenizer()
# This experimental selector needs numeric-suffix IDs and a count table.
# Empty counts supply neutral zeros, without asserting popularity.
index = build_kg_index(graph(numeric=True), tokenizer, SpanSyntax(), sitelinks={})
executor = MatchedExecutor(index, tokenizer)
result = executor.execute_set("Atlas", ["creator", "born in"], witnesses=True)
assert result["reached"] == ["Q3"]
assert result["witnesses"] == {"Q3": ["Q1", "Q2", "Q3"]}
assert result["cap_events"] == []
print(json.dumps(result, indent=2))
