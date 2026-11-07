"""ID-only gold interpreter. Standard library; no xgraph index or executor imports.

Edits replace all objects of a subject/relation pair. Execution unions every
outgoing object at each hop, with no ranking, sampling, label lookup or caps.
"""

from collections import defaultdict


def adjacency(triples, edits=()):
    graph = defaultdict(set)
    for subject, relation, obj in triples:
        graph[subject, relation].add(obj)
    replacements = defaultdict(set)
    for subject, relation, obj in edits:
        replacements[subject, relation].add(obj)
    graph.update(replacements)
    return graph


def execute(triples, start, relations, edits=()):
    graph = adjacency(triples, edits)
    frontier = {start}
    trace = [sorted(frontier)]
    for relation in relations:
        frontier = {obj for subject in frontier for obj in graph[subject, relation]}
        trace.append(sorted(frontier))
    return {"answers": sorted(frontier), "frontiers": trace}
