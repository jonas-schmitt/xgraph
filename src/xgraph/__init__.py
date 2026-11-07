"""Constrain generated spans and execute paths against any knowledge graph.

Build a :class:`KnowledgeGraph` from application IDs and labels, compile it with
:func:`build_kg_index` for one :class:`ServedTokenizer` and :class:`SpanSyntax`,
then use :class:`Oracle` on CPU or :class:`Engine` in a GPU sampling loop.
Importing xgraph does not initialize CUDA or load model weights. The interfaces
are experimental; the module-level imports remain available as well. Public
objects are imported when first accessed, so lightweight readers do not load
indexing dependencies merely by importing the package namespace.
"""

from importlib import import_module

_EXPORTS = {
    "ClosedSpan": "oracle",
    "Engine": "engine",
    "EngineUnavailable": "engine",
    "Entity": "kg",
    "KGFormatError": "kg",
    "KGIndex": "index",
    "KnowledgeGraph": "kg",
    "Oracle": "oracle",
    "OracleState": "oracle",
    "Relation": "kg",
    "ServedTokenizer": "tokenizer",
    "SpanSyntax": "syntax",
    "TokenNotAllowed": "oracle",
    "build_kg_index": "index",
    "load_kg_index": "index",
    "save_kg_index": "index",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
