from tokenizers import Tokenizer, decoders, models, pre_tokenizers

import xgraph as xg

alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
raw = Tokenizer(models.BPE({char: i for i, char in enumerate(alphabet)}, []))
raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
raw.decoder = decoders.ByteLevel()
tokenizer = xg.ServedTokenizer(raw)
kg = xg.KnowledgeGraph(
    (xg.Entity("atlas", ("Atlas",)), xg.Entity("ada", ("Ada",))),
    (xg.Relation("creator", ("creator",)),),
    (("atlas", "creator", "ada"),),
)
index = xg.build_kg_index(kg, tokenizer, xg.SpanSyntax())
oracle = xg.Oracle(index, tokenizer, index.span_syntax)
state = oracle.initial_state()
for piece in ("⟦", " Atlas", " ·", " creator", " ·", " Ada", "⟧"):
    for token in tokenizer.encode(piece):
        state = oracle.advance(state, token)
print(state.last_span.triples)  # (('atlas', 'creator', 'ada'),)
