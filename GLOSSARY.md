# xgraph

The vocabulary of xgraph. Terms are added here as they settle, so that code, docs, and conversations name each concept the same way.

Each entry names one concept. Aliases the repository already uses are given in the definition; `_Avoid_` lists only words that would mislead.

This public glossary covers the library and optional research utilities. Study
protocols, launcher scripts and complete reproduction materials are retained
separately; see [reproduction scope](docs/reproducing.md).

## Language

### Knowledge graph

**Knowledge graph**:
A set of entities, relations, and triples, abbreviated KG.

**Entity**:
A KG node with a stable id (an RDF IRI, an application string such as `item:atlas`, or a Wikidata QID such as `Q7186`), one or more labels, and optionally a description and classes. The core `Entity` stores its id and labels; descriptions and classes belong to source metadata or graph edges.

**Relation**:
A directed edge type with a stable id (an application string such as `rel:creator`, or a Wikidata PID such as `P108`) and a label.
_Avoid_: predicate, property (Wikidata's word; fine when quoting Wikidata)

**Triple**:
A fact (subject, relation, object) whose subject and object are entities.
_Avoid_: statement (in Wikidata a statement can also carry qualifiers and literal values)

**Label**:
One surface form of an entity or relation: its canonical label or an alias. Several entities can share a label.

**`mul` label**:
A label or alias in Wikidata's `mul` language: a default name shared by all languages, where some items keep their only name. The stream keeps it as the label of an item without an English label, and as an alias otherwise.

**Literal value**:
A date or quantity added to the KG as an object with rendered labels (`11 March 1952`, `1952-03-11`), so that a fact span can state it (`python -m xgraph.literals`). Strings and external identifiers stay out. A merged date is a date value joined onto another KG directory of the same items by QID (`python -m xgraph.literals merge-dates`), an entity labelled only at its stored precision (`6 June 1814` and `June 6, 1814`, `June 1814`, `1814`), so that a path span can reach it and bind the answer to it.

**KG directory**:
A source KG on disk: label JSONL, a numeric triples array, and source metadata, with optional sitelinks, counts and dates. The exact format is in `docs/guides/bring-your-own-kg.md`. The directory builder uses local Q/P numbers even for graphs unrelated to Wikidata; the in-memory API accepts arbitrary string IDs. Variants (literal values, edited KGs, probes) are KG directories too. `python -m xgraph.kqapro kg` writes KQA Pro's native KG in this format, with attribute values and fact qualifiers preserved in separate files.

**Source ID map**:
An application's preserved mapping from original entity/relation IDs to the local
Q/P numbers used by a KG directory and persisted versions. It identifies the
same nodes across builds and edits; it does not assert a Wikidata identity.

**Pinned reader**:
A request or immutable oracle state reading the base KG or one named registered
KG version throughout its decoding. Versions share the base label tables; they
are selected before decoding, rather than changed during an open span.

**Version isolation**:
Every KG read of a pinned reader uses its selected version, including inner
hops, degree ranking and the bound answer. Other registered versions contribute
no edges. Union paths unite senses and frontiers within that one version.
Slot reuse resets the binding; request replay restores it before decoding.

**Source-graph validation**:
Checking a source KG before indexing through `xgraph.kg.validate` or
`python -m xgraph.kg validate`: unique owner IDs, declared triple IDs, well-typed
labels and numeric arrays, and aliases/metadata whose owners are declared.
Errors carry the source file or table, one-based row and ID where available.
This structural check does not assert real-world truth or label writability.

**Sitelink count**:
An item's number of sitelinks, the pages of every Wikimedia site (Wikipedia in each language, Wikiquote, ...) that link to it, counted from the sitelink table (`python -m xgraph.wikidata sitelink-counts`) into a KG directory and from there into its KG index. A rough measure of how well known an entity is.

**RDF projection**:
The entities, relations, labels and asserted data triples retained from an RDFLib graph by `xgraph.rdf.from_rdflib`. Configured label predicates supply metadata, blank-node triples are skipped or rejected, and literal data objects are optional. No identity merging or inference is performed. See `docs/guides/rdflib.md`.

**RDF id mapping**:
The `rdf-ids.json` sidecar of a KG directory written by `xgraph.rdf.write_rdflib`. It maps directory-local integer storage ids back to original RDF ids; the existing index builder restores them before saving the KG index.

**RDF literal value**:
An opt-in terminal object of an RDF projection, with an opaque JSON id encoding its lexical form, language and datatype. Unlike the Wikidata literal importer, the RDF adapter accepts strings too. Export reconstructs the RDF term without normalization; different terms can share a rendered label.

**Pair deletion**:
A KG version edit `(subject, relation, None)` that removes every object of the pair without adding an edge. `xgraph.kgversions.write_kg_edits` stores arbitrary string-id edits and explicit pair deletions with checked edit hashes and a pin to the base index content. The legacy numeric version writer keeps its existing format.

**KG store**:
A KG directory in SQLite (`python -m xgraph.toolkg build`), for lookups outside decoding: the KG lookup tool, the walk, parse-then-execute, and real updates.

**KG world**:
One edited copy of the KG in which a set of MQuAKE cases is asked. In an interference-free world no case's edit rewrites a hop of another case's edited chain (`python -m xgraph.mquake --worlds`).

**Fresh KG world**:
A newly generated typed question component and its paired original, inner-hop-edit
and off-chain-edit conditions (`xgraph.fresh_worlds`). Its topology and nonce entity
labels are generated independently of model outputs. All conditions of one world
form one statistical cluster. Shared question templates support new-graph-instance
transfer, rather than unseen-language generalization.

**Controlled difficulty block**:
A paired set of fresh KG worlds sharing one semantic template and latent graph
blueprint, with depth, relation-label confusability and reachable frontier width
varied by fixed model-free construction rules. All cells, versions, arms and
families remain together in resampling.

**Difficulty interaction**:
The difference in WT minus strongest-agent joint version fidelity between deep,
confusable fresh worlds and shallow, distinct-label worlds. The v2 registration
holds frontier width narrow; it separately tests the pure factorial
depth-by-confusability interaction and sensitivity averaged over deep widths.

**Nonce entity label**:
A unique deterministic syllable string that names an entity in a fresh KG world.
It passes normalized collision checks against the declared KG labels and common
string corpus, and round-trips as several ordinary tokens in both served
tokenizers. This is a declared-corpus screen, not proof of absence from the web.

**Frozen-program replay**:
An offline diagnostic on fresh KG worlds: parse only the one-call planner's
original-condition predicted program, then execute that unchanged program with
matched execution in every KG version. Compare its ID-scored answers with
reusing its original matched-execution answer (`xgraph.fresh_worlds_analysis`).
It separates value updating from replanning; it makes no additional model call.

**Depth contrast**:
The secondary difference between WT minus agent original-condition ID accuracy
at four hops and that difference at two hops on fresh KG worlds. Whole-world
resampling retains paired arms and families within fixed template quotas;
its multiplicity and pilot-based assurance are separate from the registered
joint version fidelity primaries.

**Paired crossover model**:
The descriptive logistic model for WT-only versus agent-only correctness,
conditional on discordance, with family intercepts and a shared semantic-hop
slope. Separate-family and knowledge-status sensitivities, fixed cluster
resampling and claim boundaries belong to the study protocol. The fresh-world
predictive check uses frozen development-fitted probabilities and discordance
odds on original-condition outcomes without refitting. It predicts neither the
rate of disagreement nor absolute accuracy (`xgraph.fresh_worlds_analysis`).

**Multi-valued pair rate**:
The fraction of a fresh component's subject-relation pairs with two distinct
objects. The generator rounds the configured rate times its pair count to an
integer quota. Some gold-plan frontiers branch and reconverge; admitted questions
still have singleton denotations under every condition.

**Policy-arm hook**:
The gold-free interface for an opt-in policy on fresh KG worlds
(`policy-arm.json` from `xgraph.fresh_worlds`): request variants, KG directory,
KG store, KG versions, tokenizer-bound indexes and an ID-based output contract.
Its records use the original request ids and the independent evaluator; an
additional policy's hypothesis and compute budget require prospective registration.

**KG version**:
A named set of edits over one KG index, read by a sequence instead of the base KG (registered with `XGRAPH_KG_VERSIONS`, chosen by the per-request argument `xgraph_kg_version`): each edit (subject, relation, object) replaces every object of its (subject, relation) pair, so a sequence reading the version sees what the KG index rebuilt from the edited triples holds, without that index (`docs/kg-versions-adr.md`). An edit whose subject, relation, or object has no label is an unwritable edit: it writes no edge but still replaces its pair's objects. A KG world without new labels can be a KG version.

**Question component**:
A question's separate copy of its released subgraph within a disconnected KG. Entity numbers in the component occupy a half-open range; identical names in other questions denote different nodes. Requests anchor path mode inside the component. The optional `components` map in `kg-versions.json` restricts each reader's names, labels and triples to that range through SQL views (`KGStore.component`). These store bounds do not restrict unanchored engine requests.

**Recall stratum**:
Where an MQuAKE case's unedited hops stand in the served model's memory, probed closed book one hop at a time (`python -m xgraph.recall`): all recalled, one unrecalled before the first edit (the edited subject is then unknown to the model), unrecalled only after it, or no unedited hop. Lenient recall (`--lenient`) also counts a near miss, an answer and a dataset or KG name containing one another as whole words, as recalled.

**Case measure**:
How an MQuAKE case, one multi-hop question asked as several paraphrases, is scored from its paraphrases' answers (`python -m xgraph.editing`): MQuAKE's own measure counts a case when any paraphrase gets the edited answer (`new_any`); the panel beside it counts all (`new_all`), more than half (`new_majority`), or the answer voted over the paraphrases (`voted`).

**Relation sequence**:
The relations of a question's chain in order, such as an MQuAKE case's edited chain `P26>P108`, or MINTQA's released relation labels: questions sharing one ask the same kind of question about different entities, so their results are correlated. A clustered comparison (`python -m xgraph.editing --compare ... --cluster`) resamples cases by relation sequence instead of one by one. MINTQA's development and validation panels (`xgraph.mintqa_panels`) allocate whole relation sequences to disjoint pools before sampling questions.

**Calibration panel**:
A seeded sample of development cases on which candidate arms are run before a confirmatory test is frozen, to select its arms and measure how often they agree (`python -m xgraph.editing panel`): stratified by relation sequence so that it spans every sequence, each giving cases by the square root of its size, and each drawn case weighted by the cases of its sequence it stands for.
Development calibration selects P, X and A per model family by panel per-question accuracy, then all-three accuracy, then lower measured cost; X uses the selected P's answer delivery.

**Saturating exposure witness**:
A recorded set of question IDs exposed under the registered hypothesis and
containing an entire declared population (`xgraph.case_alias.exposure_proof`). Its presence proves that the
population has no unused question, regardless of additional development,
validation, pilot, calibration or analysis exposure lists. It establishes
exhaustion without claiming an exhaustive inventory of historical artifacts.

**Hypothesis-relative exposure**:
Prior production or inspection of an item's model or policy outcomes that the
owner's prospective ruling makes relevant to a particular hypothesis. An
explicitly disclosed different-model feasibility run may be outside that scope.
The initial-case study records those rulings verbatim and inventories ID-bearing
artifacts across worktrees before taking the population's set difference.

**Alias-only prefix reachability**:
A case-preserving substring of a question, starting and ending at Unicode word
boundaries, whose served-tokenizer encoding in the label prefix context reaches
a nonempty row range only in the alias index. This gold-free screen measures
potential access to a new token branch, not actual model use, a correctness flip
or changes to ownership and ranking of an already shared surface. An empty pool
has no estimable affected fraction.

**Update event**:
A cluster of cases connected by the subject–relation pairs their edits rewrite (`xgraph.editing.update_events`). A temporal comparison resamples whole update events, keeping questions about a shared update together.

**Joint assurance** (historically joint power):
The simulated probability that the complete H1a → H1b → H2 fixed sequence passes in both model families, integrating calibration-cluster uncertainty, repeated arm selection, fresh target clusters and serving-pilot uncertainty (`python -m xgraph.power simulate`). Each step intersects its components and both families at one-sided alpha .025. Marginal endpoint or family powers do not determine it without their dependence.

**Arm by stratum interaction**:
How much more one arm beats another on one set of cases than on a second, such as two recall strata: the paired difference of the two arms on the first stratum minus that on the second (`python -m xgraph.editing --compare ... --interaction`).

**Population contribution**:
A stratum's weighted paired score difference divided by the full panel's weight
sum (`xgraph.posthoc`). Contributions of disjoint exhaustive strata add to the
overall difference. The conditional difference instead divides by the stratum's
own weight sum. A post hoc contribution is descriptive, not causal mediation.

**Executed relation-ID sequence**:
The ordered unions of relation IDs actually followed at each capped, selected
frontier (`xgraph.posthoc.executed_sequence`). A written label can name several
IDs; only IDs present among that frontier's outgoing relations are followed.
This differs from comparing labels or their whole-index ID unions and does not
assert that an executed relation is relevant to the question.

**Discordant pair**:
A question on which exactly one of the compared arms is correct under the same
scorer. A paired direction model conditions on these pairs; their win odds do not
alone give an absolute accuracy difference. The descriptive MINTQA analysis
(`xgraph.error_analysis`) reports both directions and samples questions uniformly
within each direction with a fixed seed.

**Retrospective validation transport**:
Scoring a development-fitted analysis on already exposed validation outcomes.
It keeps validation out of coefficient fitting but remains post hoc; it is not
independent confirmation or a prospective registration.

**Code drift**:
A changed byte identity of an in-repo source, reducer or lock file recorded in
the post hoc reducer's separate `code_provenance` block. Its `--check` lists the
changed files and passes only when all statistics and evidence identities still
agree exactly. The producing code identities remain in the saved records.

### KG index

**KG index**:
What one KG and one served tokenizer compile into, once and offline: the label index, the adjacency, the relation index, and the root masks.
_Avoid_: grammar (the KG index replaces one)

**Served tokenizer**:
The tokenizer of the model whose output is constrained. Every token sequence in a KG index uses its vocabulary, so a KG index is valid only for that tokenizer.
`ServedTokenizer` owns a serialized snapshot and requires IDs from zero through
its vocabulary size minus one, including added tokens. Changes to the caller's
tokenizer leave that snapshot intact.

**Tokenizer fingerprint**:
The SHA-256 of the served tokenizer's serialized form. A KG index records it, and the oracle and the engine refuse a KG index whose fingerprint differs from their served tokenizer's.

**Special token**:
A token that stands for no text, such as end of text. It is allowed in free text and never inside a span.

**Tokenization context**:
The text assumed before a label when it is tokenized, so that the label index matches the tokens a model writes inside a span. Each label is tokenized with the label prefix in front, as text that starts at a token boundary, because the span opener always ends at one.

**Label index**:
Every entity label, tokenized in the tokenization context, one row per label, with rows sorted lexicographically by token sequence. A typed token prefix selects one row range. Oracle construction refuses entity or relation label rows that are not sorted as token tuples.

**Excluded label**:
A label the label index leaves out, reported with the reason: a blank label, a label whose written form (the label prefix and the label) contains span syntax (a delimiter or the operand separator anywhere, or an operator at its start), a label holding the text of a special token, or an encoding rejected by exact label encoding. An index records that its build excluded the operand separator and the operators (`KGIndex.expressions`); one saved before has to be rebuilt for expressions.

**Exact label encoding**:
The versioned build rule `exact-v1`: a nonempty encoding's known token bytes
must concatenate to exactly the UTF-8 bytes of the label prefix plus the source
label. Unknown bytes, empty encodings and normalization losses are excluded and
reported in every builder. New KG indexes record the rule in
`build_config["label_encoding"]`; loading an older index preserves its rows.

**Row**:
One label's token sequence in the label index, together with the entity it names.

**Start spelling mismatch**:
A descriptive error-analysis signature: a question's spelling of the gold start
has no row for its QID, and a non-space token prefix shared by that spelling and
the written WT start excludes every gold-start row. The canonical label can
still be writable. It identifies a blocked spelling branch, not a measured
accuracy gain from adding aliases (`xgraph.all_new_analysis`).

**Row range**:
The contiguous rows of the label index that share the token prefix typed so far, written `[lo, hi)`. Advancing narrows it to the rows that continue with the sampled token.

**Depth**:
The number of label tokens typed in the current segment; the next token is read from that column of the label index.

**Adjacency**:
For each subject entity, its outgoing triples as (relation, object row) pairs, ordered by relation and then by object row, with one pair per label row of the object, so that an object with aliases appears once per label.
_Avoid_: neighbour list (ambiguous about direction)

**Object slice**:
The adjacency entries of one subject and relation, in label index order, so that the objects matching a typed prefix are where the object slice and the row range overlap.

**Relation index**:
The relation labels, tokenized and ordered like the label index. In a fact span, only the relations the subject has are allowed.

**Unique-alias index**:
A KG index built with `python -m xgraph.wikidata index --unique-aliases`: relation aliases shared by several relations are omitted, as are aliases that equal another relation's canonical label. Canonical labels remain. This changes the relation index, not the entity label index.

**Initial-case alias**:
An opt-in entity surface added by `--initial-case-aliases` at KG index build time
(`initial-case-v1`): change the first code point's case, preserving every later
character. Only canonical entity labels generate aliases; punctuation, uncased
initials and case mappings to multiple code points generate none. Title-case and
sentence-case conversion of words are excluded. Exact owner/surface duplicates
produce no added row. A shared surface keeps one row per owner and uses the same
homonym and committed-path resolution as any existing shared label.

**KG index fingerprint**:
The SHA-256 of the JSON object mapping sorted saved-index filenames to their
SHA-256 hashes (`xgraph.index.kg_index_fingerprint`). It includes the index header,
labels, relations, adjacency and declared optional arrays; reports are excluded.

**Surface normalization**:
A declared equivalence on writable entity surfaces. The isolated semantic oracle
in `xgraph.identity` uses `nfc-ascii-v1`: NFC, ASCII uppercase-to-lowercase mapping,
then NFC again. It preserves punctuation and whitespace and does not perform
general Unicode case folding, compatibility normalization or transliteration.
Its `exact-v1` control keeps surfaces unchanged. Neither changes the production
label index or span language.

**Identity set**:
The distinct entity IDs owning a writable label with the same normalized key as
a completed surface, intersected with the request's component and anchor when
present. Alias multiplicity and canonical-label status give no owner priority.
This set describes possible references, rather than the intended entity.

**Normalization transducer**:
A recognizer that maps an admitted surface's bytes to its declared normalized
key. The isolated semantic oracle buffers a complete bounded reference before
normalizing it. An incremental token-masking implementation would also retain
pending UTF-8 and Unicode normalization state; that implementation is proposed,
rather than supplied by the semantic oracle.

**Explicit root binding**:
An experimental reference decision choosing one QID from an identity set before
any relation executes. The chosen ID initializes a singleton frontier in the
pinned KG version. Start ranking cannot replace it with another homonym when a
path becomes empty. The isolated semantic oracle supports this decision; the
production engine does not.

**Root binding marker**:
The experimental ` ⟦# ` text between an entity surface and its chosen QID in
`xgraph.identity`, as in the root prefix `⟦ Shared Vale ⟦# Q2⟧`. The nested closer
commits the ID; the surrounding path remains open. These are ordinary token
bytes, without a new served-tokenizer vocabulary entry.


**Base-index fingerprint**:
The SHA-256 identity `KGIndex.base_fingerprint`: source KG content (entities,
relations, triples and labels, including excluded content), served tokenizer,
build configuration and every compiled table. Builders compute it once; default
loads retain its recorded string. Compiled KG versions and bound persisted edits
compare that identity without scanning arrays. Source ordering and directory
serialization are significant. This
identity differs from the KG index fingerprint of saved artifact bytes.

**Base binding**:
Pairing a KG index and its versions by recorded base-index fingerprint, in time
proportional to headers rather than array contents. Default binding trusts
immutable local files and catches different recorded bases. Full verification
(`load_kg_index(verify=True)`, the validate CLI's `--index`, or
`XGRAPH_STRICT_FINGERPRINT=1`) recomputes index contents against the record to
also detect modified arrays. String-ID edits separately check their recorded
JSONL hashes on read. Binding does not authenticate headers or detect changes
to numeric version edits; external artifact hashes cover those threats.

**Legacy fingerprint status**:
`fingerprint_status="legacy"` marks an index or persisted edit set without a
recorded complete base identity. Compatibility loading preserves its bytes and
mask semantics; compiled versions retain the previous tokenizer and
entity/relation/label-row count check for legacy base indexes, represented as a
cheap legacy identity. Archived string-ID edits with only `base_index_identity`
still check that older content pin, including on direct oracle admission.
An unbound edit file cannot establish its original base retrospectively.
Verification cannot check legacy contents against an absent original record.
`fingerprint_status="verified"` denotes a complete recorded identity, not a
load-time integrity result. Strict loading (`strict=True`, or
`XGRAPH_STRICT_FINGERPRINT=1`) refuses legacy inputs.

**Unwritable triple**:
A triple the KG index leaves out because its subject, relation, or object has no label left to write (every label was excluded). The KG index counts them.

**Writable KG**:
The ID triples represented by a KG index in the KG version a sequence reads.
Version edits replace their entire subject-relation pair before unwritable
triples are omitted; aliases supply label rows for the same ID edge.

**Root mask**:
A precomputed mask of the tokens allowed at depth 0 of a segment, where almost every row is in range.

**Date table**:
The optional part of a KG index that holds the entities that are dates (merged dates), sorted, each with its date packed into a 32-bit key: year, month, day, and precision (`xgraph.index.pack_date`), whose interval bounds compare as the dates do (`date_lo`, `date_hi`). The KG index build makes one from a KG directory with merged dates; a date whose year does not fit a key is left out.

### Decoding

**Executed-answer mass**:
The total probability of finite trajectories of one fixed locally masked policy
that finish at the same stable answer ID, including different relation programs
and label spellings. Explicit failure outcomes complete the distribution.
This is policy probability, not factual correctness or calibration.

**Reachable-answer envelope**:
A conservative set of outcomes attainable from an unexplored trajectory prefix.
It may union alternatives beyond the per-trajectory frontier cap; recapping that
union can discard an attainable answer and invalidate a probability bound.

**Answer-mass certificate**:
A strict lower-versus-upper probability comparison proving a unique most probable
executed answer, or explicit failure, under a fixed finite policy. Upper bounds
include unexplored mass, unseen answers, failures and numerical rounding.

**Thought stage**:
An optional first model call in `xgraph.generate` with native thinking enabled and no
KG constraint. `--think-budget` caps all tokens generated in this call, including
native channel delimiters. Zero keeps the existing single-stage behaviour. The call
stops at the native thought terminator or the budget; the answer stage closes a
truncated thought through the model's chat template.
Positive budgets are supported only by the baseline and constrained arms; the other
arms, including the masked KG agent, walk and execute-and-resume, refuse them.

**Answer stage**:
The continuation after a thought stage: the same conversation with its thought in
the final assistant turn's `reasoning` (also supplied as `reasoning_content` for
template compatibility), followed by answer content. Only the
answer is decoded under the requested arm and only its content is scored. An answer
prefill belongs to this content; the thought is prompt history the engine skips.

**Shared thought**:
A thought stage saved by request id in a thought file and reused across answer arms
with `xgraph.generate --thoughts`. It fixes deliberation for a planner comparison.
Reused records still charge its original prompt tokens, generated tokens and model
call, and identify which calls actually executed in the current run.

**Span**:
A region of output between a span opener and a span closer, inside which every token is constrained by the KG index. Text outside spans is free.

**Span syntax**:
The delimiters that mark spans in the output: span opener, label prefix, separator, escape marker, and span closer, and the operand separator and operators that write expressions. They are configuration, not fixed in code.

**Span opener**:
The delimiter text that opens a span, `⟦` by default; the study also calls it the trigger.
_Avoid_: trigger token (a delimiter can take several tokens)

**Label prefix**:
The text written between the span opener (or a separator) and a label, a space by default, so that a span reads `⟦ Marie Curie⟧`. Labels are tokenized with it in front, which gives the served tokenizer's usual word tokens.

**Separator**:
The delimiter text between the segments of a fact or path span, ` ·` by default; the next label's prefix follows it, so a fact span reads `⟦ Marie Curie · employer · University of Paris⟧`.

**Operator**:
The text an expression span starts with, after the span opener and the label prefix: `earlier:`, `later:`, or `same:` by default (`SpanSyntax.operators`, the first comparing by the earlier date, the second by the later, the third by a shared value), as in `⟦ earlier: Jaws | Star Wars · publication date⟧`.

**Verdict entity**:
One of two entities, labelled `yes` and `no` (`xgraph.index.VERDICTS`), that an index built with verdicts holds (`python -m xgraph.wikidata index --verdicts`), so that a value comparison's verdict binds the answer span after it as an entity does. No unbound subject names one.

**Operand separator**:
The delimiter text between an expression's two operands, ` |` by default; the next label's prefix follows it.

**Span closer**:
The delimiter text that ends a span, `⟧` by default. It is allowed where a complete label ends in the subject segment (an entity span) or the object segment (a fact span), and after an escape.

**Mode**:
What a span is restricted to: an entity (entity span, `⟦ Marie Curie⟧`), a triple (fact span, `⟦ Marie Curie · employer · University of Paris⟧`), or a multi-hop path (path span). Entity and fact spans share the span opener: a span becomes a fact span when the model writes a separator after the subject. Grounded free text is free text that may contain spans of any enabled mode.

**Segment**:
One part of a fact or path span: subject, relation, or object. Each segment is matched against its own row range.

**Phase**:
The part of the output a sequence is in: free text, a span, or an escaped span. The engine adds an error phase for a sequence advanced by a token outside its mask.
_Avoid_: mode (that is what a span is restricted to)

**Escaped span**:
The rest of a span after an escape: free text up to the span closer.

**Branch**:
Inside a span, one way the next token can be read: as the next label token (the label branch), as part of the span closer (the closer branch), as part of the separator (the separator branch), or as part of the escape marker (the escape branch). Several branches can be alive at once, because one token's bytes can serve more than one; a token is allowed if it keeps at least one branch alive.

**Constraint state**:
The fixed-size state an engine keeps per sequence: its phase, segment, depth, row range, the progress of its closer, separator, and escape branches, its candidate subjects and relations, and the span it closed most recently; with the no-repeat rule, the fact spans it closed; in derivation and path mode, the entities its last span reached and the open path's frontier (with committed paths, each frontier entity's start rank), and with expressions a comparison's operator and its second frontier; with residual-goal STOP, which question words the start and consumed relations cover; and the KG version it reads.

**Advance**:
The per-step update of a constraint state by the sampled token: narrow the row range, and switch segment or close the span when a label completes.

**Mask**:
The per-step set of token ids of the served vocabulary that a constraint state allows next. The sampler applies it to the logits.
_Avoid_: logits mask (it is applied to logits but computed without them)

**Allowed set**:
The token ids a mask admits at one step.

**Feasible label continuation**:
A canonical label-token suffix supported by the pinned, capped frontier, together
with the legal byte-exact span-closer branches after a completed label. The
systems assay compares this language on bound answer spans.

**Canonical-token filter**:
An intersection of a text grammar's mask with the feasible label continuations'
token trie and closer automaton. Construction and application are charged to the
comparator. It removes extra tokenizations and cannot repair missing admissions.

**Logit-directed support query**:
The specified existential test for a ranked candidate token against the pinned
constraint state, stopping at the first admissible token to preserve masked
greedy decoding. It is a proposed access algorithm, not an engine option.

**Forced token**:
A token that is the only member of its allowed set, so the KG fixes it without sampling (jump-forward). Also called a KG-forced token. Where the escape marker remains allowed, a span's label token is not forced; residual-goal STOP can force the closer's tokens.

**KG draft**:
A label token the KG index proposes as a speculative draft where the label branch has exactly one continuation; the model verifies it, so it can still escape instead.

**Escape**:
Writing the escape marker inside a span: the rest of the span is free text, and the span closes unresolved. It lets the model state what the KG lacks instead of forcing a match that is grounded but wrong.

**Bound escape**:
An escape in a span whose subject the derivation binding restricts: an anchored first subject, or the answer after a path span or fact span, which may only name the entities the KG reached. Switched off (`XGRAPH_BOUND_ESCAPES=0`), such a span must name one of them, while a path's relations and unbound spans keep the escape: abstention then means the KG lacks a relation, not that the model disbelieves the value it reached.

**Escape bias**:
An engine option: a value added to the logits of the tokens that commit to the escape marker inside a span (those that start it and continue no other branch). A negative bias makes escaping costlier without forbidding it.

**Escape marker**:
The delimiter text that starts an escape, ` ⁇` by default, as in `⟦ ⁇ an unlisted lab⟧` or `⟦ Marie ⁇ née Skłodowska⟧`. It is allowed at any token boundary inside a span, except in the middle of another delimiter, for a bound subject with bound escapes disabled, or at a relation boundary where residual-goal STOP forces the closer.

**No-repeat rule**:
An engine option: a fact span cannot close with the three labels of a fact span its sequence closed before, so a model caught repeating a fact must write another object or escape. It needs escapes, the way out.

**Rejected mass**:
The probability the model put on tokens the constraint disallows at a step, before renormalizing over the allowed set. The log of the allowed mass (one minus the rejected mass) is the step's KL divergence from renormalizing; summed over a span (its pressure), it measures how far the constraint pushed the model.

**Release**:
An engine option: inside a span, a step whose rejected mass is too high (the allowed mass falls below a threshold, or the model's own top token is disallowed) may write only the escape marker, which steers the span toward closing unresolved instead of grounded on a fact the model did not want. It acts per step.

**Derivation mode**:
An engine option that binds spans by entity rather than by label: after a fact span, spans may only name the entities its object reached; a fact span's subject, and after a fact span an entity span, names one entity, the best-connected with its label. A multi-hop answer then follows the KG's facts hop by hop, so an edited fact changes it. A request can anchor it to its question's entities, where it then starts.

**Anchor**:
The entities a request names for derivation or path mode to start from (the per-request argument `xgraph_anchor`, a list of entity IDs in the index): the first span may only name them, as if a fact span had reached them. Anchors come from the question's annotated entities, or, as a linked anchor, from entity linking on the question text (`python -m xgraph.prompts wwq --anchor --link-kg`). Anchor IDs form a set before label rows are capped. An explicit anchor with no writable label rows is rejected by the oracle, engine and matched execution. A span without an anchor is unanchored.

**Anchor pre-validation**:
A CPU check of every anchor in a complete request set before model startup (`python -m xgraph.anchors`). It lists empty, malformed, unknown and unwritable anchors and exits nonzero if any fail. It rejects unknown IDs even in a mixed anchor; the engine and oracle retain their existing masking semantics. Duplicate known IDs are accepted. `--require-anchors` also rejects omitted or null anchors.

**Path span**:
In path mode, a span naming an entity and then relations only, `⟦ The Hobbit · author · place of birth⟧`: the path starts from every entity with the written label (the relations decide which senses go on), each relation must be one the current entities (the frontier) have, the KG walks it to their objects, and the spans after it, the answer among them, must name the entities it reached; the span right after it, its answer, is an entity span and cannot go on as another path. The model never names an intermediate entity. A path names at most eight relations, then must close.

**Plan**:
In path mode, the relations a path span writes, read as property ids by label or alias; an answer's plan is exact when it equals the gold chain's properties (`python -m xgraph.planning`).

**Path mode**:
An engine option (`XGRAPH_PATHS=1`) in which spans are path spans and bind the spans after them as derivation mode does. The relation path is executed by the KG inside decoding, as it is written; parse-then-execute runs the same kind of path after decoding.

**Observed path**:
A path span with opt-in frontier observations (`XGRAPH_OBSERVE=k`, launcher's `XG_OBSERVE=k`, generation's `--observe-paths k`): after the start and every relation, the engine writes ` ·→` and up to k indexed entity labels in best-connected order, joined by ` ;`, with ` (+N more)` for omitted entities of the capped frontier or a prefixed `∅` for empty. A committed observation's payload has one allowed token per step. The observation itself changes no execution or sense commitment; the model then chooses the next relation or closes and answers, or uses an enabled frontier selection. The answer span has no observation.

**Shadow history**:
The shadow-contrast proposal's second rendering of the same pinned reader's
frontier observations, with entity readback labels renamed consistently by ID.
Relation prefixes and entity selections remain aligned by their actual IDs.
This research term describes a context intervention, not an engine option.

**Shadow contrast**:
The rejected proposal to contrast real and shadow histories' next-relation
logits using the context-aware decoding score and the real reader's allowed set.
Its context construction does not establish a distinct inference rule.

**Relation-pair witness**:
A different KG subject on which two competing relation IDs have complete,
nonempty, unequal object sets. A visible witness also needs present labels and
unequal rendered label sets. A missing relation or truncated response is not
such a witness. Structural separation does not identify the intended relation.

**Active relation-pair probe**:
The rejected proposal to retrieve a relation-pair witness at an uncertain
decision, show its paired examples, then continue from the original frontier.
Diagnostic nodes never enter answer support. The specified operation reduces
to ordinary exemplar disambiguation with the same selector and context;
it adds no engine option.

**Menu mode**:
An opt-in frontier path mode (`Oracle` and `Engine`'s `menu=k`, `XGRAPH_MENU=k`, launcher's `XG_MENU=k`, generation's `--menu k`, k from one through sixteen): force a relation menu after the start and every executed relation, and after every frontier selection when observations are enabled. Zero or unset is off. It composes with observed paths and rejects audit mode and expressions. Menus change no relation support, frontier cap, start commitment, hop count or STOP coverage.

**Path demonstration set**:
An explicit prompt control (`--path-demos 4|5`, `XGRAPH_PATH_DEMOS`, launcher's `XG_PATH_DEMOS`) for constrained path mode with audit off. Four selects the original maintained path questions; five also teaches Germany → highest point → mountain range, in blind, observed or menu language. Unset retains the existing prompts: menu mode adds the fifth question, ordinary blind and observed paths keep four. Custom demonstrations and replayed prefills remain caller-owned.
An observed request with `fixed_path_demonstrations: true` preserves its supplied
assistant examples during observation/selection conversion; interface instructions
are still added to the system message.

**Effective request messages**:
The messages after generation has selected demonstrations and added interface
instructions, before the served tokenizer applies its chat template. Identical
messages in saved request files can yield different effective request messages
when one request preserves its supplied demonstrations and another converts them.

**Reproduction anchor**:
A saved observed-path run compared with its historical output on frozen question
ids, with identical effective request messages and declared generation settings
(`xgraph.mechanism_repro`). Exact reproduction remains the primary admission.

**Noise-band reproduction**:
The Qwen-only secondary mechanism admission derived from the single registered
anchor repeat (`xgraph.mechanism_admission`). It applies the registered function
to text identity and signed accuracy-count differences, then tests the original
historical-versus-anchor comparison. It requires oracle review and prompt
identity, grants no tolerance to another family and authorizes no variations.
The band is a descriptive sensitivity, without a confidence or future-drift claim.

**Selection by delivery interaction**:
The fresh-world policy contrast `(s1d1-s1d0)-(s0d1-s0d0)`, where `s` enables
frontier selection and `d` enables guaranteed answer delivery. The prospective
fallback holds effective request messages and demonstrations constant across
the four cells. Each world contributes its mean ID correctness across original,
inner-hop-edit and off-chain-edit conditions; whole-world inference retains all
cells and families. The population amendment draws independent v2 worlds only
from the deep, confusable, wide cell, so frontier selection can affect support
beyond the observation display. This is an availability intervention, without a mediation
claim.

**Fresh-world factorial assurance**:
The fallback's separate predictive probability that both frozen signed
interactions pass confirmation. Its template-specific sparse Dirichlet model
preserves each world's joint outcome vector across families, cells and conditions,
and applies the same whole-world inference, sequence sensitivity and numeric-drift
rules as confirmation. It requires its own complete measured pilot; the original
fresh-world joint version fidelity assurance cannot substitute for it.

**Adaptive observation width**:
An opt-in display rule (`observe_width=W`, `XGRAPH_OBSERVE_WIDTH`, `XG_OBSERVE_WIDTH`, `--observe-width [W]`): show every entity when the capped frontier has at most W entities, otherwise keep the ordinary top-k observation. The CLI uses eight when enabled without W; zero/unset is off. Width changes neither execution caps nor selection or answer support.

**CLOSE/NEXT decision**:
An opt-in menu rule (`menu_close_next=True`, `XGRAPH_MENU_CLOSE_NEXT=1`, `XG_MENU_CLOSE_NEXT=1`, `--menu-close-next`), requiring observations and menus. After a post-start observation and any required selection, force the opener ` ⟦!`, then admit `CLOSE⟧` or `NEXT⟧`. CLOSE forces the surrounding path closer and bound answer delivery; NEXT forces the current frontier's menu and ordinary relation separator. NEXT is unavailable at STOP, the hop cap or an exhausted frontier. Required selections precede this decision; their preselection menus are suppressed. The start menu remains automatic, after any required start selection. Decision wrappers consume no relation or STOP coverage.

**Deferred frontier selection** (DEFER):
An opt-in alternative at a required frontier selection (`observe_defer=N`, `XGRAPH_OBSERVE_DEFER`, `XG_OBSERVE_DEFER`, `--observe-defer [N]`): after the forced selection opener, `DEFER⟧` preserves every capped frontier entity and its start rank, consuming that boundary's selection without another observation. A path permits at most N deferrals; the CLI uses two when enabled without N, and zero/unset is off. An entity named DEFER still uses its indexed label prefix and observation marker. This option implies guaranteed answer delivery; the answer chooses a label within the final committed frontier.

**Relation menu**:
In menu mode, a GPU-forced suggestion such as ` ⟦? mountain range ; country (+2 more)⟧` within the path. Relation scores order distinct available relation ids; ties use versioned frontier adjacency frequency, then relation index id. Each id displays its canonical English label when writable, otherwise its best question-scoring writable alias, with ties by label text. An id is omitted only when every alias is blocked. The same label policy serves terminal audits. The limit truncates display alone; omitted choices remain writable. ` (+N more)` counts omitted ids and a prefixed `∅` denotes none. Parsers ignore menu text rather than executing its labels.

**Menu marker**:
The canonical token sequence for `" " + syntax.opener + "?"`, default ` ⟦?`. The configured closer ends the menu, without ending the surrounding path. Indexed labels exclude both delimiters. With entity observations off, this marker commits a complete start or relation label; a longer label remains writable until commitment. With observations on, the marker and menu follow the entity payload automatically.

**Request path settings**:
An opt-in `xgraph_path_options` dictionary, enabled by
`XGRAPH_REQUEST_PATH_OPTIONS=1` (Engine's `request_path_options=True`), that selects
one request's menu and observation settings within the engine's allocated
capabilities. `Engine.set_path_options` installs it after `begin`; omitted switches
are off and begin/reset restore constructor defaults. It keeps the KG, support,
base path options and end-after-answer fixed. `path_demos` records the client's
rendered demonstration count, without changing engine masks or rewriting history.
This lets a menu/observed first call and a blind masked second call share one
server and KG tables.

**Frontier selection**:
An optional observed-path decision (`XGRAPH_OBSERVE_SELECT=1`, launcher's `XG_OBSERVE_SELECT=1`, generation's `--observe-select`; default off): after an observation, ` ⟦=` and an entity label choose every owner of that label within the capped frontier, including entities not displayed. The observation marker commits the selection and the engine forces a new observation of the narrowed frontier. Original start ranks survive, and the next relation executes from the narrowed set. Selections are recorded by boundary, 0 after the start and n after relation n; they consume no relation or STOP coverage.

**Guaranteed answer delivery**:
An optional observed-path rule (`XGRAPH_OBSERVE_ANSWER=1`, launcher's `XG_OBSERVE_ANSWER=1`, generation's `--observe-answer`; default off): after a normally closed executed path, force the canonical answer prefix, then an entity span bound to the final committed frontier, then only the configured EOS ids. This implies end after answer and forbids an escaped answer. Token budgets can still truncate the output. Escaped paths and unselected entity-only spans retain their existing semantics.

**Completion debt**:
For a declared acceptance condition, the minimum number of further served tokens
needed to reach acceptance from a constraint state, or infinity if none can.
The budget-solvency audit includes mandatory observations, a normally closed
executed path, its bound answer, closers and EOS. Debt is relative to the pinned
reader's capped token language, rather than question correctness or every path
in the source KG. An immediate-close continuation supplies an upper bound,
which need not be a lower bound when another relation can shorten the answer.

**Budget solvency**:
A rejected proposal to retain remaining-token state and mask successors whose
completion debt exceeds it. The audit reduces this operation to existing
distance-to-acceptance pruning on the engine's transition system;
it adds no production option or mask.

**Terminal endpoint observation**:
An opt-in readback after a revised relation path closes (`observe_terminal=W`,
`XGRAPH_OBSERVE_TERMINAL`, launcher's `XG_OBSERVE_TERMINAL`, `--observe-terminal [W]`).
Zero or unset is off; the bare CLI option uses eight. For multiple deliverable
endpoints, the engine forces `Endpoints: ⟦ labels⟧` before the bound answer, with
up to W distinct entities ranked by descending sitelinks, descending versioned
adjacency degree, then first label-index row. Start commitment and the answer-row
cap precede the display. Singleton support skips the display. Undisplayed
endpoints remain selectable; no further relation edit is allowed. Forced display,
answer-prefix and EOS tokens are generated tokens and observation tokens. The
second-call arms are `audit-terminal` and `two-pass-terminal`.

**Terminal display order**:
The presentation order of the terminal observation's already selected entities
(`--terminal-order identity|reversed|random`, `--terminal-seed`). It never changes
the displayed set or bound answer support. Identity preserves the original
ranking; random uses a seeded permutation shared by oracle and CUDA engine.

**WT-PC**:
A separate development policy that averages constrained endpoint choices over
permutations of a fixed WT revision's terminal display, optionally subtracting
the question-free endpoint log prior. Votes use canonical entity ids and every
cached and new call is charged.

**Required frontier selection**:
An optional observed-path rule (`XGRAPH_OBSERVE_REQUIRED=1`, launcher's `XG_OBSERVE_REQUIRED=1`, generation's `--observe-required`; default off, implies frontier selection): after an unselected observation with more than one distinct entity in its capped frontier, force the selection opener before any relation, closer or escape, including before STOP and the hop cap. Display truncation and shared labels do not reduce this count. One label decision per boundary keeps every matching entity, as frontier selection does; the resulting observation permits continuation or closure even when the selected label still has several owners.

**Observed-path loop guard**:
An optional observed-path rule (`XGRAPH_OBSERVE_NO_LOOP=1`, launcher's `XG_OBSERVE_NO_LOOP=1`, generation's `--observe-no-loop`; default off): within one path, a relation id cannot execute twice from the same capped entity set. Frontier identity ignores display order, aliases and start ranks. A shared relation label is unavailable when any supported relation id it names would repeat. The fixed-size history holds every permitted hop; an exhausted frontier cannot take a separator, while its closer remains available. Selecting a previously held entity is still allowed.

**Observation tokens**:
Generated tokens in an observed path's committed observation marker and payload, or a relation menu, including a truncated payload and excluding prompt tokens (`observation_tokens` in generation records and summaries). Guaranteed answer delivery also counts its forced answer prefix and supplied final EOS tokens. Required frontier selection counts its forced selection opener; optional selection openers and model-chosen selection labels are excluded. They are also included in `generated_tokens`, which charges all decode steps. Punctuation is encoded independently and concatenated with unchanged label-index rows: the canonical observation tokenization, which need not equal encoding the full text.

CLOSE/NEXT additionally charges its forced decision opener, CLOSE's surrounding path closer and NEXT's relation separator. Command words and their nested closer, including DEFER, are decisions and are excluded from this subset. Every one of these tokens still consumes the ordinary generation budget.

**Audit mode**:
An opt-in path mode (`XGRAPH_AUDIT=1`, launcher's `XG_AUDIT=1`, generation's `--audit`): write the ordinary blind path, then read its audit block and make exactly one audit decision. It needs frontier support, sitelink counts, observations and expressions off, and served EOS ids. It forces a bound answer and EOS after the decision; STOP, KG versions, anchors and the ordinary execution caps still apply to blind planning. Audit-off keeps the existing path language.

**Audit block**:
GPU-forced text after a blind path's closer: `Executed: ⟦ labels⟧`, then `Relations (terminal): ⟦ relations⟧` and `Relations (penultimate): ⟦ relations⟧`, then `Decision: ⟦`. Endpoints use sitelinks, versioned degree and first-row order, after the frontier cap. `XGRAPH_AUDIT_ENTITIES` (`XG_AUDIT_ENTITIES`, `--audit-entities`) bounds endpoint display, default 20. `XGRAPH_AUDIT_RELATIONS` (`XG_AUDIT_RELATIONS`, `--audit-relations`) bounds each menu, default 8; zero hides both menus. Singleton endpoints are shown; `(+N more)` counts omitted capped entities or available relation ids. Forced displays, decision and answer prefixes, and final EOS count as observation tokens; model decisions and labels do not.

**Audit decision**:
One of `KEEP⟧`, `SELECT ⟦= label⟧`, `APPEND · relation⟧`, `REPLACE · relation⟧`, or `DROP⟧` after an audit block's decision opener. SELECT keeps every terminal owner of its label. APPEND takes a terminal relation below the hop cap; REPLACE takes a penultimate relation instead of the last written relation; DROP removes that relation only from paths with more than one. Menus and display bounds do not restrict choices. Committed paths choose the best surviving start after the decision. Edit commands force another endpoint display, then the bound answer prefix; no further planning is allowed. Parsed and recorded plans retain the original relations, decision and resulting relations.

**Relation score**:
A host-compiled per-request rank for audit and per-step relation menus (`audit.RelationScores`, `Engine.set_relation_question`, `Engine.set_audit` or `Engine.set_relation_scores`): the maximum IDF-weighted question stem overlap with a relation label or alias, divided by the square root of its stem count. IDF counts each relation once per stem. Up to 64 positive-scoring relation ids carry equal-score ranks in each engine slot; unlisted ids follow them. Menu ties use frontier adjacency frequency, then relation index id. Scores order displays and never constrain executable support.
An optional request field `relation_ranks` (serving extra `xgraph_relation_ranks`)
replaces these lexical relation ranks with explicit relation-ID/rank pairs.
`xgraph.relation_rank` builds them offline from question text and relation names
within the request's KG version and question component using the frozen dense
encoder. Alias display priorities remain lexical. Omission preserves the lexical
path, including its ties and menu size.

**End after answer**:
An opt-in path-mode option (`XGRAPH_END_AFTER_ANSWER=1`; `Oracle` and `Engine` take `end_after_answer=True` and `eos_token_ids`): after the entity answer span immediately following a closed path, relation plan or comparison closes, only the served model's supplied end-of-sequence (EOS) token ids are allowed. Earlier masks stay the same. An escaped answer does not trigger it; an unrelated entity span does not either.

**Frontier**:
In a path span, the entities the path has reached so far: first every entity the subject label may name that has triples (only the anchor's entities when the request has one), then, after each relation, its objects.

**Frontier cap**:
The limit on distinct entities retained at each path boundary (`MAX_REACHED` in
the oracle). Retention uses first label-index row, with committed paths first
using start rank. It can remove the sole valid continuation. The answer-row cap
uses the same limit but counts label rows, so aliases also consume it.

**Witness edge sequence**:
A sequence of writable KG triples joining an admitted start to a resolved path
endpoint, with one relation ID compatible with each executed relation label.
Consecutive edges share an entity. The soundness argument uses witnesses;
constraint states need not store them (`docs/engine-contract.md`).

**Stored witness**:
A full witness edge sequence retained in the original engine or program answer
record. Stored endpoint IDs, entity chains without relation IDs, printed frontiers
and relation menus alone are not stored witness edges.

**Reconstructed witness**:
A witness edge sequence found after generation by searching a pinned KG for a
recorded positive-hop program and an emitted answer ID or compatible answer label.
It establishes support in that pinned graph, without recovering actual execution
events, hidden resolved IDs or the model's use of the evidence. The independent
checker in `eval/witness-audit-2026-10-30/checker.py` keeps it separate from a
stored witness.

**Path soundness**:
Every resolved positive-hop frontier path endpoint and its bound answer has a
witness edge sequence in the pinned version's writable KG, under the premises
in `docs/engine-contract.md`. This guarantees grounding of resolved IDs, not
question correctness or completeness over graph paths.

**Frontier entity visit**:
One entity present at one boundary of an executed path, including its start. Summing
these visits measures occupied frontier slots across boundaries; the same entity
at a later boundary counts again. This is separate from constraint-state advances,
label-token steps and adjacency reads. A stopped path contributes no later visits.

**Enumerated walk**:
One explicit sequence of directed, relation-labelled KG edges from a start entity.
A walk may revisit entities and relations. Distinct walks can reach the same
entity or have the same tokenized labels, so a walk count is neither a frontier
size nor a token-trie node count.

**Entailed frontier**:
A proposed frontier that also follows edges derivable from a declared positive
rule set in the sequence's KG version. The CPU feasibility replay
(`xgraph.entailment_spike`) bounds base edges per written relation and retains
matched execution's selections, commitment and caps. It changes no decoding mask.
P31 membership inherits along P279; P31 itself is not transitive. This replay
implements a path-rule subset, not a complete OWL profile.

**Derivation witness**:
A proposed proof of an entailed edge: a directed acyclic graph of rule
applications whose leaves identify stored triples and schema axioms in a pinned
KG version and rule set. An endpoint witness can use these proofs in place of
stored edges. A proof establishes derivability under its declared rules, not
question relevance or real-world truth.

**Support ladder**:
A path-planning control that keeps the start constraint, syntax and hop cap fixed and varies relation support (`XGRAPH_RELATIONS`, or `Oracle` and `Engine`'s `relations` argument): the live frontier, the global relation vocabulary, or relation successors. The weaker supports execute no path during decoding and leave answers unbound; matched execution and engine delivery can deliver every arm afterwards.

**Relation vocabulary**:
Every writable relation label and alias in the KG index's relation index. Vocabulary support admits them at every relation segment, independently of the start and earlier relations.

**Repair visibility**:
The prompt-only WT repair intervention: after
identical frozen W first calls, vary added question-ranked global (R), frontier-
local (L), question-blind global (S), or absent (N) relation suggestions. Repair
candidates, endpoints, instructions and execution constraints remain fixed.

**Recorded stride gate**:
The variable-stride development score and its completion receipt, bound by file
paths and SHA-256 in the repair-visibility child registration. Pilot authorization
requires an owner decision binding those hashes and the registration; it verifies
the score's Boolean failure and a decision time at or after receipt completion.
This evidence replaces the pilot's calendar proxy; confirmation keeps its embargo.

**Cohort projection**:
The repair-visibility confirmation input containing exactly `id`, `question` and
`q_entity` for every sealed WebQSP core ID. The owner exporter verifies the
registered RoG question-only shards before selecting the core, then the existing
cohort freeze preserves its topic clusters. Gold and model outputs are forbidden.

**Owner source export**:
The repair-visibility manifest materialized from registered metadata and frozen
W recipes. It pins the histories, original requests, executor inputs and frozen
confirmation cohort. Gold retains its registered identity until scoring. A
stat-only preview lists paths and commands without reading sealed inputs; only
the owner's export hashes W histories.
Pilot answer-source identities come from registered panel metadata; annotations
are read only by scoring after complete generation and finish receipts.

**Source-pin amendment**:
A prospective child registration that updates implementation content hashes
before study data exist. It preserves arms, cohorts, matching, analysis, gates,
budgets and authorization evidence, and records the parent identity and exact diff.
It may re-pin the eligible-population registration and its input identity only
when the eligible ID list equals the parent's; every other confirmation field
must still match.

**Question-blind sham**:
The repair-visibility S arm's deterministic hash ordering of all writable relation
IDs and lexical alias selection, without questions or gold exclusions. Its
suggestion count and served prompt token budget match R/L; shared bounded padding
equalizes token length without altering suggestions.

**Initial support interaction**:
The conditional R-minus-N repair accuracy benefit on questions whose original W
capped committed support excludes gold, minus that benefit where it includes
gold. The original support defines both strata before the intervention; revised
programs never define eligibility or strata.

**Relation successor**:
A pair of relation ids for which some KG entity is an object of the first and a subject of the second, in the KG version being read. Successor support admits any vocabulary relation first, then only successors of the preceding label's compatible ids. Ambiguous labels retain every compatible id; this is a schema walk, with no entity frontier.

**Schema graph**:
In PSR, a graph whose nodes are entity types and whose edges are typed relations:
a type-level triple exists when some KG triple connects entities with those
types. It requires entity type membership as input. Relation successor support
does not supply this type membership.

**Schema-level path**:
In PSR, a sequence of entity types and relations searched in the schema graph
before grounding its relation sequence into instance-level KG paths. It is not
an executed path span or its live frontier.

**Relation plan**:
A syntactically complete path span written under vocabulary or successor support. Its labels name the index, but the plan has not been executed and has no resolved entity result or answer binding. The span immediately after it keeps path mode's entity-only answer syntax.

**Committed path**:
In path mode with the option `XGRAPH_COMMITTED_PATHS=1`, a path span that resolves to what one start entity reaches: the first entity the subject label may name, in best-connected order (as a fact span's subject is chosen), from which the whole path is non-empty. The path still offers the relations of its whole frontier; only what it resolves to, and so the answer it binds, commits to one sense. Each frontier entity carries the rank of the best start that reaches it, and the closer keeps the entities of the least rank present. An anchored path that starts from one entity is unchanged.

**Best-connected order**:
The order in which a span picks one of the entities a label names: most adjacency entries (its degree, in the KG version the sequence reads) first, then first label index row. It chooses derivation mode's fact span subject and bound entity span, and ranks a committed path's starts.

**Sitelink rank**:
An engine option (`XGRAPH_START_RANK=sitelinks`; the default is `degree`) that puts the entities with the most sitelinks first in best-connected order, then degree and first label index row: for every choice that order makes, a committed path's starts and derivation mode's subject and bound entity span alike. It needs a KG index with sitelink counts and refuses one without; KG versions keep the counts, and their own degree breaks ties.

**Unquoted start**:
An engine option (`XGRAPH_UNQUOTED_STARTS=1`) under which an unbound subject (every subject outside derivation and path mode; there, one no anchor, fact span, or path has bound, such as a path's start) cannot name a label that starts with a double quotation mark, straight or typographic, or a guillemet. A model that copies the opening quote of a title its question quotes then writes the title instead of being steered into another label that starts with one. Objects and bound subjects, such as the answer after a path, name such labels as before; an entity whose every label starts with one cannot start a path.

**Forced opener**:
A request that starts inside a span (the per-request argument `xgraph_open`): its prompt ends with the span opener as a prefilled assistant turn (`python -m xgraph.generate --open`), and its constraint state starts as if the opener had just been written, so its first token continues the span and the answer begins with a span, in path mode a path. It composes with an anchor and a KG version.

**Replayed prefill**:
A request whose prompt ends with a prefilled assistant turn that the engine advances over (the per-request argument `xgraph_prefill`, the prefill's text; a request's `prefill_replay` field in `python -m xgraph.generate`): the constraint state follows the prompt tokens that hold the prefill, through the request's anchor and KG version, as if the model had written them, so that a prefilled path span binds the answer the model writes after it to the path's frontier. A prompt that does not end with the prefill, or a prefill token the mask disallows, declines the request: it is left unmasked, and the decline is counted and reported. Plan repair delivers a chosen plan this way (`python -m xgraph.repair deliver`), and engine delivery any plan matched execution read (`--executed`).

**Pending request**:
One adapter slot's staged admission settings, held until begin/end and its
options, anchor, KG version, question and opener are applied. A replayed prefill
keeps the same record until prompt history is ready. Repeated admissions retain
their begin/end order; an absent anchor preserves one staged by an earlier
callback, while an explicit empty anchor remains distinct.
Field order and completed admission stages survive a failed flush, so retrying
does not repeat a successful begin/end batch or an earlier completed stage.

**Host policy normalization**:
Constructor argument checks shared by the oracle and engine, including implied
observation, frontier selection and end-after-answer settings and served-model
EOS ids. It shares no transition, frontier, ranking or native execution code.

**Arm configuration**:
Validated generation-reader settings and prompt sources prepared during argument
parsing, before model construction. Each arm keeps its own settings and protocol.

**Generation run**:
Execution results and the timing context needed to write generation records and
receipts. It retains the model through receipt writing.

**Replay status**:
The per-request worker receipt for a replayed prefill, transported using `xgraph_replay_id` and saved as `replay_status` by ordinary generation. `replayed` confirms that the worker advanced the prefill under its mask; `declined` records a tail mismatch or disallowed replay token; `unconfirmed` records a missing or invalid receipt. Declined and unconfirmed replays score as failures while retaining raw completion text and all costs. A caller without a status consumer receives an error on a declined replay.


**Unresolved span**:
A span closed after an escape. Its text is unverified.

**Grounded**:
A resolved span names an entity, triple, or executed path of the KG index. Relation plans under vocabulary or successor support are unexecuted and carry no such guarantee. Grounded does not mean correct: the KG may lack the true fact, and the model may pick an irrelevant one.
_Avoid_: verified, factual

### Milestones

**Quality gate**:
The check that ends the first milestone: whether constrained spans reduce factual errors without lowering answer accuracy, with the unresolved-span rate reported alongside. Kernel tuning starts only after it passes.

**Baseline**:
The same model and prompt decoding without the engine. The quality gate compares constrained output against it.
_Avoid_: unconstrained model (the model is the same; only decoding differs)

**Arm**:
One configuration of model, prompt, and decoding whose outputs are scored together, such as the baseline, the constrained arm, a retrieval arm, or the fallback arm (`python -m xgraph.generate --arm`).

**Competitor qualification**:
A prospective check that a published method's specified search, required inputs
and released trained components support a faithful comparison with the frozen
readers inside the engineering and resource limits. A citation, source release
or cached retrieval alone does not qualify an arm. The rule and exclusion
boundary form part of the study protocol.

**Frozen validation manifest**:
A sealed specification written before any validation output exists
(`xgraph.mintqa_validation`): the clean code commit, panel and KG input hashes,
model inventory, ordered arm commands and prompt controls, registered comparisons
and inference definitions. Generation and scoring refuse a changed specification,
code or input. The manifest declares whether weights use SHA-256 or file size and
nanosecond modification time. A completed stage binds its inputs, outputs and
provenance to that same specification. It supplies internal validation on exposed
MINTQA, not evidence of a fresh benchmark.

**Completion receipt**:
A record binding a registered stage's outputs and launch events by hash to its
seal and stage specification. The menu-control queue issues one only after
generation validation or successful CPU output collection. A successful process
without a receipt remains unvalidated; resume checks the receipt and every bound
artifact before skipping the stage.

**Composite arm**:
An arm whose records a CPU stage built from other runs, such as matched execution of free paths, plan-length routing, or a concatenation of runs over parts of the questions. Its cost sources are its records and every stage and run it consumed, declared once each (`python -m xgraph.frontier`, `name=records.jsonl+run.jsonl+...`). Timed CPU stages add their own resource costs; an untimed source is listed under `not_counted`.

**Generation completion contract**:
A stage's frozen generation flags, request hash, clean code commit and complete
request-ID panel, checked before its completion receipt is written
(`xgraph.menu_control.execute_stage`). Terminal endpoint observation width must
match that stage's flags: absent or zero for W first calls, eight for terminal
second calls. Missing output metadata for the width means zero.

**Reproduction anchor**:
The historical-observation factorial's selection-off, delivery-off arm (`s0d0`),
compared with the historical observed-path output on frozen development ids before
the GPU owner runs variations. This is an experimental arm, distinct from the
entities a request anchors its path to.

**Exact reproduction**:
The primary reproduction-anchor status, `exact-reproduction`: all frozen output
texts match and every effective request message, including the answer prefill,
has identical bytes.

**Near reproduction**:
The secondary reproduction-anchor status, `near-reproduction`, whose text-count
and exact-match accuracy tolerances are registered and hashed before generation.
Every effective request message must still have identical bytes; differing texts
require a first divergent generation token. Both model families must pass exact
or near reproduction before factorial variations run (`xgraph.mechanism_repro`).

**Numeric-drift band**:
The registered sensitivity attached to each reproduction-factorial contrast:
plus/minus the absolute population-weighted accuracy difference between the
reproduction anchor and historical output, scored identically. An absolute
contrast at or below that band is labelled not identified. It is separate from
sampling uncertainty and does not bound unobserved drift in other arms.

**CPU stage**:
A command that prepares or transforms records for an answer, such as matched execution, answer delivery, plan repair, or routing. Its summary marks `cost_kind: cpu_stage` and records elapsed wall seconds, this process's CPU seconds, and peak resident memory (`peak_rss_mib`, the process high-water mark). It makes no new model calls, even if its records copy generation fields.

**Readiness decision**:
A read-only report from finished producer artifacts (`xgraph.readiness`). It keeps
a registered numerical pass or failure separate from incomplete evidence and an
unresolved registration. A cost-only pass does not certify live answer quality;
a complete semantics benchmark does not admit a transfer configuration. The report
records consumed input hashes and writes only inside its own checkout's target
directory.

**GPU allowance**:
A stage's maximum charged GPU occupancy within its queue's remaining budget,
derived from the registered workload forecast with startup and workload headroom.
Durable launch intent immediately before Docker and container end after Docker
returns are recorded through `XG_GPU_EVENTS`. CPU preparation, transformations,
locks, headroom waits and retry sleeps consume no container allowance. Failed
launch intent is charged; a missing end charges through confirmed cleanup.
An unclosed attempt, invalid events or unconfirmed cleanup retains the reservation.

**Stage wall bound**:
A separate elapsed-time limit covering preparation, locks, headroom waits and
execution together, preventing indefinite waits without charging them as
container occupancy. Controlled serving trials retain their forecast-derived
trial watchdogs.

**GPU ledger correction**:
An appended accounting event that preserves a failed attempt's original start and
end records. Menu-control excludes a proven GPU-free attempt only when a correction
records that no container started, identifies both original records by hash, and
binds the supporting evidence by file identity. An unfinished reservation cannot
be released by this correction.
**GPU budget ledger**:
The fresh-world owner queue's persistent, seal-bound record of GPU attempts,
reservations and charged container occupancy, including failed model startup.
Stage wall time is recorded separately. Invalid events or unconfirmed cleanup
retain the full reservation. Only GPU stages append attempts and charged seconds;
CPU stages leave this measured input fixed for scoring, analysis and provenance.
An unclosed attempt requires reconciliation before resume. A completion receipt
requires valid outputs, native sidecars and hash-bound container events.

**Same-build noise floor**:
Observed differences between repeated executions with the same declared source,
runtime image, model, tokenizer, KG index, request order and decoding settings.
It is an empirical repeatability measurement, conditional on the recorded build
and cache identities. One repeated run does not establish a bound on future
run-to-run variation or transfer that bound to another arm.
Any later near-reproduction tolerance amendment uses a fixed function registered
before the repeat. Passing that derived band alone does not authorize factorial
variations; effective request identity and both-family admission still apply.

**Batch invariance**:
The runtime property that a request's numerical results are independent of the
other requests processed with it. It needs supported kernels and model backends;
a fixed decoding seed alone does not establish it.

**Request cost distribution**:
The p50 and p95 of prompt tokens, generated tokens, or model calls per question, with the number of questions whose cost is known (`distributions` in generation summaries and the accuracy-cost frontier). A multi-call reader sums its calls' tokens per question; a composite arm sums its generation sources by question id before taking percentiles. These token and call distributions are independent of latency measurements.

**Cost per correct answer**:
All questions' cost divided by the number of correct answers, including the cost
of incorrect answers. The survey-weighted version divides weighted total cost
by weighted correct-answer mass (`xgraph.validation_analysis.cost_per_correct`).
Observation tokens remain a subset of generated tokens and are not added again.

**Correctness transition matrix**:
The paired wrong-to-wrong, wrong-to-right, right-to-wrong and right-to-right
counts between two arms on the same questions, with survey-weighted masses and
shares (`xgraph.validation_analysis.transition_matrix`). The off-diagonal
weighted difference equals the second arm's accuracy minus the first arm's.

**Failure signature**:
A deterministic label supported by a saved incorrect record, such as a missing
final answer, repeated lookup or first relation disagreement with the released
chain (`xgraph.error_analysis`). The first supported rule supplies a primary
label that partitions errors; overlapping labels retain every supported rule.
It is an observable description, not an established causal explanation.

**Answer surface type**:
A lexical date-like, number-like, boolean-like or other-label classification of
the released answer (`xgraph.error_analysis.question_strata`). It is a proxy
kept separate from semantic answer type, which remains unrecorded when the
question metadata does not supply it.

**Development-to-validation shrinkage**:
A shared pair of arms' development accuracy difference minus its validation
difference (`xgraph.validation_analysis.shared_shrinkage`). Positive shrinkage
means the signed difference contracted; a negative value means it expanded.
It describes those measured arms and supplies no correction for unvalidated arms.

**GPU ledger**:
The owner queue's persistent record of GPU stage attempts, reservations, status
and charged GPU time (`xgraph.operator_queue`). Generation charges owned container
occupancy, including model startup and confirmed cleanup, separately from stage
wall time. Native agreement and replay charge their GPU stage's elapsed time.
Unfinished or unmeasured attempts retain their full reservation, including cleanup
failures whose GPU lifetime is unknown. Ledger updates atomically
replace the file. Frozen manifests and completion receipts are write-once
artifacts.

**CPU queue rehearsal**:
An end-to-end check of an owner queue's real CLI and shell launcher against its
frozen stage catalog, cohort and registration, in an isolated repository with
synthetic inputs and external execution stand-ins. Production verification,
request building, retained-token accounting, witnesses, scoring, receipts and
ledgers execute on the CPU. It qualifies orchestration and failure handling;
native agreement, serving fidelity, model quality and GPU cost remain owner work.

**Controlled serving trial**:
One frozen workload measured alone on the GPU under a declared cache condition, concurrency and memory budget, repeated with randomized arm order (`docs/reproducing.md`). Startup, warm-up and the measured interval have separate boundaries. Cost uncertainty is estimated across repeated trials.

**Trial watchdog**:
The separate supervisor process that bounds a controlled serving trial's wall
time from its registered forecast, requests thread stacks, kills a stuck worker
and its descendants, stops its owned server and retains a failed trial report.
Its deadline works even while the worker holds the Python GIL in native code.

**Request heartbeat**:
A flushed progress line after a declared number of completed questions in a
serving trial's warm-up or measured workload. It reports completion counts and
phase; it contains no question or answer content.

**Serving seal**:
A separate immutable consumer specification (`xgraph.post_queue`) binding the
nominated validation policy, fixed development timing panels, ordered serving
and transfer commands, code and input hashes, and cost inference rules before
GPU execution. A consumer checks the validation producer's original checkout
read-only. Complete output receipts bind stage artifacts to this serving seal;
partial outputs are preserved and block continuation.

**GPU-hour ledger**:
A persistent record of reserved GPU time, including failed startup and teardown.
Each charge identifies its source trial and hash and states the timed boundary.
An unmeasured workload forecast is separate from an accrued charge; moving a
failed trial aside to permit a restart does not remove its charge.
The reduced serving study's `serving-budget.json` binds one lifetime cap to its
serving seal. Serial cell workers and engine replay reserve from that same cap
before launch, retaining failed or unknown intervals. Its conservative boundary
encloses worker preparation through confirmed cleanup; wheel builds and headroom
waits precede cell reservations. Exhaustion is terminal `INCOMPLETE` with no top-up.

**Pilot restart**:
A new attempt after a settled failed validation budget pilot. A frozen
restart binds the old plan, ledger and archived container events by hash, carries
all prior charges into a fresh append-only ledger, and retains the lifetime cap.
Fresh event, log and timing destinations prevent reuse of failed policy prices.
An attempt settled as a measured failure before launch carries zero occupancy
only when its hash-bound event file is empty. A launched container requires a
positive charge. Repeated restarts verify and retain the inherited ledger prefix.

**Pilot stage reuse receipt**:
A hash-bound record of a complete successful generation stage, its original
command, timing metadata, opaque artifacts and reviewed unchanged producing code.
A pilot restart may reuse that stage's artifacts and successful policy price.
Its original occupancy remains charged once through the inherited ledger; reuse
adds no new occupancy. Failed attempts supply charges rather than policy prices.

**Container occupancy**:
The reserved interval from durable `XG_GPU_EVENTS` container launch intent
immediately before Docker through its end, including model startup, idle
residency and failed launches. Wheel preparation, build and launch locks,
headroom admission and retry sleeps are excluded. An unfinished launch is
conservatively charged through confirmed cleanup; invalid events retain the
attempt's reservation. This measures reserved time, not GPU utilization or
controlled serving cost.

**Pilot restart**:
A new attempt after a settled failed validation budget pilot. A frozen
restart binds the old plan, ledger and archived container events by hash, carries
all prior charges into a fresh append-only ledger, and retains the lifetime cap.
Fresh event, log and timing destinations prevent reuse of failed policy prices.

**Pilot stage reuse receipt**:
A hash-bound record of a complete successful generation stage, its original
command, timing metadata, opaque artifacts and reviewed unchanged producing code.
A pilot restart may reuse that stage's artifacts and successful policy price.
Its original occupancy remains charged once through the inherited ledger; reuse
adds no new occupancy. Failed attempts supply charges rather than policy prices.

**Population-weighted dedicated cost ratio**:
The ratio of policy and reference-agent means of attributed dedicated GPU-seconds
per admitted question (`cost.weighted_dedicated_ratio`). Frozen survey weights
include timing-subset inclusion probabilities. Its paired bootstrap resamples
complete repetition blocks and questions or whole relation sequences within hop
strata, keeping each stratum's population mass fixed. The upper bound is compared
with the declared saving threshold. Batched per-question attribution is an
allocation of the reserved interval, rather than a device busy-time measurement.

**Population-matched timing subset**:
A fixed sample without replacement from frozen MINTQA development members
(`serve_mintqa.timing_panel`). Integer quotas approximate the population mixture
over equal survey-weight/hop strata. Subset weights divide each member's source
weight by its within-stratum selection probability, preserving stratum mass.
Its physical batched throughput describes that particular mixture; weighted
batched attribution does not measure a new population-mixture workload.

**ID-hash timing subset**:
A fixed prefix of development question IDs ordered by the SHA-256 bytes of
`seed:id`, with ID as tie-breaker (`serve_mintqa.timing_panel`). Smaller prefixes
are nested in larger prefixes. Selection reads only IDs; each selected member's
source survey weight is divided by the subset inclusion probability. Physical
throughput describes that fixed mixture; weighted attribution is a sensitivity.

**Live policy trial**:
A controlled serving trial in which actual generated replies determine later
prompts, tool operations, answer selection and conditional escalation
(`serve_mintqa.run_policy`). Every admitted question is charged through final
delivery, including forced displays, unsuccessful calls and CPU/tool idle time.
Native engine intervals and cache usage describe model work; the complete
reserved workload interval defines dedicated GPU-seconds. Quality must be scored
from these retained outputs rather than inherited from a frozen reader replay.

**Frozen policy catalog**:
The complete recipes in `xgraph.policy_catalog`, including first-call
demonstrations, display width, generation budget and seed, the optional second
call's builder and terminal display, and the agent's tool budget and feedback.
Each stage also records its development generation argv and omitted defaults,
including context, KV allocation and memory share. A CPU argv audit compares
the real launcher with original development provenance, lists every permitted
path difference in the validation seal and leaves unfinished sources pending.
`B2` is blind five-demonstration path planning followed by plain two-pass
revision; `BT` adds terminal readback to the second call at width eight.
A validation seal records one recipe or a prospectively declared family mapping.
Serving and WebQSP can read that same seal through `--policy-freeze`; their
artifacts retain its hash and each selected recipe's identity. WebQSP keeps its
Freebase demonstrations and complete answer-set delivery as explicit transfer
variants.

**Frozen reader replay**:
A controlled serving trial in which every recorded model call is issued sequentially and the reader's actual tool operations execute, with recorded replies determining subsequent decisions and prompts. New completions supply timing and token usage; the recorded trajectory supplies final delivery. It measures serving cost for fixed decisions, with quality supplied by the matched frozen evaluation.

**Client admission**:
The instant a question enters the load generator's queue. Whole-question latency and visible-content time to first token include the wait from admission until a worker starts it, followed by all required model and tool calls through final delivery (`xgraph.serving.measure_workload`).

**Matched prompt serving delta**:
The constrained arm's serving measurement minus its unconstrained control's, paired within a repetition with identical prompts, opener, sampling, token budgets and question-order seed. It includes effects of changed output histories and batching; kernel attribution requires a separate measurement.

**Accuracy-cost frontier**:
Every arm on one benchmark, accuracy against measured cost per question: workload wall seconds, CPU seconds where measured, prompt and generated tokens, model calls, and tool calls, summed over the arm's cost sources (`python -m xgraph.frontier`), with request cost distributions and the largest recorded process memory peak. An arm is on the Pareto front when no other arm is at least as accurate and at least as cheap on the chosen cost axis and better on one. Historical batched wall time alone does not establish controlled serving cost.
_Avoid_: frontier alone (path mode's frontier is the entities an open path reached)

**Agent accuracy-cost envelope**:
The best measured agent accuracy available as a cost budget increases
(`xgraph.budget_frontier`): collapse tied costs to their best accuracy, take a
cumulative maximum, then interpolate linearly in log cost between positive-cost
knots. Raw points retain non-monotone budget outcomes. There is no estimate
below the cheapest measured point; beyond the largest cost, the best observed
policy remains affordable without predicting an unmeasured gain. A cheaper,
more accurate policy can dominate every measured agent even where no
matched-cost interpolation exists. This exploratory curve is separate from
path mode's entity frontier.

**Fallback arm**:
An arm that answers constrained first and answers again without the engine where the constrained answer wrote the escape marker or hit the token limit.

**Plan-length routing**:
Path mode first; a question whose path-mode answer wrote no path span, or a plan of at most one relation, takes another arm's answer instead, such as one-hop retrieval's (`python -m xgraph.route`). It reads only path mode's own output.

**Parse-then-execute**:
A baseline in which the model writes a relation path without the engine and the KG store executes it afterwards (`python -m xgraph.programs`).

**Matched execution**:
Parse-then-execute with path mode's semantics (`python -m xgraph.programs execute --matched`): over the KG index and served tokenizer path mode ran with, the path executes as a committed path does in the engine (labels read as label index rows, a relation label or alias followed as the union of its relations, the start committed in best-connected order, frontiers capped at the engine's limit in its order, read through the record's KG version), and the answer is the best connected of the entities the answer span may name, since which one the model would write cannot be known offline. Its offline repairs, which the engine never makes, are options: a relation the frontier lacks replaced by its most similar one (`--repair`), and a start that names nothing as written read without the quotation marks wrapping it (`--strip-quotes`, quote stripping). Each record keeps its plan as the executor read it, labels of the KG index, for engine delivery.

**Answer delivery**:
How an executed plan's answer is chosen among the entities its frontier lets the answer span name: executor delivery takes matched execution's answer, the first in best-connected order; engine delivery replays the plan as a replayed prefill in path mode, and the model writes the answer inside that frontier (`python -m xgraph.repair deliver`: a repaired plan, with `--all` every selected plan, with `--executed` the plans matched execution read from free paths). Comparing planners, such as path mode and matched execution of paths written without the engine, under one answer delivery keeps planning apart from answer selection.

**Execute-and-resume**:
A black-box interface (`python -m xgraph.generate --arm resume`, `xgraph.resume`) that writes a free opened path, executes it by matched execution, and at the first unavailable relation freezes the successful prefix. Text feedback shows that prefix, what its live frontier reached and a page of its writable relations in the request's KG version. The model writes only the remaining suffix, or `⟦⟧` to close at the prefix; `Page: N` requests a later relation page. Every continuation or page request consumes one of `--resume-max` extra calls (default three). No logits or engine mask enter planning. Only an executable final plan with at least one relation delivers an answer, by executor delivery or the existing separate engine replay.

**Derivations**:
For an entity of a path's final frontier, the number of distinct relation paths that reach it from the start the path resolves from, one relation and one entity of each step's frontier per step (`MatchedExecutor.derivations`). A label naming two relations that both link the same entities counts twice. It is unrelated to derivation mode.

**Endpoint witness**:
One start-to-endpoint chain of entity ids from matched execution
(`MatchedExecutor.execute_set(..., witnesses=True)`), rendered with entity labels
for a plan audit. It follows the same KG version, anchor, start commitment,
selections and caps as the endpoint's answer support. Among surviving chains it
chooses lexicographically by descending sitelink count, then numeric entity id,
at each step from the start. Witness selection needs indexed sitelink counts and
does not change endpoint ordering or answer support.

**Multiplicity delivery**:
An answer delivery that re-executes a path-mode answer's written path by matched execution, in the record's KG version and with the run's start rank and unquoted starts, and delivers the frontier entity with the most derivations where the frontier has two or more entities and that count is above one; among tied entities it keeps the model's answer if it is one of them, else delivers the first in the sitelink rank's best-connected order (degree order for a KG index without sitelink counts), whatever start rank the run used. Otherwise the model's answer stays (`python -m xgraph.programs deliver --rule multiplicity`; each record's `delivery` says `multiplicity` or `model`).

**Residual-goal STOP**:
A rule that closes a path after a completed relation when the question's answer head (the explicit word extracted by `repair.answer_head`, excluding who, whom, whose and where) is a whole word of that relation's indexed labels or aliases, plural-stripped, and no question content stem still unused by consumed relations matches a relation offered by the new whole frontier. Content stems use `repair.content_stems`: remove the written start's words from question content, accumulate consumed relations' indexed labels and aliases, and remove the head's stem. It needs neither a singleton nor a type check, and does not append relations. Its CPU replay (`python -m xgraph.stopping`) stops at the first qualifying proper prefix under matched execution and records the decision per record. For a truncation it retains the model's answer if its label names an entity of the committed prefix frontier; otherwise it uses multiplicity delivery, then matched execution's best-connected answer. This is offline answer delivery, not a prediction of the model's new answer after online stopping.

In decoding (`XGRAPH_STOP=residual-goal`, off by default), committed paths apply the same rule at each relation boundary, offering only the closer among delimiter branches. A longer relation label sharing the completed label's prefix stays writable until its own boundary. The request supplies its question as `xgraph_question`; the host compiles its STOP relations, and the engine updates consumed words in one pass. Comparison spans keep their expression rules.

**STOP relations**:
A request's relation ids that name its answer head and those that cover relevant question words, compiled on the host from the question and indexed relation labels and aliases (`stopping.RelationTerms.for_question`). Each relevant raw question word has a bit: consumed relations cover every word sharing their stems, while the written start removes only exact words. The residual relation set shrinks as those words are covered. Words no indexed relation can cover need no bit, since they cannot block STOP.
The shared vocabulary indexes words and stems to relation ids once; question compilation visits only matching entries. Host coverage stores only relations covering a question word; an absent relation has zero coverage. The engine's dense per-slot tables keep the same layout.

**Engine admission time**:
In `python -m xgraph.bench`, the wall time to reset and begin a replay's slots and install their KG version, STOP relations and forced opener, including device completion (`setup_ms_median`, and divided by batch size `setup_ms_per_request`). It excludes engine construction and history/logits allocation. It is separate from the per-step CUDA event times for sync and masking, and does not measure the model server's request scheduling or prefill.

**Replenished replay**:
A fixed-horizon engine assay that replaces a completed request in its slot on the
next step, with fresh state and its own pinned version. All slots stay active;
partially replayed requests at the horizon retain their measured costs.

**Support-set churn**:
Changes in feasible label continuations when a replay slot receives a replacement
request. It describes a workload's changing supports, independently of whether
those supports hit a warm compilation cache.

**Plan repair**:
Alternatives to a path-mode answer's plan, built without gold and executed by matched execution (`python -m xgraph.repair`): its proper prefixes (stopping earlier), single-relation deletions, and, bounded, single substitutions and insertions of relations its frontier offers at that point, those sharing most words with the question first; candidates that reach nothing are dropped, the original kept. A verifier chooses one from the question and the candidates' relation sequences alone, without the entities they reach (or, as separate arms, with their number and coarse class), and the chosen candidate's delivered answer replaces path mode's; choosing the original, or an unreadable reply, keeps path mode's own answer. The delivered answer is matched execution's, or, as a separate arm, the model's: the chosen plan as a replayed prefill, after which the model writes the answer inside its frontier.

**Shortlist**:
The plans plan repair's verifier is shown instead of every candidate (`python -m xgraph.repair requests --shortlist K`): the original and the K - 1 other candidates ranked highest without gold, by how much of the question's content their relation labels cover, whether the question's answer head (the word after a leading which or what, such as `capital`) ends their last relation label, whether their coarse class suits that head, less the label words the question lacks and their difference in length from the original. They are numbered in a seeded order that hides which is the original, and the verifier may reply 0 to keep the plan written first (the keep decision); a reply that names no plan also keeps it. `python -m xgraph.repair oracle --shortlist K` reports, on MINTQA, how often a choice within the shortlist can be right and how many recoverable errors (path mode wrong, some alternative right) it keeps.

**Plan audit**:
An opt-in two-call MINTQA diagnostic (`xgraph.mintqa_rematch.audit_requests`, queue
`--audit`): a generated path and answer followed by one fresh masked call shown the
executed plan and endpoint display, with lexically ranked terminal and penultimate
relation menus (`audit`) or endpoints alone (`audit-nomenu`). Its instruction
permits keeping the plan, selecting an endpoint or one terminal relation edit;
the existing support/revise policy accepts usable second answers and charges
both calls, retaining the first answer on failure. Menu and observed first calls
are read with frontier selections applied to their endpoints and menus. The
assistant history keeps these selections while omitting forced displays; relation
plans contain only relation labels, with selections recorded by boundary.
Edited repair alternatives are fresh relation plans without inherited selections.
Edit bounds are instructions,
not additional engine constraints.

**Executed alternative**:
A plan repair candidate re-executed by matched execution in the request's KG
version before display. It retains its endpoint count, displayed endpoints and
execution cap events. The witness plan audit ranks these candidates as two-pass
revision does (`repair.shortlisted`), without using their cached answers.

**Global-menu plan audit**:
The opt-in prompt control in `mintqa_rematch.audit_requests` (`--menu-source global`):
each terminal and penultimate menu has the corresponding local menu's displayed
relation count, but ranks all writable KG relation ids against the question.
Alias IDF overlap determines the score, with zero frequency and numeric-id ties.
Explicit `--menu-source local` supplies its matched local control. Both use the
same explanation and local frontier count/truncation metadata. These menus are
prompt content; decoding retains frontier KG execution and bound answers.
The legacy plan audit is unchanged when `--menu-source` is absent.

**Witness plan audit**:
The opt-in `audit-witness` plan audit (`mintqa_rematch.audit_witness_requests`):
endpoint witnesses for the executed first path and a bounded set of executed
alternatives, together with `audit`'s terminal and penultimate relation menus.
The model may also adopt a displayed alternative, writing one complete path from
the original start and its bound Answer span. The second call's ordinary path
mask and the support/revise acceptance and charging policy are unchanged.
The queue selects this arm through `--audit-arms audit-witness`.

**Selective verification**:
Plan repair that asks its verifier only when a frozen rule over pre-verification features fires (`python -m xgraph.repair requests --trigger-rule`, with `--family` selecting its rule). Otherwise the original answer stays. A lexical margin is the highest alternative shortlist score minus the original's score within the actual verifier menu; it is unavailable when there is no alternative. Rules use one or two thresholds on menu structure, lexical margin, plan length, frontier sizes, or an explicitly recorded path log-probability. Endpoint identities, answers, verifier choices, and gold are not features.

An offline evaluation (`python -m xgraph.repair selective`) cross-fits the rule over two halves of relation-sequence clusters: each half is evaluated with the rule fitted on the other. Gold supplies training outcomes and scoring, never inference features. The all-development refit written separately is the frozen rule for a subsequent confirmation; its training result is not the cross-fitted estimate. Extra calls count verification and, for engine delivery, answer-delivery prefills.

**Walk**:
A baseline in which the model chooses, round by round, a relation of the current entities from a numbered menu of all of them, and the KG store follows it (`python -m xgraph.generate --arm walk`, `xgraph.walk`), read through the request's KG version where it names one; the answer is the best-connected entity reached.
The opt-in minimum hop control (`--min-hops`, default zero) withholds the stop choice until that many relations have been followed; refused early stops consume decision rounds, and an unmet minimum delivers no answer.
With `--matched-paths`, the walk uses the indexed live frontier and every writable relation label and alias, keeps all surviving start senses until committed answer delivery, and applies the selected start rank and unquoted starts. This is the same-state menu control.

**Variable-stride action**:
An opt-in matched walk choice (`--walk-stride 2`) that selects one executable
two-relation composition as a single ordinal action alongside atomic relations.
The composition expands into ordinary physical relations before execution: each
hop retains its frontier cap, KG version and start ranks, and both count against
the hop limit. The staged interface uses model calls for choices and the existing
replayed-prefill API for separate engine answer delivery. It adds no native
engine mask or slot state.

**Atomic lookahead control**:
The matched walk with `--walk-stride 1`: it displays the same executable
compositions and reached entities as variable-stride actions, while accepting
only atomic choices. Ranking, graph state and answer delivery are shared; the
second relation requires another decision.

**Variable-stride menu**:
The shared bounded ordinal display for variable-stride actions and the atomic
lookahead control: five WT-ranked atomic relation ids and up to five ranked
pairs drawn from five relations at each hop. Endpoint observations show all
entities through width eight, otherwise three in best-connected order.
`(+N more)` marks display omissions; complete capped execution frontiers remain
recorded. Undisplayed relations cannot be chosen through this ordinal interface.

**Completed variable-stride plan**:
A concrete executable plan with at least one relation whose planner ends with
`stop`, `terminal` or `hop_limit`, without a planner failure. Accuracy conditional
on these plans is a diagnostic; answer delivery failures stay in its denominator.
A concrete prefix delivered after `invalid_choice` or `decision_limit` is outside
this diagnostic. The registered development gate still scores every question
under its original failure policy (`eval/variable-stride-pilot-2026-10-29/`).

**Context refusal**:
A request whose complete chat prompt plus its completion allowance exceeds the
configured context limit is refused before model issuance. Variable-stride
records retain its token count and excess in `prompt_checks`, mark the question
with `error=context_limit`, and deliver no partial plan. Earlier model calls
remain charged; the question remains in its arm's scoring denominator.

**One-hop retrieval**:
A retrieval arm: the one-hop facts of a question's topic entities (named by its gold query, or linked in its text) listed in the prompt before it (`python -m xgraph.rag`, `--linked`).

**SubgraphRAG**:
A trained triple retriever followed by a frozen answer model. The in-repo CPU
scorer (`xgraph.subgraphrag.TripleScorer`) accepts cached text embeddings and
checkpoint tensors as data. Its published-retrieval reader replay is a separate
comparison: it cannot establish retriever quality or end-to-end retrieval cost.

**RLM_Wikidata**:
The released Wikidata Search Traces harness: a model batches graph reads in a
persistent Python namespace and delegates interpretation to graph-free model
subcalls. Its final-answer gate checks shape and previously read or supplied
identifiers. Preserving that loop alone does not qualify its richer graph-read
contract on the KG store. Its scoped qualification record is retained with the
study materials; no local generation arm is admitted.

**Directional distance encoding (DDE)**:
SubgraphRAG's topic/non-topic one-hot diffused by the mean over incoming triples
for successive rounds, then independently over reversed triples. Isolated rows
receive zeros in diffusion rounds. These are structural diffusion features,
rather than integer shortest-path distances (`xgraph.subgraphrag.directional_encoding`).

**Iterative retrieval**:
A multi-hop retrieval baseline on the KG store (`python -m xgraph.generate --arm iterative`, `xgraph.iterative`): round 0 is one-hop retrieval; then the model answers or names, as `Next: ⟦ label⟧`, the entity whose facts it needs next, and their one-hop facts follow, up to a number of lookups, read through the request's KG version where it names one. With a written start (`--written-start`), a short generation first names the entity the question starts from, and round 0 lists its facts before the topic entities'.

**Written start**:
The entity a question starts from as the model names it in one short generation before iterative retrieval (`--written-start`, `xgraph.iterative.start_messages`), whose facts open round 0 ahead of the linked topic entities'; its tokens count towards the record's.

**Matched demonstrations**:
Few-shot examples for a retrieval arm that are path mode's: the same questions with the same numbers of hops, without a word on escapes (`python -m xgraph.rag --matched-demos`; iterative retrieval's always are), so that a comparison with path mode does not also compare demonstrations. The KG agent's (`--matched-demos`) are each path-mode request's own demonstrations as tool trajectories: the question, an `execute_path` call with the demonstration's path, its answer as the result, and the answer span.

**KG agent**:
A baseline in which the model answers by calling tools over the KG store, read through the request's KG version where it names one, within a call budget (`python -m xgraph.generate --arm agent`, `xgraph.agent`): entity search by label, an entity's relations, following one relation, and executing a whole path (`execute_path`, parse-then-execute as a tool); then it writes its answer as an answer span. Options bring it level with path mode: matched demonstrations, `execute_path` as matched execution (`--matched-paths`), and pages of results after an offset (`--pages`).

**Two-call planner/executor**:
An unconstrained complete relation-path call, matched CPU execution, then a
revision call with WT's endpoint feedback and executed alternatives
(`xgraph.planner_executor`). It uses saved WT-D ranks for static starting menus,
has no live within-path observations or selections, and delivers the committed
capped endpoint set. The first answer uses executor ranking; a usable second
answer must name a reached endpoint. Both calls remain charged. This is a
complete-policy control, not a pure masking ablation.

**Agent tool budget**:
The maximum external tool calls a KG agent may consume (`--max-calls`), including
failed and invalid calls. Model rounds for cap handling and final answer delivery
also consume model calls and tokens. A record reaches the tool cap when its
`tool_calls` equals the declared budget; the separate `budget_spent` flag records
the agent's cap-handling state. Budget diagnostics retain both counts.

**Agent context limit**:
A per-turn admission check using the served tokenizer and chat template. A prompt
plus that turn's completion budget above the model context ends only that episode
with `finish_reason=context_limit`, `error=context_limit`, `declined=true` and no answer.
Equality fits. Previously spent model calls,
tokens and tool calls remain charged, and the question remains in the scoring
denominator. History and the turn's generation allowance are never shortened
(`xgraph.agent.run_agent`). Also called the context limit.

**Agent answer parser**:
The versioned reading of an agent's final physical reply (`xgraph.agent_answers`).
`agent-answer-v1` is the frozen `xgraph.prompts.parse_answer_set` behavior and the
default for records without `answer_parser_version`. The opt-in
`--agent-answer-parser agent-answer-v2` reads JSON string arrays, an `answers`
object containing an array, or one label per line, including inside one outer
span. It removes span delimiters from each member, drops blank and abstaining
labels, deduplicates in order, and flattens one array level with an explicit
recovery status. It rejects bracketed unquoted comma lists without guessing where
labels end. Records keep the version, `raw_final_reply`, `answer_parse_status`,
`answers` and `legacy_answers`; versioned scoring reads the recorded version.

**Agent final-answer controller**:
The opt-in `--agent-final-controller agent-final-retry-v1` protocol
(`xgraph.agent_final`): a tool reply after the final-answer instruction causes
one request for a final answer only. Neither that tool reply nor a tool reply to
the retry is executed. Every physical model reply remains in `turns`, `rounds`
and token charges; context admission also applies to the retry. Records keep
`final_answer_controller_version` and `final_answer_retries`. This switch is
independent of the answer parser and is off by default.

**Failure-aware agent**:
An opt-in KG agent (`--agent-feedback`) with matched demonstrations, matched execution and pages. A failed whole-path call reports its failed hop and a page of writable relation labels and aliases from the whole frontier just before it. The agent must make another valid tool call before answering while calls remain; a refused answer costs a model turn without spending a tool call. After two refused answers the model's own answer stands, with `finish_reason=retry_refused`, so the feedback adds retries but never removes an answer.

**Masked KG agent**:
A KG agent whose path-call arguments are path spans (`--agent-mask`): `Action: execute_path ⟦ Start · relation⟧`, or `Action: follow ⟦ Start · relation⟧` for one relation. Each round has fresh execution state and reads its request's KG version. The relation support ladder applies inside these spans; JSON search and relation lookups remain free. Matched demonstrations, execution and pages are enabled. The agent delivers its own answer, and records its executed plans for separate delivery comparisons. `unmasked_calls` counts path calls that bypassed the span mask, including JSON calls.

**Forced path call**:
The agent's first assistant turn prefilled with `Action: execute_path ⟦` (`--agent-force-path`). A masked agent replays it through `xgraph_prefill`; an unmasked agent continues the identical prompt freely. Later rounds have no forced prefill. It composes with failure feedback.

**Agent schema control**:
A KG agent using standard structured outputs for its JSON actions (`--agent-schema static|turn`). The static schema restricts relation arguments to an enum of writable indexed relation labels and aliases. The per-turn schema recomputes that enum from the outgoing relations of all entities reached so far, through the request's KG version, with the vocabulary as its bootstrap before any entity is known. Its enum remains fixed within each path call. Both use the ordinary matched JSON prompt and leave the agent's answer free; native vLLM structural tags constrain the JSON after `Action:`. `schema_bypass_calls` counts path calls written outside that JSON format.

**Failed hop**:
The first relation position at which a whole path reaches no entities (`failed_hop`, one-based); zero when no start can be bound and null for invalid arguments. Its `frontier_relations` page comes from the KG index under the request's version, anchors, start ranking and frontier cap, including every surviving start rank. It names labels path mode admits, excluding unwritable edges and excluded labels.

**Path retry**:
A valid `execute_path` call after a failed one, possibly with lookups between them. A failure-aware agent records the failed action's zero-based index as `retry_of` and whether the written start or relation sequence changed as `plan_changed`; whitespace and page offsets are ignored. Per-question `retries` counts these calls and `retry_changed_plan` says whether any changed the written plan. Looking up relations can satisfy the required next call without being a path retry.

**Successful-prefix retention**:
A later-hop retry keeps the failed call's written start and every relation before
its failed hop. The diagnostic in `xgraph.webqsp_diagnostics` pairs consecutive
execute_path actions, allowing intervening lookups; aliases remain different
spellings. It describes saved actions, not hidden model memory.

**Known-prefix share**:
Among retries retaining a successful prefix, the ratio of already successful
physical relation positions to all requested relation positions in those retries.
It includes the requested suffix even when execution fails before reaching it.

**Free-cap token sensitivity**:
All WT prompt plus generated tokens divided by its fixed reference agent's tokens
after making every agent question reaching the tool-call cap free. It retains
the whole cohort and all WT cost; it is a bookkeeping sensitivity, not a policy
or GPU-time measure.

**Fixed-program selection effect**:
The change in gold-compatible capped endpoint support after removing entity
selections from a saved program, with its start, relation interpretation, KG
version, anchors, ranking and execution caps held fixed. Support gains and losses
are both possible; the replay does not predict a new model answer.

**Final relation-plan exactness**:
Whether a record's final relation sequence equals its question's gold chain relations (`python -m xgraph.planning exact`). For a KG agent this is the last readable whole-path call, otherwise its follow calls in order; malformed whole-path calls are ignored. Relation labels and aliases map to ids, with whitespace and string-form relation lists read as the tool reads them. Missing outputs remain in the declared panel's denominator. It does not score the start, execution success or answer accuracy, and does not reconstruct a branched search from individual follow calls.

**Chain-of-Relations (CoR)**:
A comparator that ranks relations and searches their chains depth first (`python -m xgraph.generate --arm cor`, `xgraph.cor`). The default upstream mode loads CoR prompts and uses the released stack and separate reasoning decisions: stop, filter, forward, or backtrack. It starts from the first supplied topic entity, traverses both directions over the request's KG store or KG version, and keeps intermediate entities implicit. Empty filtering ends the search; it and exhausted search fall back to direct generation. An optional `--max-calls` caps every model call, reserving the last for fallback. `--cor-mode reconstructed` retains the historical outgoing-only adaptation. Backend and answer-delivery deviations are in `docs/reproducing.md`.

**CoR prompts**:
The released CoR YAML prompt file, supplied locally with `--cor-prompts`, read only as data and never distributed by xgraph. `xgraph.cor.CorPrompts` validates its blocks and placeholders; generation provenance and records keep the file's SHA-256. Prompt text stays outside the repository because the upstream repository provides no license. The upstream exploration and reasoning temperatures are exposed as `--cor-temperature-exploration` and `--cor-temperature-reasoning`: both 0.01 selects the paper setting; defaults retain the released runner's 0.3 and 0.1. `cor_config` records the depth and temperature profile per question. For MINTQA's four-hop transfer, explicitly select `--cor-depth 4`.

**CoR parse repair**:
Reading a reconstructed CoR model reply in another format than its prompted JSON object or answer line: a JSON string or list, plain labels or ids, or an answer with repeated span delimiters. Records keep the original reply in `turns` and `replies`; a `parse_repair` trace event names its stage, zero-based turn, repairs and parsed selection. Repairs issue no model calls and keep relation choices within the offered menu and filter choices within the candidate set. These shared parsers also serve GCR; upstream CoR uses the released response keys instead.

**Graph-constrained Reasoning (GCR)**:
A comparator that enumerates bounded entity-relation paths from a question's starts, constrains a model to generate paths with a KG-Trie, then asks a general model to synthesize the answer from those paths and their hypothesis answers (`python -m xgraph.generate --arm gcr`, `xgraph.gcr`). By default it samples `--gcr-k` independent trajectories; `--gcr-decoding beam` selects joint GCR beams instead. Every issued scoring request, trajectory and model stage is charged. A separate synthesis model runs in a fresh process using a GCR path handoff. Released specialized weights give a native-method transfer; frozen general weights in both stages give a shared-condition adaptation. Its prompt profiles and deviations belong to the separate study records; neither variant establishes published-score reproduction.

**GCR path handoff**:
A sealed JSON artifact containing the planner's model-free state, complete path
and hypothesis evidence, usage and literal synthesis prompts (`xgraph.gcr_stages`).
Its binding covers inputs, settings and stage sources. A fresh synthesis process
reads it without constructing the planner. A completed, receipted path stage can
be reused on resume while failed attempts remain charged to the lifetime ledger.

**Load-progress watchdog**:
A startup silence deadline extended by every new log byte, including repeated
text and carriage-return progress (`xgraph.load_progress`). Readiness disables
it during generation. A timeout stops the launcher and removes its owned
container before recording its end and settling the elapsed charge. Unconfirmed
cleanup retains the reservation.

**Total startup bound**:
A separate limit from the first engine-start marker to readiness. Log activity
and repeated start markers cannot extend it. Each new engine gets a fresh bound;
the stage and lifetime allowances still apply independently.

**Strict free-memory admission**:
A launcher check that requires MemFree, rather than only MemAvailable, to cover
the stage's declared memory need immediately before starting its container
(`XG_REQUIRE_MEMFREE` in the study launcher). On unified memory it can
drop cached file pages under explicit roots and wait; it refuses the launch if
free memory remains short. Dropping cached pages does not change file contents.

**Headroom waiter**:
A process executing the study's canonical GPU headroom launcher from its source
repository or one of its registered worktrees. Interpreter execution uses argv[1], direct execution
argv[0], with relative paths resolved from the process's working directory and
symlinks resolved. Readers, shell command strings and fixture copies do not count
(`xgraph.gpu_idle`).

**Engine-init infrastructure retry**:
One automatic GCR process-stage retry for a CUDA out-of-memory failure proven to
precede engine readiness and requests. The failed attempt, partial provenance
and retry remain receipted; both launches and the retry's cache recovery consume
the same stage and lifetime allowances. An uncertain cleanup blocks recovery.

**Think-on-Graph (ToG)**:
A published reader that prunes relations for each frontier entity, scores their adjacent entities, keeps the global highest relation-score times entity-score products and reasons after each layer (`xgraph.tog`, `xgraph.generate --arm tog`). Its official Wikidata prompts are external literal data; native depth and the four-hop transfer are separate queue arms. It selects explicit intermediate entities, unlike CoR. Source and adaptation details are retained with the study materials.

**Relation-based Think-on-Graph (ToG-R)**:
ToG's relation-only variant (`--arm tog --tog-variant tog-r`): globally prune
relations, search every adjacent entity, reason over complete endpoint sets, then
randomly retain entity/path candidates for the next iteration. Distinct relation
chains sharing an endpoint remain separate beam candidates. It issues no entity
scoring calls. The local transfer uses a reconstructed relation-chain evidence
format with the official Wikidata reasoning demonstrations; its qualification
and deviations are retained with the study materials.

**Decoding on Graphs (DoG)**:
A published reader that alternates graph-constrained triples with unconstrained explanations over a retrieved subgraph (`xgraph.dog`, `xgraph.generate --arm dog`). A next triple touches either endpoint of any visited triple or question entity. Token beams search each triple, and chain beams sum its triple scores. The MINTQA transfer uses a gold-free BM25 prefilter per BFS hop followed by fp32 Stella reranking, with shortest connector triples within a top-k budget, retaining every question for scoring. Full Stella retrieval and the historical BM25 fallback remain explicit alternatives. Released WebQSP graphs need no embedding retrieval. Candidate overflow fails preparation unless diagnostic truncation is explicitly selected on the historical path; unknown coverage stays null. It is a qualified transfer rather than published-score reproduction.

**DoG trie key**:
The serialized question id and triple step identifying one immutable sampling
trie. A later step uses another key because Model Runner V2 retains a finished
slot's processor binding until that slot is reused. This identity affects no
prompt, sampling setting or recorded answer.

**DoG oracle-retrieval upper bound (dog-gold)**:
The same DoG reader, beams and decoding caps, given the union of its frozen
retrieved graph and the mapped gold chain triples (`xgraph.dog gold`,
`xgraph.generate --arm dog-gold`). All retrieved triples keep their order; only
missing gold triples are appended, so the union may exceed retrieval's top-k.
Unmapped questions keep the retrieved graph and remain in the denominator, with
their count reported. Linked topics are unchanged. This is an oracle-retrieval
upper bound, never DoG's score; outputs, provenance and paired reports label it.

**DoG answer delivery**:
The Answer span extracted from DoG's raw answer call. MINTQA scoring accepts a
saved `answered` delivery when reparsing a complete answer line in that call reproduces the span, including
when trailing raw text reaches its token cap. The raw stop reason and usage stay
recorded; explicit errors, escapes and truncation flags still fail.

**DoG trie key**:
The serialized question id and triple step identifying one immutable sampling
trie. A later step uses another key because Model Runner V2 retains a finished
slot's processor binding until that slot is reused. This identity affects no
prompt, sampling setting or recorded answer.

**MINTQA baseline pilot**:
A seeded subset of the frozen development panel, equal in size per hop. Selection maximises questions retained in whole relation-sequence clusters, splitting at most one cluster per hop when an exact fit is impossible. Parent weights are renormalised to each full-panel hop's mass. Comparisons pair each arm with cached `agent-strong-8` on these ids only. The bootstrap conditions on the selected clusters; this development diagnostic does not establish population or confirmatory evidence. Serial question times or aggregate batched wall times project the full-panel workload under the pilot's recorded scheduling conditions; batched projection requires the same hop mixture.

**Baseline validation subpanel**:
A gold-free, seeded subset of the frozen MINTQA internal validation panel used to
compare a development-selected published reader with WT and the strongest agent
(`xgraph.mintqa_baseline_validation`). It retains parent relative weights within
hops and favours whole relation sequences. Its descriptive paired intervals
condition on the selected clusters. Local owner exports supply both WT calls and
the agent after freeze; their complete costs remain charged.

**GCR screening**:
A fixed, seeded subset of complete calibration cases, shared across model families,
on which GCR's joint beam must pass an accuracy and projected runtime gate before
becoming a full calibration A candidate. Cases
are selected by seeded case-id hash within relation-sequence quotas. The frozen
eligibility decision precedes full-panel GCR results.

**GCR path**:
An explicit-entity path serialized as `<PATH> Start -> relation -> Middle -> relation -> End </PATH>` (`xgraph.path_trie`), with canonical labels, every intermediate entity, and one space inside each marker. It includes a nonzero number of outgoing relations up to `--gcr-hops`; bounded cycles are allowed. This serialization is separate from xgraph's relation-only path span. The released native tokenizer has special tokens for the markers; the entire padded string is tokenized for the KG-Trie.

**Paper GCR prompt profile**:
The zero-shot planner and inductive reader prompts from the GCR paper's Figures 10 and 11, selected by `--gcr-prompts` pointing to the released prompt source. Class-level string literals are read without executing upstream code. It requires linked single-answer requests, sends one user message per stage and records the source hash. The historical default retains reconstructed prompts. Transfer qualification covers its forced initial path, header formatting, beam backend and other adaptation details.

**GCR pilot admission**:
The CPU check (`python -m xgraph.gcr prepare`) that binds a proposed pilot to basepilot2's members, weights, questions and linked starts, then counts every directed labelled walk through four hops by endpoint multiplicity. Counts include bounded cycles and parallel relations. An oversized exact trie refuses model launch; it never selects a smaller graph or drops a question. `python -m xgraph.gcr depth-report` separately audits the actual returned path depths after generation.

**Native GCR qualification**:
Hash-bound evidence for a frozen trained-planner transfer and its synthesis
readers (`xgraph.webqsp_gcr_qualification`). Fixed-score CPU checks compare the
hf-group decoder with the release-pinned Hugging Face reference; CPU prompt
checks cover each reader. Real model evidence must then verify complete paths,
tokenization, scores, synthesis and usage under the same recipe in both families.
CPU preparation stays pending until that evidence exists. Qualification does
not establish published-score reproduction, answer accuracy or a measured price.
Continuation evidence follows sealed stage receipts; prompt content hashes and
the recorded serving chat content format bind synthesis to the frozen recipe.

**GCR parse repair**:
Reading a generated start or synthesis reply through CoR's shared parsers (`xgraph.cor.parse_selection`, `parse_fallback`). Starts accept JSON, labels, ids, lists, and the first entity in a native reasoning-path header or path marker. Synthesis delivers the first readable answer, repairing repeated or missing span closers. With a request's `answer_mode: set`, synthesis instead reads every answer from a JSON array or final answer lines, keeping commas inside names. A `parse_repair` trace event records the stage, zero-based turn, repairs and parsed value; raw replies and all model usage remain in the record. Linked topics or anchors bypass start generation.

**KG-Trie**:
GCR's token trie of the complete serialized GCR paths available to one request under its starts and KG version. It uses the served tokenizer, including both path markers, and admits exactly the next tokens of those paths. For sampling, the vLLM processor (`xgraph.path_trie:GCRLogitsProcessor`) receives each request's serialized trie through the per-request argument `gcr`; it forces one initial GCR path and removes the constraint after its closer. Host beam search uses the same trie's allowed sets over model logprobs. Empty-trie policy `open` permits unconstrained generation, following upstream's fail-open behavior; `closed` returns an explicit unanswered record without generating a path.

**GCR beam**:
A token prefix in width-`--gcr-k` joint path-and-hypothesis search (`--gcr-decoding beam`). The KG-Trie constrains the path; after its closer, hypothesis tokens are unconstrained until any served tokenizer or model generation-config EOS or the token bound. With default `--gcr-beam-semantics vllm`, the score sums raw generated-token logprobs, including EOS, with pinned vLLM's default length penalty of one: divide by original prompt plus generated tokens, excluding final EOS. Exact ties use lexicographic generated-token-id order. Separate bounded pools retain live and EOS-completed sequences. Several beams may share a path and differ in their hypotheses. Discarded live beams reaching the token bound remain an audit flag; only returned truncated evidence or an unfinished reader marks a delivered record truncated. Opt-in `--gcr-beam-semantics hf` uses ordinary Transformers beam candidate admission, generated-length scoring including EOS, and its configured stopping rule; it preserves duplicate hypotheses and selected incomplete prefixes at the token limit. Opt-in `--gcr-beam-semantics hf-group` matches the released size-one group search: K groups, diversity penalty one, per-group EOS hypotheses and stopping, with duplicates preserved. Its `beam_score` includes diversity; `joint_logprob` remains raw, and `search_logprob` records the penalized sum. The host decoder, score formula and costs are described in `docs/reproducing.md`.

**GCR path beam**:
The historical path-only comparison variant (`xgraph.gcr.beam_search_paths`), retained for fixed-score tests only. It sums raw path logprobs without normalization and finalizes at the closer. No CLI decoding mode selects it.

**GCR beam audit**:
The charged `beam_step` turns and raw token logprobs that accompany a GCR beam.
An HF tie fallback uses a full vocabulary row for selection, then retains only
raw-score witnesses for the surviving live and EOS extensions, configured EOS
ids and the provider's sampled token. `full_logprob_count` records the original
row size. Selection still reads the complete row; audit compaction changes no
beam score, stopping rule, returned sequence or charged request.

**GCR completion receipt**:
A benchmark variant's completion record binding its frozen manifest hash,
source commit, launch argv and recorded environment to the hashes of its output,
generation provenance, summary, timing and container event inventory. Reuse and timing comparisons verify
the receipt and charged usage; historical controls additionally pin every
artifact and their recorded launch configuration in the rerun manifest.

**GCR reservation**:
The benchmark's registered maximum container occupancy across attempts, including
failed attempts and model startup. A persistent ledger retains charges across
owner relaunches, including conservative historical wall charges where container
boundaries were not recorded. Uncertain occupancy or unverified owned-container
cleanup keeps the reservation active and blocks another launch.

**Stage wall deadline**:
The elapsed-time bound for one benchmark attempt, with time reserved inside it
for process termination and verified owned-container cleanup.

**Progress deadline**:
The maximum interval without advancing GCR scoring diagnostics, starting at the
first scoring diagnostic after admission and model startup. Ordinary logging
noise does not renew the deadline; the container occupancy allowance and stage
wall deadline always apply.

**GCR beam shortfall**:
A deficit in returned beam hypotheses, recorded by a `beam_shortfall` trace event and status. Default `vllm` counts distinct joint sequences with complete paths against `--gcr-k`, retains token-limited sequences with complete paths as explicitly truncated evidence, and never pads with duplicates. Both HF modes count the scorer-selected hypotheses against the requested return count, preserve duplicates and can retain incomplete path prefixes at the token limit. The record includes available token paths and whether live prefixes hit the token limit. If no path completes, status is `beam_no_path` and synthesis is skipped. An open empty trie explicitly uses the legacy trajectory decoder, recorded as `sampling_empty_trie`.

**Truncated KG-Trie**:
A GCR trie that retains only the first `--gcr-max-paths` graph paths in stable depth-first relation/object id order, with starts in QID order (`--gcr-path-overflow truncate`). Enumeration probes one additional path to establish overflow. Each record's `trie.truncated` distinguishes a cut from an exact fit; the generation summary counts cuts as `gcr.truncated_tries`. This is a resource cap, not a relevance filter. The default overflow policy remains `error`.

**Hypothesis answer**:
The unconstrained model suggestion after a GCR path's `</PATH>`. It may name entities outside that path and outside the KG. The synthesis stage consumes it alongside the path; it is not a grounded answer or an executor-selected endpoint.

**Walk observation**:
What the walk tells the model about the current entities each round (`--observe`): their labels (`labels`), their dominant class under instance of, else subclass of, and their number (`types`), that class alone (`type-only`), their number alone (`count`), nothing (`none`), a wrong class drawn by a seeded rule with the true number (`placebo-type`), the true class of another request's current entities in the same round with the true number (`shuffled-type`), the true number behind padding of as many tokens as the types observation (`filler`), or the types observation together with walk previews (`previews`), or with the same previews permuted among the menu's relations (`shuffled-previews`), or a preview control. The menu, the KG walked, and the answer stay the same, so only what the model plans from changes: an emulation of a path that is told what each relation reached without being told its value.

**Walk preview**:
What a relation on the walk's menu would reach from the current entities, shown beside it before the model chooses, without labels: their dominant class and their number, `place of birth → ⟨city⟩, 1 entity` (`--observe previews`).

**Preview control**:
A walk observation that holds everything of the walk previews but the candidate relations' classes fixed: the same menu rows, order, arrows, stop option, and true counts, with coarse classes in the type slot (`--observe` `slot-previews`, the true classes, which the others control; `neutral-previews`, no class; `count-shuffled-previews`, the true classes permuted among relations of one count stratum, seeded by `--observe-seed`; `static-previews`, each relation's static class). What a request is shown does not depend on the other requests of its batch.

**Type slot**:
Where a preview control shows a class: the coarse class in angle brackets, padded inside them to as many tokens of the served tokenizer as the longest coarse class has, so that the slot's length tells nothing of the class.

**Count stratum**:
The band a walk preview's number falls in, one entity, two to five, six to 64, or more (`xgraph.walk.STRATA`); `count-shuffled-previews` permutes classes only within one, so that a shown class never comes from a relation whose number is far from the one shown.

**Static class**:
A relation's coarse class from the KG alone: the coarse class most of its triples reach over the whole KG, each object's class read as the walk reads a class, `unknown` when it has none (`xgraph.walk.relation_classes`); computed from the KG store as it is, never through a KG version, it is what a static relation-to-class prior would show (`--observe static-previews`).

**Coarse class**:
A class of a short fixed list (`human`, `country`, `human settlement`, `film`, ...; `xgraph.walk.COARSE_CLASSES`) that the walk shows for a class with `--coarse-types` and in every preview control: the first of the list among the class and its superclasses under subclass of, nearest first, or `unknown`; it keeps a class as specific as an entity from naming which entities the walk reached.

**Decompose-then-execute replay**:
A baseline that executes a GWalk run's recorded decomposition again on the CPU, without a model call (`python -m xgraph.gwalk_replay`): from the subject GWalk extracted, each hop's relation template takes its object from GWalk's edit bank, the KG store, or GWalk's own answer for the hop, the first of them, in a chosen order, that has one. In the order bank, KG, answer it forecasts GWalk reading the KG (`python -m xgraph.gwalk --kg-db`); from the KG alone it is decompose-then-execute with GWalk's plans.

**Relation inventory**:
The relations a method can plan with. GWalk's is its relation catalogue, the relation templates it snaps a decomposed relation to: MQuAKE's 36 relations, or with an open relation inventory a template for every label and alias of each relation of the KG it reads (`python -m xgraph.gwalk --relations`). Path mode's is every relation of the KG index. A question whose chain uses a relation outside MQuAKE's inventory tests editing beyond it (`python -m xgraph.interventions select --outside-inventory`).

**Confirmatory test**:
An evaluation whose hypotheses, data, and analysis are pre-registered before any of its outputs exist, on questions no earlier work used. Experiments outside one are exploratory.

**Overlap registry**:
What each case of a confirmation set shares with data seen before it, written before any confirmation output (`python -m xgraph.overlap registry`): per named reference (a development release, MINTQA, a training corpus) the shared question, original and edited chain, entities, triples, exact edit triples, edited pairs, and relation sequence, and per prompt the demonstrated questions, entities, facts, and relation sequences. A stratum of it is the cases that share none of the overlaps it names, such as relation sequences unseen in CF-3k and MINTQA.

**Freebase MID**:
Freebase's entity identifier, such as `m.02mjmr`. RoG-WebQSP replaces many MIDs with entity names; the remaining opaque MID nodes include mediators. The WebQSP converter keeps a question-local identity map and uses numeric Q/P compatibility ids for the existing KG format. These numbers are not Wikidata ids, and names do not reconstruct native MID identity.

**CVT**:
A compound value type, or mediator, representing a multi-argument Freebase fact. The WebQSP bridge retains its node and both directed edges, like the released GCR graph builder. A path through a CVT takes two physical hops. Opaque MIDs remain writable by their MID, and are not all assumed to be CVTs.

**WebQSP set delivery**:
The matched executor's complete answer support after committing to the best surviving start, delivered as a set (`python -m xgraph.webqsp deliver`, `MatchedExecutor.execute_set`). Start, live-frontier and answer-label-row caps match the engine; each encounter is reported. Raw singleton delivery and an uncapped CPU replay of the fixed plan remain sensitivities. Removing caps offline cannot repair relation choices already made under a capped frontier.
The opt-in catalog declares `--complete-path-set`: intermediate entity selections
remain part of the plan, while a terminal selection ranks the complete committed
endpoint set rather than discarding its other members. This is a delivery variant.

**WebQSP policy catalog**:
The opt-in transfer matrix (`xgraph.webqsp_queue --catalog`): S, the committed
forced-opener path policy; blind5, with an additional predeclared synthetic
Freebase demonstration; menuobs, with relation suggestions and frontier
observations; the frozen `W`, `M2`, `MT`, `WT`, `B2` and `BT` policies; and
matched budgeted agents; native trained GCR with a family-specific reader; and
DoG on its released-graph secondary population. The validation pilot's DoG arm
uses an explicitly labelled BM25 timing proxy. ToG-R and CoR are declared blocked
comparators until their local Freebase adapters are qualified.
Offline execution X and shared-backbone GCR remain
in the separate bounded comparison outside this opt-in catalog.
Catalog preparation records the demonstrations and verifies tokenizer-bound
index reuse. Equal zero Freebase sitelink counts retain degree ordering.

**DoG retrieved-graph population**:
The separately labelled secondary WebQSP population from DoG's pinned retrieved
graphs, capped at the first 120 triples and filtered for at least one gold answer
appearing as a subject or object. `xgraph.webqsp dog-panel` verifies the released
file identity and question/gold alignment before building question components.
The study's additive importer accepts the declared-filter
population with its documented paper-count deviation, recording unrecoverable
paper membership explicitly. This is a test-phase population, outside the
validation pilot. An explicit `--secondary-pilot` is a separate test-phase cost
cohort with the released graphs, excluded from validation and policy selection.
Path mode reads those same graphs. This population cannot be reconstructed by
truncating RoG's unranked graphs or compared directly with its full population.

**RoG/GCR label protocol on released subgraphs**:
The bounded WebQSP comparison's primary protocol: GCR's released normalized containment and line scoring, `gcr_hit` and `gcr_f1`, on the questions supplied by RoG. Strict ranked Hits@1 and exact label-set F1 are secondary. Every arm requests a complete answer set; names cannot recover authoritative Freebase MIDs, so native-MID F1 is outside this comparison.

**Validation budget pilot**:
A seeded validation cohort stratified by the maximum reachable gold physical hop count (or unreachable) and answer-set size (`xgraph.webqsp_queue --pilot`). Every family and arm uses the same subset, reduced uniformly when the upper runtime assumptions including model startups exceed the pilot budget. The active registration prices measured per-arm container occupancy with event and ledger identities bound by the serving completion receipt; launcher wall remains separate. Legacy timings retain their declared launcher boundary. These are exploratory costs, not controlled serving latencies.

**Receipted pilot reuse**:
A validation stage's successful measured timing retained through an immutable
stage receipt in a registered continuation. It remains a price, while the
historical occupancy ledger already carries its charge. New stages add new
occupancy; historical failed attempts stay charged and never become prices.

**Post-pilot sealing procedure**:
The checked CPU sequence of measured sizing, execution preparation and owner
queue preview. Pending prices, qualification
or power block preparation. Its stat-only preview opens no pilot outputs.

**CPU address-space probe**:
A synthetic resource screen using a consumer source archive and replicas built
from permitted file sizes, numeric array headers and SQLite row counts. It records
the kernel's peak virtual address space and resident memory under an address-space
limit. It reads no sealed question, gold, graph payload
or generation output. Matching shapes alone does not establish a bound for every
graph topology or generated plan.

**WebQSP test registration**:
A prospective specification (`xgraph.webqsp_test`) sealing the shared policy,
ID-only sampling rule, development-gate conditions, inference and total resource
limit before test access. A separate execution seal binds the recorded gates,
validation-pilot timings, selected IDs, gold-free inputs and owner commands.
Complete stage receipts are required before gold or released outputs are scored.
A native single-completion record can supply its call count through the known
generator contract at the completion audit boundary, without editing its bytes.
Multi-call readers retain explicit counts and per-call token conservation.
A sealed GCR failure leaves only its comparisons INCOMPLETE; primary comparisons
still require their frozen strongest-agent reference.

**Question dependence cluster**:
A bootstrap unit derived from question-side fields alone (`xgraph.webqsp_test.question_clusters`).
Transitive topic overlap joins shared released topic entities and normalized duplicate question
text. The first-topic sensitivity groups by a question's own first released topic entity, retaining
released order without treating a secondary topic as a bridge; a question with no topic stands
alone. This is a deterministic annotation convention, not an authoritative primary-entity label.
The exact-question sensitivity joins only casefolded, whitespace-normalized duplicate text.
Clusters are formed on the population before restricting to the ID-selected sample.

**Post-seal accounting continuation**:
An explicitly authorized child execution of a sealed WebQSP test, implemented by
`xgraph.webqsp_continuation`. It binds the parent registration, execution, archived
ledger and owner authority, imports the parent's cumulative charge once, and uses
fresh output and source/environment paths. A normalized-science comparison retains
every scientific field and generation argument. Only new complete stage receipts
permit resume; the parent's charged orphan stays in its immutable history.

**Effective cluster count**:
The Kish concentration diagnostic `(sum(n_c))^2 / sum(n_c^2)` for cluster sizes `n_c`.
It describes size imbalance, not a measured outcome variance or an independent question count.
Power transported from another benchmark and perfect within-cluster dependence stress are
separate scenarios; the diagnostic alone identifies neither power nor joint assurance.

**Mandatory core cohort**:
The common ID prefix for the mandatory WebQSP arms in both families. Amendment 4
sizes WT, W, B2 and each family's fixed strongest agent from measured pilot prices,
counting shared physical stages once, and requires the registered transported
power screen. Published baselines use a separate panel. Earlier registrations
included both agents, native GCR and SubgraphRAG in core pricing; those immutable
registrations retain their original rules.

**Published-baseline subpanel**:
One seeded whole-cluster prefix drawn from the fixed WebQSP core for every
admitted expensive comparator (`xgraph.webqsp_test.cluster_subpanel`). Matched
validation prices determine its size within a separate post-freeze GPU budget
bucket, shared equally among admitted baseline groups. The most expensive group
sets the common panel size; groups unable to fund the registered minimum are
infeasible at that bound. It never reduces the core. DoG uses its released-population
intersection without replacement. Its comparisons and power screen are
secondary/descriptive, limiting the competitive claim to this panel.

**GPU budget bucket**:
A separately capped allocation of physical launcher elapsed time in the WebQSP
execution ledger. Amendment 4 records the core and published-baseline subpanel
buckets independently, including failed attempts. Qualification consumes the
core's internal reserve; baseline generation cannot borrow from that bucket.

**Output embargo**:
A sealed interval during which generated test answers cannot be opened, scored
or used to change a policy or sample. WebQSP's score CLI checks the embargo before
execution/output access. An earlier owner override records why and binds both
seals; its full record appears in the score report. Automatic queue integrity
and accounting checks do not disclose answers or compute correctness.

**Scoring replica**:
A CPU-only synthetic execution layout with invented questions, gold, graphs,
generation records and stage receipts. Its stage structure and cohort sizes
come from execution-seal metadata; the frozen consumer runs from a read-only
Git archive. It checks reducer arithmetic and resource readiness without
establishing real-output completeness, serving behavior or empirical accuracy.

**Continuation scoring supplement**:
An additive report after an accepted release for a post-seal accounting
continuation. It binds
the unchanged core score, records the authorized protocol deviation and parent
charges, and invokes the unchanged registered diagnostic reducers with an explicit
diagnostic sibling binding. It writes separate derived reports without changing
the generation registration, receipts or core score.

**Resource-child outcome handoff**:
An external dispatcher bound to the exact approved resource child's source and
seal identities. It preserves distinct generation and implementation commits,
validates an owner release before ordinary core scoring, and requires a bound
existing core score with the current physical GPU ledger before diagnostic reuse.
It is separate from the accounting continuation's scoring supplement and changes
no generation scope, retry permission, allowance or frozen reducer.
Its inherited cost-receipt scoring amendment reads each retained receipt from
its declared original custody path, authenticating the original execution seal
and exact job. It uses separate source-bound owner authority and invocation,
changes no statistical definition and creates no duplicate child receipt.

**Diagnostic sibling binding**:
An explicit invocation record connecting an unchanged additive diagnostic
registration to a descendant generation registration. It verifies the preserved
sibling declaration, ancestry and identical diagnostic-relevant scientific
fields. Source identities resolve to their registered Git exports; reducers and
both original registrations remain unchanged.
The accounting-recovery adapter additionally verifies the approved canonical
child's source, metadata, persistent claims and frozen science against that
exact parent before changing invocation identity. Its metadata-only owner
preflight does not score outputs or qualify real-output completeness.

**Freebase semantic hop**:
One forward entity relation after contracting each maximal run of authoritatively
typed internal CVTs to one edge (`xgraph.webqsp_test.semantic_hops`). Opaque MIDs
do not identify CVTs. Only complete simple forward-chain operations qualify;
filters, joins, inverse traversal, ordering and aggregates stay unsupported.
Missing types or chains and depth-ambiguous parses remain unclassified for the
secondary depth interaction, without changing the primary denominator. Physical
generation hops and token-segment depth retain their existing meanings.

**GCR beam semantics**:
The scoring, grouping, EOS, stopping and finalization rules used by a GCR beam
decoder. The released group beam uses one beam per diverse group; ordinary HF
beam and the historical vLLM host port are distinct choices. A timing-only
feasibility gate cannot choose among them using answer accuracy.

**GCR timing replacement**:
Using matched native validation launcher timings after choosing GCR beam
semantics. Amendment 4 prices GCR only in the published-baseline tier and
supersedes the earlier core anchoring rule. Earlier registrations replaced only
the two GCR rates while preserving the original priced core. Timings from another
benchmark remain feasibility evidence, not matched WebQSP prices.

**WebQSP-native GCR price**:
A complete receipted launcher measurement of the registered native hf-group
planner and family synthesis reader on fixed non-test RoG question components
(`xgraph.webqsp_gcr_pricing`). A CPU KG-Trie census precedes it. Amendment 6 uses
this price for full-cohort or seeded published-baseline subpanel admission;
an unaffordable or terminally unavailable comparison receives a cited published,
non-paired row. Missing measurements stay pending. A cross-dataset projection
cannot substitute for this price or exclude native WebQSP GCR.

**GCR feasibility exclusion**:
Prospective removal of both families' native GCR arms before the execution seal,
when hash-bound native bench evidence establishes that neither hf-group nor hf
meets the registered per-question bound. Missing evidence stays pending. Report
the exclusion with its bench evidence; a sealed arm's later failure is INCOMPLETE
for its comparisons and never changes the seal or primary reference. Amendment 6
supersedes the cross-dataset bench gate with the WebQSP-native GCR price.

**GCR cohort decision**:
A sealed application of the registered baseline allocation to a verified
WebQSP-native GCR price (`xgraph.webqsp_gcr_decision`). It binds its parent
registration, receipt chain and full execution-reserve charge. A cost exceeding
even the whole baseline bucket's minimum-panel bound establishes infeasibility
before other prices exist. This does not decide pending groups, resize the
mandatory core or create an execution seal. Amendment 6's descriptive wall
breakdown supplies no admission rate under that historical rule. Amendment 8
prospectively uses the receipted components for matched amortization.

**GCR matched amortization**:
The prospective admission projection in amendment 8: each family's complete
receipted launcher wall minus its measured generation workload is its fixed
launcher cost. Divide that fixed cost by the unchanged validation pilot cohort
size and add measured workload per question (`xgraph.webqsp_gcr_pricing.amortized_native_price`).
The basis matches the other optional baseline groups. The registered minimum
panel charges at least one complete fixed cost. This is a projection from
validation components, not measured full-panel runtime or steady-state latency.

**Funded launcher wall cap**:
The hard complete-wall allowance for one native GCR family launcher, including
cleanup (`launcher_wall_cap_seconds`). Amendment 8 divides GCR's equal baseline
group share between families in proportion to their frozen admission prices.
An overrun retains its elapsed charge and leaves that sealed arm INCOMPLETE;
there is no borrowing, retry or top-up. This allowance never changes either
GPU budget bucket.

**Endpoint contextual calibration**:
WT-PC's ranking of the unchanged WT endpoint set by smoothed conditional endpoint frequency,
optionally divided by the smoothed question-free prior frequency. With paired draws and fixed
support, its log score reduces to `log(n_q+1)-log(n_0+1)`. Its claim remains ranked Hits@1;
full-set hit and F1 cannot improve from reordering the same set.

**Ranked Hits@1**:
Whether the first predicted answer belongs to the gold answer set. WebQSP scoring reports it separately from any-hit. GCR's released Hit uses normalized substring containment across all prediction lines, and the native WebQSP evaluation script labels any-hit as Hits@1. Exact set F1 and those legacy measures remain separate (`python -m xgraph.webqsp score`).

**Gold-free component preparation**:
Building question components, requests, a KG store and served-tokenizer indexes
from only released question IDs, question text, question entities and graph
triples. It neither extracts answers nor computes coverage. The shared converter
uses `gold=False`; `xgraph.cwq` always uses this preparation contract. Scoring
extracts answer fields separately.

**CWQ validation pilot**:
The fixed hash-ordered validation cohort in `xgraph.cwq_queue`, shared by both
families and all admitted policies. It reuses the Freebase transfer recipes for
WT, W and B2 with each family's preselected strongest agent. Every generation
stage and CPU transformation has a timing and hash-bound receipt; its GPU allowance
includes container startup and unsuccessful launches, separately from stage wall
time and CPU transformation cost. Missing published validation
retrieval is recorded explicitly; a prospective exclusion narrows the test's
comparator claims rather than inventing a validation retrieval proxy.
Native single-call records retain their token IDs and use the registered stage's
one model call for accounting; agent rounds and composite arms keep explicit
call usage. Generation receipts include summary and provenance sidecars, and
validation scoring reconciles stage timings with receipts and the reservation ledger.

**Owned container**:
The container identified by the daemon-issued ID saved by CWQ's generation
launcher in `XG_CONTAINER_STATE`, outside its temporary wheel directory.
Timeout and TERM cleanup target only this ID. A successful daemon query must
confirm its absence before the reservation ledger closes the stage.

**Unresolved reservation charge**:
An active reservation-ledger entry retained when owned-container absence cannot
be established. Completed wall time is charged through `charged_through`, while
the active entry preserves the possibility of continuing device use. CWQ refuses
resume and scoring until the owner reconciles the interruption.

**CWQ dependence unit**:
The union-find component joining questions that share a CWQ source family or
CWQ primary topic. `xgraph.cwq_test.population_audits` chooses one unit for both
splits before hash sampling: if the union fails the registered concentration
limits on either complete validation or test question census, both use source
family alone. Concentration of the selected pilot and test sample still gates
admission; a sample cannot trigger another unit change. Selection uses question
metadata only and is sealed before outcomes.

**CWQ primary topic**:
The first entity in a released question's ordered `q_entity` list. Questions
without one join the CWQ dependence unit through their source family alone.
Other topic entities and normalized question text do not join the primary unit.

**CWQ source family**:
The released WebQSP ID prefix recognized by `xgraph.cwq_test.source_family`.
An unmatched ID is its own family. This is an operational ancestry proxy, not a
verified native parent annotation; shared ancestry remains in the claim boundary.

**CWQ dependence sensitivity**:
The former conservative unit joining all shared topic entities, normalized
duplicate question text and CWQ source families transitively. Its concentration,
clustered assurance and paired comparisons are disclosed alongside the primary
unit. Sensitivity results never gate admission or the primary NI conjunction.

**CWQ clustered assurance**:
The prospective two-family, zero-advantage joint NI simulation in
`xgraph.cwq_test.joint_assurance`. A Bayesian bootstrap on paired pilot cluster
means integrates variance uncertainty; discordance-pattern prior mass prevents
an all-agreement pilot implying known zero variance. Target cluster sizes and
cross-family covariance determine joint normal predictive draws. Admission uses
the simulation probability's Monte Carlo lower bound after concentration checks.
This is a model-based approximation to the paired cluster-bootstrap test, not a
population guarantee or a product of marginal powers.

**CWQ priority and exposure check**:
The admission check in `xgraph.cwq_test.admission_decision` that requires a complete
ID-only project history, retains ancestry overlap in the claim boundary, and
reserves the CWQ test cap only after protecting the full WebQSP core and the
admitted novelty evaluation. The evidence is a dated, hash-pinned owner budget
snapshot and exposure inventory, not a claim that code can prove history complete.

**CWQ test-registration draft**:
The prospective, unadmitted specification in `xgraph.cwq_test`: a complete
released test-ID population, gold-free hash sampling and question-component
clusters, joint ranked Hits@1 non-inferiority against preselected agents,
secondary set scores, cost sizing and prospective admission rules. Draft creation
cannot open answers. The CWQ readiness reader consumes complete pilot evidence
and the priority and exposure check before issuing an admission decision.
The registered scorer requires an admitted dated freeze and complete output
receipts before projecting test answers. An unfinished matrix is INCOMPLETE.

**Representable question**:
A question of an external benchmark whose gold chain the benchmark's KG holds unambiguously, by a rule fixed before any model run (for MINTQA, `python -m xgraph.mintqa_bench`: every hop's label triple maps to exactly one KG triple, and the triples connect; for 2WikiMultihopQA, `python -m xgraph.twowiki`: a compositional or inference question whose evidence is a chain to an entity answer, every hop held in the KG under exactly one property its relation name names). Scores are reported over the representable questions and over all of them, the others counted as wrong.

**Expression**:
A program written as one span that the KG executes (`python -m xgraph.expressions`): a path, `⟦ The Hobbit · author · place of birth⟧`, which gives the entities or dates it reaches; a date comparison, `⟦ earlier: Jaws | Star Wars · publication date⟧` (or `later:`), which walks each operand's via path (the relations after the operands) to dates and gives the operand whose date is earlier (later); or a value comparison, `⟦ same: Jaws | Alien · country of origin⟧`, which gives yes when the two operands' via paths reach a shared value and no when they reach values and share none. Its execution has a status that tells an empty walk apart from a name the KG lacks (unbound), an escape, two dates that overlap at their date precision (undetermined), and an ill-typed walk. In path mode with committed paths, the oracle also constrains date comparisons inside decoding (`Oracle(..., expressions=True)`, a date comparison span), whose result, the operand it gives or both when undetermined, binds the answer span after it, and value comparisons over an index with verdict entities, whose verdict entity binds it; the engine mirrors both (`Engine(..., expressions=True)`).

**Expression structure**:
The kind of expression a 2WikiMultihopQA question needs, by its type, answer, and evidence (`xgraph.expressions.structure`): an entity path, a date path (a path to a date answer), a date comparison, a yes/no comparison (a value comparison), a date difference ("Who lived longer"), or a count (more directors). The first four have a gold expression built from the dataset alone, anchored to its QIDs.

**Date precision**:
How much of a date is known: a year, a month, or a day. Truthy dumps store a year as its 1 January and a month as its first day, so a date of the KG takes the coarsest precision its value allows and covers the interval of that precision: 1979 is neither before nor after 15 June 1979, and 1814 stored as 1 January matches a gold "1814" but not "6 June 1814".

**Date store**:
The date objects of a KG directory with literal values in SQLite (`python -m xgraph.expressions dates`): each triple whose object is a date, with its year, month, and day. Beside the KG store of a KG with the same item triples, it stands in for a KG store of the literal KG.

**Executable-program coverage**:
The share of a benchmark's questions whose gold expression, executed over the KG store and the date store, gives the gold answer (`python -m xgraph.expressions profile`), per expression structure, anchored to the dataset's QIDs and as written with the KG's labels: what the KG can answer through the language, rather than whether the evidence triples are present.

**KoPL program**:
A KQA Pro question's native program: an ordered list of functions, text inputs, and dependencies naming earlier steps. Its entity-answer function is `What` (`xgraph.kqapro` also accepts `QueryName`). Qualifier operations refer to the particular fact that supplied a value or relation object.

**Operation coverage**:
The share of questions whose complete KoPL programs translate to paths or expressions (`python -m xgraph.kqapro profile`), decided without their answers or choices. Paths and the additional expressions are reported separately, with unsupported and unannotated questions retained. Executable-program coverage then checks whether the translated program reaches the supplied answer; reaching a gold endpoint does not justify dropping an unsupported filter or set operation.

**Relaxed chain coverage**:
KQA Pro's separate diagnostic for named-entity relation chains ending in an entity answer or a date-valued attribute query after dropping `FilterConcept` steps on the chain's entities (`python -m xgraph.kqapro profile`). Every other operation and dependency must still fit one chain. It includes strictly admitted chains and reports the additional ones separately. Execution ignores the dropped filters: gold membership measures reachability under these relaxed semantics, not complete-program operation coverage or equivalent answers to the original program.

**Intervention**:
A KG version that changes one fact of the KG a question is asked over, with the gold answer it implies, computed by executing the question's plan in it (`python -m xgraph.interventions`): on the question's gold chain (an on-chain intervention replaces a hop's object by a substitute, so that the plan reaches a new endpoint) or off it (an off-chain intervention replaces the object of a chain entity's fact under a relation the chain does not use, so that the answer must not change). A question is asked under several conditions: its original KG and its interventions.

**Plausibility stratum**:
How plausible an on-chain intervention's substitute is, by a fixed rule over the replaced object's class: a popular member (among the most sitelinks), a long-tail member (from the less linked half), or a popular entity of another class (cross-class); for MQuAKE also the real-world fact the case's edit replaced (`real`).

**Value source**:
Where an answer's value comes from while the relations are held fixed: bound by executing them in the KG (path mode), read from their facts in the prompt, written by the model as fact spans, or recalled closed book.

**Joint version fidelity**:
The share of questions answered right under every condition: the original KG, each on-chain intervention (sensitivity: answered with the new endpoint), and each off-chain one (invariance: answered as under the original KG).

**Sampled path**:
A relation path drawn at random from a KG directory for training (`python -m xgraph.pathdata examples`): from a start entity that its label names alone, hop by hop through articles that are no class, kept only when its relations reach exactly one endpoint from the start (or, with `--max-endpoints`, up to that many, the path's last among them) and it visits no held-out entity (one a benchmark question names, listed by `python -m xgraph.pathdata holdout`) and, unless `--no-holdout-compositions`, contains no held-out relation composition.

**Relation-balanced sampling**:
Drawing sampled paths around the relations of a relation inventory, the relations with at least a given number of triples in the KG (`python -m xgraph.pathdata examples --balance`), so that each reaches a minimum of occurrences where the KG and the holdout allow it and no relation sequence dominates; a relation the KG offers no more paths for is exhausted. Its inventory and weights come from the KG, never from a benchmark.

**Path-writing example**:
A chat example made from a sampled path: a question about the path (templated from its relation labels and aliases, or a naturalized question), and as the reply its path span and the endpoint as the answer span, a reply path mode accepts. Training on them (`python -m xgraph.train --paths`) asks whether a model learns to plan paths without benchmark data.

**Endpoint frontier**:
The entities a path-writing example's path resolves to as path mode with committed paths resolves it, found by matched execution over the KG index and served tokenizer path mode runs with; its size is recorded as the example's `endpoint_frontier` (`python -m xgraph.pathdata examples --index`, or `frontiers` for written examples). With more than one, the reply's answer names one of them arbitrarily, and `python -m xgraph.train --answer-loss singleton` gives its label no loss.

**Naturalized question**:
A path-writing example's question rewritten by a model from the start's label and the relation labels alone (`python -m xgraph.naturalize`), kept when a lexical check finds no leaked entity the path reaches and no dropped or repeated hop, and, with a reconstruction round, when a model reads the relations back from it in order.

### Engines

**Planning mask**:
The opt-in host-side processor in `xgraph.planbench_mask`: tokens continue
applicable PDDL action lines, advancing the fact state when an action closes.
It shares no KG index or GPU engine state. Primary EOS is independent of goal
satisfaction; goal-reached stopping is a separately named control.

**Planning problem**:
A PlanBench PDDL problem with objects, initial facts, goal facts and grounded
actions whose preconditions and effects come from its domain file
(`xgraph.planbench_pddl`). Planning actions modify facts rather than traverse KG
adjacency.

**Applicable ground action**:
An action with every parameter bound to a problem object, whose positive
preconditions hold and whose negative preconditions do not hold in the current
planning state. Applicability alone does not imply progress toward the goal.

**Planning goal success**:
A complete valid plan whose final state satisfies the supplied goal. An
inapplicable later action invalidates a plan even if an earlier state reached the
goal. This differs from plan validity, which checks executability alone.

**Planning renaming cluster**:
One Blocksworld problem together with its Mystery Blocksworld renaming. The
PlanBench panel keeps the pair together and records object count as its size
stratum, so renamings are not treated as independent problems.

**Engine**:
An implementation of advance and mask over a KG index that runs on the GPU, batched over all rows of a decoding step, with the KG index and every constraint state in GPU memory.

**Active slot**:
A request slot admitted by `Engine.begin`: `sync` advances its history and
`apply_mask` constrains its logits rows. Construction, `reset` and `end` leave
slots inactive. The direct sequence operations `advance` and `fill_mask` do not
depend on activation and do not activate slots.

**Oracle**:
A scalar CPU implementation of advance and mask, one sequence at a time, written for clarity. Tests check the engine against it step by step; it never serves requests. Also called the reference implementation.
_Avoid_: CPU prototype (there is none; the engine is GPU-native from the start)

**Engine/oracle agreement**:
Equal token masks and corresponding advances, including resolution and binding,
from corresponding initial states. The universal premise applies at every
reachable state; finite agreement tests supply evidence for selected histories
and configurations (`docs/engine-contract.md`).

**Engine adapter**:
The thin layer that hosts an engine inside an LLM server, such as a vLLM logits processor or an SGLang grammar backend.

**Worker extension**:
A class vLLM adds to its worker at construction to expose named RPC methods.
`xgraph.vllm.XGraphWorkerExtension` exposes `xgraph_collect_replay_status` for
replay status consumers. Method names and replay IDs cross the frontend boundary
as data; receipt storage stays in the worker that runs the engine adapter.

**Engine stand-in**:
A test object exposing engine host or tensor methods on CPU, backed by the oracle
or handwritten masks. It exercises Python consumers and record contracts without
establishing native engine/oracle agreement.

### Library documentation

**Demonstration tokenizer**:
A local byte-level BPE vocabulary used by CPU examples without model weights or
downloads. It exercises the same served-tokenizer/index contract, but an LLM
must use the tokenizer belonging to its weights.

**Documentation site**:
The Sphinx/Furo HTML guide built from `docs/index.md`, with API reference pages
generated from public docstrings. Library workflows and integration guides have
their own navigation; historical experiment records live under Research.

**CPU configuration check**:
Execution of an example's setup or worker-call contract with a stand-in runtime,
without loading a model or initializing CUDA. A dry run prints configuration;
neither operation establishes native agreement or serving behavior.

### Research planning

**Claims/evidence ledger**:
A planning inventory linking each candidate paper claim to byte-pinned primary
records, exact values, its evidence class and boundary, reviewer objections,
pending registered decisions and support status. Its generated rendering is
checked against the JSON inventory; it is not manuscript text.

**Status addendum**:
A dated overlay on a frozen claims/evidence ledger that updates only current
study status, evidence boundaries and unresolved dependencies from immutable
decision metadata. It preserves scientific rules, positive-claim support states,
original evidence pointers and compute; recorded non-admission is not an effect
estimate. An incomplete input binding keeps the addendum preparation-only.

**Evidence class**:
The origin of a claim's support: registered confirmatory, sealed validation,
descriptive validation, post hoc development, systems measurement, negative
result or pending. Sealing outcomes does not make an analysis selected after
reading them confirmatory. Support status separately records whether the
candidate claim is supported, bounded, pending, unsupported or retracted.

**Unsourced compute**:
Compute whose durable measured or conservatively charged receipt is unavailable
or outside the permitted evidence scope. Its amount is unknown, not zero; token
counts, forecasts and budget caps cannot fill it. Recorded compute retains failed
and aborted attempts and deduplicates reuse and recovery notes.

**Prior-work register**:
A dated evidence inventory linking a work's primary source and checked method
description to ledger claims, precise overlap, bounded differences, source
limitations and corrections to project notes. Source verification does not
qualify a released implementation, reproduce its results or establish priority.

**Primary verification scope**:
The part of a primary source actually checked: method sections, an abstract,
software documentation or artifact-specific terms. A cached record's checksum
establishes local integrity; treating it as primary requires independently
verified source origin. Missing detail in an abstract does not establish an
absent feature.

**Artifact terms register**:
A dated inventory separating each dataset, KG, model, software release and
generated artifact's primary licence evidence, verification status, intended-use
assessment and redistribution constraints. A code licence does not automatically
license separately hosted data or weights. A verified grant does not by itself
prove that a proposed use satisfies its conditions; release options are distinct
from an actual publication or a paper's release statement.

**Secondary-only terms evidence**:
A mirror's licence field or a later user's description of an upstream artifact's
terms, recorded as `verified-secondary-only`. It can establish the mirror's
declaration without establishing the original rightsholder's grant.

**Artifact paper decision**:
An evidence-register recommendation: `cite-only` keeps a literature or
qualification citation; `use-without-redistribution` keeps the disclosed research
scope while excluding the external package and content-bearing derivatives;
`must-drop` removes an unidentified artifact from paper/resource claims until
identified. The decision neither creates permission nor erases historical use.

**Sparse snapshot**:
An input document retaining only the lines cited by the register or report,
verbatim at their original line numbers; every other line is empty. Its input
manifest records the snapshot's hash, byte size, line count and cited line set,
plus the original document's hash, byte size and line count. Checking the
snapshot does not reauthenticate the omitted original text.

**Git object input**:
A harvested repository file pinned by its full `commit:path`, blob SHA-1,
content SHA-256 and byte size. The prior-work register verifies its bytes and
resolves its line citations through Git's object database, independently of
later working-tree edits. Reproduction requires the recorded commit and blob.

**Compute disclosure**:
A sourced inventory of run attempts, including failed launches, interruptions
and reuse. GPU run time is additive across attempts; it is distinct from device
utilization and the union of overlapping intervals on one physical GPU.
Generation workload timers provide lower bounds when startup and teardown are
missing. A recorded stage interval can provide an upper bound when it includes
waiting or CPU work; an absent upper bound remains unknown. CPU process time and
CPU wall time are reported separately.

**Evidence snapshot**:
A byte-pinned copy of a mutable source, its cited lines or selected metadata,
with original path, byte count, hash and exact selectors. Compressed copies
retain and verify exact original bytes. Dated records derive their frozen facts
from these retained bytes. A field recovered from an earlier committed disclosure
is labelled as recovery rather than a surviving primary receipt.

**Prefix pin**:
A byte length and SHA-256 binding only that initial part of an append-only source.
Later appends leave its identity and cited reads unchanged; rewriting or truncating
the pinned prefix fails verification.

**Git-object pin**:
A full `commit:path` reference and blob ID identifying tracked evidence independently
of its working copy. Verification reads the git object and checks its byte identity.

**Live-source drift**:
A changed, missing or unreadable original behind an evidence snapshot. The
compute disclosure reports it as a warning independently of frozen evidence
verification. A changed snapshot, rewritten append-only prefix or mismatched
derived value remains a verification failure.

**Cleanup cross-check**:
A recorded absence of overlap between evidence paths and an owner's cleanup
selection. It stores the plan path and hash, selection count and sorted-path
digest, and overlap count without listing the selected paths. The check can be
replayed from the unchanged plan; a missing or changed plan is not re-verifiable
and produces a warning independently of frozen evidence verification.

**Scratch cleanup plan**:
A fixed inventory separating regenerable working artifacts from retained inputs
and evidence. Execution binds the reviewed inventory and audit commit, checks
references and live use again, and records every removal. Allocated bytes from
`du` and blocks reclaimable after retained hardlinks are separate quantities.

## Flagged ambiguities

- **"trigger"**: the study's word for the span opener; code and docs say span opener.
- **"label"**: an entity's surface form or a relation's; say entity label or relation label where it matters.
- **"range"**: a row range in the label index or a range of token ids in the vocabulary; say which.
- **"row"**: a row of the label index. A slot of the decoding batch is a sequence, even where frameworks call it a row of the logits.


**Recovery child**:
A prospective execution derived from a frozen study after an infrastructure
failure. It retains the original scientific schedule and inherited charges,
while binding its repaired source, environment, owner records and remaining
budget. Creating new output paths cannot reset its persistent spending history
(`xgraph.serving_recovery`).

**Inventory archive**:
The lossless stream of process and GPU observations retained by
`xgraph.serving_recovery_monitor`. Complete end-of-file verification and observed
owned-resource absence are required for qualification; a signal acknowledgement
provides no cleanup proof.

**Engineering amendment**:
An append-only, source-qualified control change bound to the original recovery
registration, preparation, execution and shared claims. It retains the native
build, scientific schedule and cumulative budget. Original provenance and the
successor's effective controller/monitor/relay bindings are recorded separately;
the amendment's typed claim cannot reset a launch or spending history.
