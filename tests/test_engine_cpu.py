"""CPU semantic checks of extracted native helpers; CUDA agreement stays separate."""

import ctypes as ct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import byte_level_tokenizer
from tests.native_cpu import (
    SOURCE,
    NativeCPU,
    SequentialEngine,
    compile_cpp,
    compile_helpers,
    compile_lifecycle_helpers,
    function,
)
from xgraph import Entity, KnowledgeGraph, Oracle, ServedTokenizer, SpanSyntax, build_kg_index
from xgraph.engine import Engine, entity_rows


def test_dense_relation_ranks_keep_lexical_alias_ranks_and_default_bytes():
    from tests.test_oracle import audit_world
    from xgraph.audit import RelationScores

    _, index, served, _ = audit_world()
    terms = RelationScores(index, served)
    engine = Engine.__new__(Engine)
    engine._index, engine._audit_terms, engine._audit, engine._menu = index, terms, False, 5
    scores, aliases = [], []
    engine._native = SimpleNamespace(
        set_relation_scores=lambda slot, packed, stream: scores.append(packed.tobytes()),
        set_alias_ranks=lambda slot, packed, stream: aliases.append(packed.tobytes()),
    )
    question = "Which alternate?"
    engine.set_relation_question(0, question)
    expected = (
        np.array(
            [(index.relations.owner_ids.index(r), n) for r, n in terms.for_question(question)],
            dtype=np.int32,
        )
        .ravel()
        .tobytes()
    )
    assert scores == [expected] and aliases == [terms.alias_ranks(question).tobytes()]
    engine.set_relation_question(0, question, ranked=[["P1", 0]])
    assert scores[-1] == np.array([index.relations.owner_ids.index("P1"), 0], np.int32).tobytes()
    assert aliases[-1] == aliases[0]
    engine.set_relation_question(0, question)
    assert scores[-1] == scores[0] and aliases[-1] == aliases[0]


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    return compile_helpers(tmp_path_factory.mktemp("native-cpu"))


@pytest.mark.parametrize("label", ["yesman", "yes"])
@pytest.mark.parametrize("version_view", [False, True])
def test_unbound_verdict_closure_matches_oracle(native_library, label, version_view):
    from xgraph.kgversions import build_kg_versions

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity("application", (label,)),), (), ()), served, syntax, verdicts=True
    )
    oracle = Oracle(index, served, syntax)
    native = NativeCPU(native_library, index, served, syntax)
    state = oracle.initial_state()
    text = "⟦ yes" if label == "yesman" else "⟦ yes⟧"
    for token in served.encode(text):
        assert native.allowed() == oracle.allowed(state)
        state = oracle.advance(state, token)
        native.advance(token)
    assert native.allowed() == oracle.allowed(state)
    if label == "yes":
        view = build_kg_versions(index, {"edit": []}).view(index, "edit") if version_view else index
        assert native.closed(view).entity_ids == state.last_span.entity_ids == ("application",)


def test_bound_comparison_verdict_is_preserved(native_library):
    from tests.test_oracle import EXPRESSIONS, compare_kg

    kg, dates, served = compare_kg()
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax, dates=dates, verdicts=True)
    oracle = Oracle(index, served, syntax, **EXPRESSIONS)
    native = NativeCPU(native_library, index, served, syntax, **EXPRESSIONS)
    state = oracle.initial_state()
    for token in served.encode("⟦ same: Alien | Hair · country of origin⟧ so ⟦ yes⟧"):
        assert native.allowed() == oracle.allowed(state)
        state = oracle.advance(state, token)
        native.advance(token)
        if state.last_span is not None and state.last_span.resolved:
            assert native.closed().entity_ids == state.last_span.entity_ids


def test_unbound_reserved_id_is_ordinary_without_verdicts(native_library):
    from xgraph.index import VERDICTS

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity(VERDICTS["yes"], ("Application",)),), (), ()), served, syntax
    )
    native = NativeCPU(native_library, index, served, syntax)
    for token in served.encode("⟦ Application⟧"):
        native.advance(token)
    assert native.closed().entity_ids == (VERDICTS["yes"],)


@pytest.mark.parametrize("good", [False, True])
def test_entity_rows_include_trailing_unwritable_owners(good):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    entities = ((Entity("good", ("Good",)),) if good else ()) + (Entity("bad", ("Bad⟧",)),)
    index = build_kg_index(KnowledgeGraph(entities, (), ()), served, syntax)
    rows = entity_rows(index.labels)
    assert rows["entity_row_offsets"].tolist() == ([0, 1, 1] if good else [0, 0])
    assert rows["entity_rows"].tolist() == ([0] if good else [])


def test_mixed_anchor_with_trailing_unwritable_owner_matches_oracle():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity("good", ("Good",)), Entity("bad", ("Bad⟧",))), (), ()),
        served,
        syntax,
    )
    uploaded = []
    engine = Engine.__new__(Engine)
    engine._index, engine._entity_index = index, None
    engine._native = SimpleNamespace(set_reached=lambda slot, rows, stream: uploaded.append(rows))
    oracle = Oracle(index, served, syntax, derivation=True)
    engine.anchor(0, ["good", "bad", "good"])
    assert uploaded[0].tolist() == sorted(oracle.anchored_state(["good", "bad"]).reached)
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        engine.anchor(0, ["bad"])
    assert len(uploaded) == 1


def test_native_anchor_slot_rejects_negative_index(tmp_path):
    source = SOURCE.read_text()
    reached = function(source, "  void set_reached(")
    sequences = function(source, "  void check_sequences(")
    cpp = (
        r"""
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
constexpr int32_t kMaxReached=64,kMaxFacts=64,kMemoryCounts=7;
constexpr int cudaMemcpyHostToDevice=1;
using cudaStream_t=uintptr_t;
cudaStream_t as_stream(uintptr_t stream) { return stream; }
void check(int, const char*) {}
int copies=0;
int cudaMemcpyAsync(void* dst,const void* src,size_t n,int,cudaStream_t) {
 ++copies; std::memcpy(dst,src,n); return 0;
}
int cudaStreamSynchronize(cudaStream_t) { return 0; }
template<class T> struct HostArray {
 const T* p;size_t n;size_t size() const {return n;} const T* data() const {return p;}
};
struct Storage {int32_t* p;int32_t* get(){return p;}};
struct Engine {
 int32_t rows_backing[4096]{},counts_backing[4096]{};
 Storage fact_rows_{rows_backing+2048},fact_counts_{counts_backing+2048};
 struct Tables {struct {int32_t memory_ints=640;} layout;} tables_;
 int32_t max_sequences_=2;
"""
        + reached
        + "\n"
        + sequences
        + r"""
};
int main() {
 Engine e; const int32_t rows[]={17};
 for(int32_t slot : {-1,-2,2,std::numeric_limits<int32_t>::max()}) {
  bool rejected=false;
  try { e.set_reached(slot,HostArray<int32_t>{rows,1},0); }
  catch(const std::invalid_argument&) { rejected=true; }
  if(!rejected || copies) {std::cerr<<"slot "<<slot<<" was accepted or copied"; return 1;}
 }
 for(int32_t slot : {0,1}) {
  e.set_reached(slot,HostArray<int32_t>{rows,1},0);
  if(e.rows_backing[2048+slot*640+192]!=17 || e.counts_backing[2048+slot*7+1]!=1)
   return 2;
 }
 if(copies!=4) return 3;
}
"""
    )
    binary = compile_cpp(tmp_path, cpp)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def bias_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-bias")
    compile_helpers(directory)
    source = SOURCE.read_text()
    constants = (
        source[
            source.index("constexpr int32_t kBiasTokens") : source.index(
                "// Whether a token continues"
            )
        ]
        if "constexpr int32_t kBiasTokens" in source
        else ""
    )
    collector = function(source, "    if (t.escape_bias != 0.0f && pos == 0 && startable)")
    continuation = function(source, "__device__ bool continues_label")
    configure = (
        "t.bias_capacity=t.escape.offsets[1]-t.escape.offsets[0];"
        if "int32_t bias_capacity;" in source and "MemoryLayout layout;" not in source
        else ""
    )
    code = (
        (directory / "native.cpp").read_text()
        + constants
        + continuation
        + r"""
extern "C" int gather_bias(int32_t* bias_tokens) {
 const auto& s=state;
 const Memory f=memory_at(mem,counts,false,0,t.layout);
 const int32_t row=0, pos=s.esc_pos;
 int32_t bias_count[1]{};
 const Options options=span_options(t,s,f);
 const bool observing=s.observe_offset<s.observe_length || s.observe_pos>0 ||
  s.select_pos>0 || requires_selection(t,f,s) || s.defer_pos>0 || s.decision_phase>0;
 const bool startable=!observing && s.lo>=0 && !options.stop && may_escape(t,f,s);
"""
        + configure
        + collector
        + r"""
 return bias_count[0];
}
"""
    )
    return compile_cpp(directory, code, shared=True)


def escape_bias_world(shared=False):
    escape = "abcdefghij"
    merges = [(escape[: k - 1], escape[k - 1]) for k in range(2, len(escape) + 1)]
    served = ServedTokenizer(byte_level_tokenizer(merges))
    syntax = SpanSyntax(opener="[", closer="]", separator="/", label_prefix="", escape=escape)
    label = "abacus" if shared else "Atlas"
    entities = (Entity("item", (label,)),) + ((Entity("other", ("az",)),) if shared else ())
    index = build_kg_index(KnowledgeGraph(entities, (), ()), served, syntax)
    expected = {
        t
        for t in range(served.vocab_size)
        if served.token_bytes(t)
        and escape.encode().startswith(served.token_bytes(t))
        and (not shared or served.token_bytes(t) not in (b"a", b"ab"))
    }
    assert len(expected) == (8 if shared else 10)
    return index, served, syntax, expected


@pytest.mark.parametrize("shared", [False, True])
def test_escape_bias_covers_more_than_eight_prefix_tokens(bias_library, shared):
    index, served, syntax, expected = escape_bias_world(shared)
    native = NativeCPU(bias_library, index, served, syntax, escape_bias=-3)
    oracle = Oracle(index, served, syntax)
    state = oracle.initial_state()
    for token in served.encode(syntax.opener):
        native.advance(token)
        state = oracle.advance(state, token)
    assert native.allowed() == oracle.allowed(state)
    out = np.zeros(32, np.int32)
    native.lib.gather_bias.argtypes = [ct.c_void_p]
    count = native.lib.gather_bias(out.ctypes.data)
    assert set(out[:count].tolist()) == expected


def operator_escape_bias_world():
    from tests.test_oracle import EXPRESSIONS, compare_kg

    escape = "earlier?xy"
    merges = [(escape[: k - 1], escape[k - 1]) for k in range(2, len(escape) + 1)]
    served = ServedTokenizer(byte_level_tokenizer(merges))
    syntax = SpanSyntax(opener="[", closer="]", separator="/", label_prefix="", escape=escape)
    kg, dates, _ = compare_kg()
    index = build_kg_index(kg, served, syntax, dates=dates, verdicts=True)
    # These alone leave the shared earlier prefix for the escape's question mark.
    expected = {
        t
        for t in range(served.vocab_size)
        if served.token_bytes(t) in (b"earlier?", b"earlier?x", b"earlier?xy")
    }
    assert len(expected) == 3
    return index, served, syntax, expected, EXPRESSIONS


def test_escape_bias_excludes_shared_operator_prefix_tokens(bias_library):
    index, served, syntax, expected, options = operator_escape_bias_world()
    native = NativeCPU(bias_library, index, served, syntax, escape_bias=-0.5, **options)
    oracle = Oracle(index, served, syntax, **options)
    state = oracle.initial_state()
    for token in served.encode(syntax.opener):
        native.advance(token)
        state = oracle.advance(state, token)
    assert native.allowed() == oracle.allowed(state)
    out = np.zeros(32, np.int32)
    native.lib.gather_bias.argtypes = [ct.c_void_p]
    count = native.lib.gather_bias(out.ctypes.data)
    assert set(out[:count].tolist()) == expected


@pytest.fixture(scope="module")
def lifecycle_library(tmp_path_factory):
    return compile_lifecycle_helpers(tmp_path_factory.mktemp("native-lifecycle"))


@pytest.fixture
def sequential_engine(lifecycle_library, monkeypatch):
    """Run the real CUDA agreement test bodies with serialized native kernels."""
    import torch

    from tests import test_engine

    def factory(*args, **kwargs):
        return SequentialEngine(lifecycle_library, *args, **kwargs)

    monkeypatch.setattr(test_engine, "Engine", factory)
    for name in ("tensor", "zeros"):
        original = getattr(torch, name)

        def on_cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, on_cpu)
    return factory


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("bias", [-3.0, -0.5])
def test_escape_bias_logit_rows_match_oracle_on_cpu(sequential_engine, shared, bias):
    from tests.test_engine import test_escape_bias_covers_more_than_eight_prefix_tokens

    test_escape_bias_covers_more_than_eight_prefix_tokens(shared, bias)


def test_operator_escape_bias_logit_rows_match_oracle_on_cpu(sequential_engine):
    from tests.test_engine import test_escape_bias_excludes_shared_operator_prefix_tokens

    test_escape_bias_excludes_shared_operator_prefix_tokens()


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_version_slot_reset_matches_rebuilt_engines_on_cpu(sequential_engine, committed, anchored):
    from tests.test_engine import test_version_isolation_mixed_slots_and_reset_equal_rebuilt_engines

    test_version_isolation_mixed_slots_and_reset_equal_rebuilt_engines(committed, anchored)


def test_reset_clears_versions_and_deactivates_history_replay(lifecycle_library):
    from tests.test_oracle import ISOLATION_EDITS, ISOLATION_KG
    from xgraph.kgversions import build_kg_versions

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(ISOLATION_KG, served, syntax)
    versions = build_kg_versions(index, ISOLATION_EDITS)
    options = dict(paths=True, committed_paths=True, escapes=False)
    native = SequentialEngine(
        lifecycle_library, index, served, syntax, 2, versions=versions, **options
    )
    slots = np.array([1, 0], np.int32)
    native.begin(slots)
    native.set_version(0, "inner")
    native.set_version(1, "terminal")
    native.anchor(0, ["Q1"])
    native.anchor(1, ["Q1"])
    # Invalid prompt tokens must be skipped, and the consumed prefix not replayed.
    history = np.full((2, 3), -99, np.int32)
    token = served.encode(syntax.opener)[0]
    history[:, 2] = token
    prompt, total = np.array([2, 2], np.int32), np.array([3, 3], np.int32)
    native.sync(slots, history, prompt, total)
    native.sync(slots, history, prompt, total)
    np.testing.assert_array_equal(native.states_host()["pos"], [1, 1])
    assert all(native.counts_host()[:, 1] > 0)  # anchors
    assert all(native.counts_host()[:, 3] > 0)  # pinned KG versions

    native.reset()
    np.testing.assert_array_equal(native.counts_host(), 0)
    native.sync(slots, history, prompt, total)
    assert all(not values.any() for values in native.states_host().values())
    logits = np.full((2, served.vocab_size), 1.25, np.float32)
    native.apply_mask(logits, slots)
    np.testing.assert_array_equal(logits, 1.25)

    # Admission is explicit after reset, and only the admitted slot advances.
    native.begin(slots[:1])
    native.sync(slots, history, prompt, total)
    oracle = Oracle(index, served, syntax, **options)
    state = oracle.advance(oracle.initial_state(), token)
    np.testing.assert_array_equal(native.states_host()["pos"], [0, len(state.tail)])
    native.fill_mask()
    for slot, expected in ((0, oracle.initial_state()), (1, state)):
        words = native.mask_host()[slot]
        allowed = frozenset(
            t for t in range(served.vocab_size) if int(words[t >> 5]) & (1 << (t & 31))
        )
        assert allowed == oracle.allowed(expected)


def test_all_capability_layout_sizes_and_offsets(tmp_path):
    source = SOURCE.read_text()
    definitions = source[source.index("// Table codes") : source.index("void check(cudaError_t")]
    code = (
        """
#include <cstdint>
#include <iostream>
#define __device__
"""
        + definitions
        + r"""
int main() {
 const int expected[4][2]={{640,1040},{704,1104},{1184,1584},{1248,1648}};
 for(int mask=0;mask<64;++mask) {
  const int words=(mask&1)?3:0;
  const MemoryLayout layout(words,mask&2,mask&4,mask&8,mask&16,mask&32,10);
  const int count=expected[((mask&4)?2:0)+((mask&2)?1:0)][(mask&(8|16|32))?1:0]+2*words;
  if(layout.memory_ints!=count || layout.menu_offset!=((mask&16)?count-400:0) ||
     layout.bias_capacity!=10 || layout.relation_words!=words) return 1;
  int32_t own[10000]{}, counts[kMemoryCounts]{};
  const Memory m=memory_at(own,counts,false,5,layout);
  if(m.rows!=own || m.reached!=own+192 || m.frontier!=own+256 ||
     m.frontier_ranks!=own+320 || m.frontier_b!=own+384 ||
     m.frontier_b_ranks!=own+448 || m.starts!=own+512 ||
     m.relation_support!=reinterpret_cast<uint32_t*>(own+640) ||
     m.next_support!=reinterpret_cast<uint32_t*>(own+640+words) ||
     m.observation_rows!=own+640+2*words || m.loop_rows!=own+704+2*words ||
     m.audit_items!=((mask&16)?own+count-400:m.observation_rows) || m.slot!=5) return 2;
 }
 std::cout << "64 capability layouts preserve allocation sizes and every row-list offset\n";
}
"""
    )
    binary = compile_cpp(tmp_path, code)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def rank_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-ranking")
    compile_helpers(directory)
    code = (
        (directory / "native.cpp").read_text()
        + r"""
extern "C" void synthetic_offsets(const int64_t* offsets, int version) {
 static const int64_t versions[]={0,2}; static const int32_t subjects[]={0,1};
 if(version) {
  t.versions.offsets=versions; t.versions.subjects=subjects;
  t.versions.adjacency=offsets; t.versions.count=1;
 } else t.adjacency=offsets;
 counts[3]=version;
}
extern "C" int64_t inspect_degree(int row) {
 return entity_degree(t,memory_at(mem,counts,false,0,t.layout),row);
}
extern "C" int inspect_best() {
 return best_row(t,memory_at(mem,counts,false,0,t.layout),0,2,true);
}
extern "C" int inspect_starts(int32_t* out) {
 int32_t ranks[kMaxReached]{}, n=0;
 return ranked_starts(t,memory_at(mem,counts,false,0,t.layout),0,2,
                      Frontier{out,ranks,&n});
}
extern "C" int inspect_observation() {
 static int32_t offsets[100]{}; t.observation_offsets=offsets; t.observe=2;
 State s{}; s.lo=0; s.hi=2;
 const Memory f=memory_at(mem,counts,false,0,t.layout);
 prepare_observation(t,f,s);
 return f.observation_rows[0];
}
extern "C" int inspect_terminal() {
 t.observe_terminal=2;
 const Memory f=memory_at(mem,counts,false,0,t.layout);
 *f.reached_count=2; f.reached[0]=0; f.reached[1]=1; State s{};
 prepare_audit(t,f,s,false,true);
 return f.audit_items[1];
}
"""
    )
    return compile_cpp(directory, code, shared=True)


@pytest.mark.parametrize("degree", [2**31 - 1, 2**31, 2**32 - 1, 2**32])
@pytest.mark.parametrize("version", [False, True])
@pytest.mark.parametrize(
    "links,rank,shape,expected,terminal",
    [
        ((10, 0), "degree", "first", 0, 0),
        ((10, 0), "sitelinks", "first", 0, 0),
        ((0, 10), "degree", "first", 0, 1),
        ((0, 10), "sitelinks", "first", 1, 1),
        ((7, 7), "degree", "first", 0, 0),
        ((7, 7), "sitelinks", "first", 0, 0),
        ((7, 7), "degree", "second", 1, 1),
        ((7, 7), "sitelinks", "second", 1, 1),
        ((7, 7), "degree", "equal", 0, 0),
        ((7, 7), "sitelinks", "equal", 0, 0),
    ],
)
def test_native_degree_ranking_int32_boundary(
    rank_library, degree, version, links, rank, shape, expected, terminal
):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity("large", ("Shared",)), Entity("small", ("Shared",))), (), ()),
        served,
        syntax,
        sitelinks=dict(zip(("large", "small"), links, strict=True)),
    )
    native = NativeCPU(rank_library, index, served, syntax, derivation=True, start_rank=rank)
    native.lib.synthetic_offsets.argtypes = [ct.c_void_p, ct.c_int]
    native.lib.inspect_degree.restype = ct.c_int64
    native.lib.inspect_starts.argtypes = [ct.c_void_p]
    a, b = {"first": (degree, 1), "second": (1, degree), "equal": (degree, degree)}[shape]
    offsets = np.array([0, a, a + b], np.int64)
    native.lib.synthetic_offsets(offsets.ctypes.data, version)
    assert native.lib.inspect_degree(0) == a
    assert native.lib.inspect_degree(1) == b
    assert native.lib.inspect_best() == expected
    rows = np.zeros(64, np.int32)
    assert native.lib.inspect_starts(rows.ctypes.data) == 2
    assert rows[:2].tolist() == [expected, 1 - expected]
    assert native.lib.inspect_observation() == expected
    assert native.lib.inspect_terminal() == terminal


@pytest.mark.parametrize("implementation", ["engine", "oracle"])
@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, (False, False, False)),
        ({"paths": True, "observe": 1, "observe_required": True}, (False, True, False)),
        ({"paths": True, "observe": 1, "observe_answer": True}, (True, False, True)),
        ({"paths": True, "observe": 1, "menu": 2, "menu_close_next": True}, (True, False, True)),
        (
            {"paths": True, "observe": 1, "observe_required": True, "observe_defer": 1},
            (True, True, True),
        ),
        ({"paths": True, "observe_terminal": 2}, (True, False, False)),
        ({"paths": True, "audit": True}, (True, False, False)),
    ],
)
def test_host_policy_implications_match_explicit_expectations(
    monkeypatch, implementation, options, expected
):
    from dataclasses import replace

    import xgraph.engine as module

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(KnowledgeGraph((Entity("e", ("Entity",)),), (), ()), served, syntax)
    index = replace(index, sitelinks=np.array([1], dtype=np.int64))
    captured = {}

    def native(*args, **kwargs):
        # The native constructor's end_after_answer argument occupies ABI position 15.
        captured.update(
            end=args[15],
            select=kwargs.get("observe_select", False),
            answer=kwargs.get("observe_answer", False),
        )
        return SimpleNamespace()

    monkeypatch.setattr(module, "_engine", SimpleNamespace(Engine=native))
    value = (
        module.Engine(index, served, syntax, 1, eos_token_ids=[0], **options)
        if implementation == "engine"
        else Oracle(index, served, syntax, eos_token_ids=[0], **options)
    )
    actual = (
        (captured["end"], captured["select"], captured["answer"])
        if implementation == "engine"
        else (value._end_after_answer, value._observe_select, value._observe_answer)
    )
    assert actual == expected


@pytest.mark.parametrize("implementation", ["engine", "oracle"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"end_after_answer": True}, "end_after_answer needs path mode"),
        ({"committed_paths": True}, "committed_paths needs path mode"),
        ({"paths": True, "menu": True}, "menu must be an integer"),
        ({"paths": True, "observe": True}, "observe must be an integer"),
        ({"paths": True, "observe": 1, "observe_width": 65}, "observe_width must be an integer"),
        ({"paths": True, "observe_defer": 9}, "observe_defer must be an integer"),
        ({"paths": True, "menu_close_next": True}, "menu_close_next needs observe"),
        (
            {"paths": True, "observe_required": True},
            "observe_required and observe_no_loop need observe",
        ),
        ({"paths": True, "observe_terminal": 65}, "observe_terminal must be an integer"),
        (
            {"paths": True, "observe_terminal": 2, "observe": 1},
            "observe_terminal needs frontier paths",
        ),
        ({"paths": True, "audit": True, "observe": 1}, "audit needs frontier path mode"),
        ({"audit_entities": True}, "audit_entities must be an integer"),
        ({"audit_relations": 65}, "audit_relations must be an integer"),
        ({"terminal_order": "random"}, "terminal_order/terminal_seed need observe_terminal"),
        ({"terminal_seed": -1}, "terminal_seed must be an unsigned"),
        ({"paths": True, "observe_answer": True, "eos_token_ids": []}, "nonempty EOS token ids"),
        ({"paths": True, "observe_terminal": 2}, "needs an index with sitelink counts"),
        # Multiple-invalid cases pin existing validation precedence.
        (
            {"paths": True, "menu": 17, "observe_answer": True, "eos_token_ids": []},
            "menu must be an integer",
        ),
        (
            {"paths": True, "observe_answer": True, "observe": 65, "eos_token_ids": []},
            "nonempty EOS token ids",
        ),
    ],
)
def test_host_policy_rejections_have_independent_expected_errors(
    monkeypatch, implementation, options, message
):
    import xgraph.engine as module

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(KnowledgeGraph((Entity("e", ("Entity",)),), (), ()), served, syntax)
    monkeypatch.setattr(module, "_engine", SimpleNamespace())
    options = {"eos_token_ids": [0], **options}
    with pytest.raises(ValueError, match=message):
        if implementation == "engine":
            module.Engine(index, served, syntax, 1, **options)
        else:
            Oracle(index, served, syntax, **options)
