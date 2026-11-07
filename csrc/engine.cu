// The GPU engine: advance and mask for every sequence of a decoding step.
//
// Everything a step touches lives in GPU memory: the label index and relation
// index rows, the adjacency, the precomputed tables, one constraint state per
// sequence, and the output mask. The semantics are the oracle's
// (src/xgraph/oracle.py); the tables are built on the host by
// src/xgraph/engine.py. This first version is correct but untuned
// (docs/gpu-native-engine-adr.md).
//
// Observed paths (observe > 0): an additional canonical marker branch commits a
// complete start/relation label; the GPU selects a display list from its capped,
// versioned frontier. Payload masks contain one indexed-label or punctuation token.
// State cursors plus up to kMaxReached label rows are copied for speculative rows;
// no host transfer, tokenization or extra execution hop is involved. The usual
// separator/closer (including STOP) follows the payload. See observed-paths-adr.md.

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

// Table codes for a token in free text or in an escaped span.
constexpr int8_t kReject = -1;    // not allowed
constexpr int8_t kComplete = -2;  // the delimiter ends exactly at the token's end

enum Phase : int32_t { kFree = 0, kSpan = 1, kEscaped = 2, kError = 3 };
enum Segment : int32_t { kSubject = 0, kRelation = 1, kObject = 2 };
enum Closed : int32_t {
  kNone = 0,
  kUnresolved = 1,
  kEntitySpan = 2,
  kFactSpan = 3,
  kPathSpan = 4,
  kComparison = 5,
  kRelationPlan = 6,  // syntax-only path, unexecuted and unbound
};

// The span a sequence closed most recently: its kind, and its entity rows, or for a
// fact span its object, subject, and relation rows, or for a comparison the first label
// rows of its operands' start entities (lo: A's, hi: B's), its result in subj_lo (a date
// comparison's 0: A, 1: B, 2: both, undetermined; a value comparison's 3: yes, 4: no), and
// its operator in subj_hi.
// Entity spans use subj_lo == -1 for an unbound closure: export excludes verdict owners.
struct LastSpan {
  int32_t kind;  // Closed
  int32_t lo, hi;
  int32_t subj_lo, subj_hi, rel_lo, rel_hi;
};

// The constraint state of one sequence. All fields are zero in the initial state.
// Row ranges index the label index, except lo and hi in the relation segment and
// the relation rows, which index the relation index.
struct State {
  int32_t phase;    // Phase
  int32_t pos;      // free text: opener bytes matched; escaped span: closer bytes matched
  int32_t segment;  // Segment
  int32_t depth;
  int32_t lo, hi;  // label branch: the row range (possibly empty); -1, -1 once it is dead
  int32_t close_pos;  // closer branch: bytes matched, 0 when not alive
  int32_t close_lo, close_hi;  // rows of the complete label the closer started after
  int32_t sep_pos;  // separator branch: bytes matched, 0 when not alive
  int32_t sep_lo, sep_hi;  // rows of the complete label the separator started after
  int32_t esc_pos;  // escape branch: bytes matched, 0 when not alive
  int32_t subj_lo, subj_hi;  // relation and object segments: the candidate subject rows
  int32_t rel_lo, rel_hi;    // object segment: the candidate relation rows
  int32_t subj_key;  // relation and object segments: the first row of the subject label
  int32_t hops;      // path mode: the relations of the open path, this one included
  // Expressions: the operator branch (bytes matched after the label prefix's, 0 when not
  // alive, and the first operator those bytes are consistent with), the operator the span
  // started with (its position + 1, 0 for none), and in a comparison's subject segment the
  // operand being written (0: A, 1: B).
  int32_t op_pos;
  int32_t op_kind;
  int32_t op;
  int32_t operand;
  int32_t stop_used_lo, stop_used_hi;  // STOP: consumed raw question words, a uint64 mask
  int32_t answer_complete;  // end after answer: only the served model's EOS ids may follow
  LastSpan last;
  // Observed paths: canonical marker branch, then a GPU-selected list of label rows.
  // Appended so every pre-existing field keeps its offset and off-mode value.
  int32_t observe_pos, observe_lo, observe_hi, observe_depth;
  int32_t observe_offset, observe_length, observed, observe_count, observe_more;
  int32_t select_pos, selecting, selected;
  int32_t select_segment, select_lo, select_hi, select_depth;
  int32_t answer_pos;
  int32_t loop_count;
  int32_t audit_phase, audit_item, audit_offset, audit_items;
  int32_t audit_choices, audit_pos, audit_action, audit_hops;
  int32_t defer_count, defer_pos;
  int32_t decision_phase, decision_pos, decision_choices;
};
constexpr int kStateFields = sizeof(State) / sizeof(int32_t);

// A sequence's memory. The fact spans it closed, its first kMaxFacts of them, for the
// no-repeat rule: each as the first label index row of its subject label, the first
// relation index row of its relation label, and the first label index row of its object
// label, which name the three labels (recorded and read only with repeats off). And in
// derivation mode the object rows its last fact span reached, its first kMaxReached of
// them in row order (a count of 0: no fact span closed yet). In path mode the open path's
// frontier, and with committed paths each frontier entity's start rank beside it; with
// expressions, a comparison's second frontier (operand B's) with its ranks, and each
// operand's ranked starts. And the KG version the sequence reads (0: the base KG), kept
// beside the counts.
constexpr int32_t kMaxFacts = 64;
constexpr int32_t kMaxReached = 64;
// Path mode: the most relations a path names; after them it must close (the oracle's
// MAX_PATH_RELATIONS).
constexpr int32_t kMaxPathRelations = 8;
// Exact capped entity sets, a hash prefilter, and complete relation-label ranges.
constexpr int32_t kLoopEntryInts = kMaxReached + 4;
constexpr int32_t kLoopMemoryInts = kMaxPathRelations * kLoopEntryInts;
constexpr int32_t kMaxStopWords = 64;
constexpr int32_t kMaxStopBytes = 4096;
// Per sequence: the facts, then seven lists of kMaxReached (reached rows, frontier, its ranks,
// operand B's frontier, its ranks, A's starts, B's starts).
constexpr int32_t kMemoryInts = kMaxFacts * 3 + 7 * kMaxReached;
// Per sequence: the fact count, the reached count, the frontier count, the KG version, and
// operand B's frontier count, then whether this slot has STOP relations installed.
constexpr int32_t kMemoryCounts = 7;
constexpr int32_t kAuditItems = 400;
constexpr int32_t kAuditScores = 64;
// Immutable slot and row buffer sizes, computed once from setup capabilities.
struct MemoryLayout {
  int32_t relation_words, observation_offset, loop_offset, menu_offset, memory_ints;
  int32_t bias_capacity;
  MemoryLayout() = default;
  MemoryLayout(int32_t words, bool observe, bool no_loop, bool audit, bool menu,
               bool terminal, int32_t bias)
      : relation_words(words), observation_offset(kMemoryInts + 2 * words),
        loop_offset(observation_offset + kMaxReached), menu_offset(0),
        memory_ints(observation_offset + (observe ? kMaxReached : 0) +
                    (no_loop ? kLoopMemoryInts : 0)), bias_capacity(bias) {
    if (menu) menu_offset = memory_ints;
    if (audit || menu || terminal) memory_ints += kAuditItems;
  }
};

struct Memory {
  int32_t* rows;            // [kMaxFacts * 3] of this sequence
  int32_t* count;           // of this sequence
  int32_t* reached;         // [kMaxReached] of this sequence
  int32_t* reached_count;   // of this sequence
  int32_t* frontier;        // path mode: [kMaxReached], the open path's current entities
  int32_t* frontier_count;  // as each entity's first label row
  // Committed paths: [kMaxReached], the rank of the best start that reaches each frontier
  // entity; the frontier is then ordered by rank and then row, otherwise by row.
  int32_t* frontier_ranks;
  // Expressions: operand B's frontier, its ranks, and its count, as for A above; and the
  // starts each operand was ranked with, best first (starts[k]: A's start of rank k,
  // starts[kMaxReached + k]: B's), as each entity's first label row.
  int32_t* frontier_b;
  int32_t* frontier_b_ranks;
  int32_t* frontier_b_count;
  int32_t* starts;
  bool record;  // advancing: closing a span updates the memory
  int32_t version;  // the KG version the sequence reads, 0 for the base KG (see edges_of)
  int32_t slot;  // immutable per-request STOP tables are addressed by the real slot, even for drafts
  bool stop;     // this slot has STOP relations installed, cleared by begin/end/reset
  uint32_t* relation_support;  // successor: [relation_words] next relation ids
  uint32_t* next_support;      // same size; scratch while consuming an ambiguous label
  int32_t* observation_rows;   // optional [kMaxReached], after the successor bitmaps
  int32_t* loop_rows;          // optional [kMaxPathRelations, kLoopEntryInts]
  int32_t* audit_items;        // shared audit/menu descriptors; menu storage follows observations/history
  int32_t* audit_score_count;
};

// The base memory plus two successor bitmaps, held at own, and kMemoryCounts counts.
__device__ Memory memory_at(int32_t* own, int32_t* counts, bool record, int32_t slot,
                            const MemoryLayout& layout) {
  Memory m;
  m.rows = own;
  m.count = counts;
  m.reached = own + kMaxFacts * 3;
  m.reached_count = counts + 1;
  m.frontier = m.reached + kMaxReached;
  m.frontier_count = counts + 2;
  m.frontier_ranks = m.frontier + kMaxReached;
  m.frontier_b = m.frontier_ranks + kMaxReached;
  m.frontier_b_ranks = m.frontier_b + kMaxReached;
  m.frontier_b_count = counts + 4;
  m.starts = m.frontier_b_ranks + kMaxReached;
  m.record = record;
  m.version = counts[3];
  m.slot = slot;
  m.stop = counts[5] != 0;
  m.relation_support = reinterpret_cast<uint32_t*>(own + kMemoryInts);
  m.next_support = m.relation_support + layout.relation_words;
  m.observation_rows = own + layout.observation_offset;
  m.loop_rows = own + layout.loop_offset;
  m.audit_items = layout.menu_offset ? own + layout.menu_offset : m.observation_rows;
  m.audit_score_count = counts + 6;
  return m;
}

// Expressions: the most operators a syntax may have.
constexpr int32_t kMaxOperators = 4;

void check(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    cudaGetLastError();  // clear it, so that the next call does not report it again
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(size_t size) : size_(size) {
    if (size_ > 0) check(cudaMalloc(&ptr_, size_ * sizeof(T)), "cudaMalloc");
  }
  DeviceArray(const T* host, size_t size) : DeviceArray(size) {
    if (size_ > 0) {
      check(cudaMemcpy(ptr_, host, size_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
  }
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept { *this = std::move(other); }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    return *this;
  }
  T* get() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

// Rows of the label index or the relation index, sorted by token sequence.
struct Rows {
  const int32_t* tokens;
  const int64_t* offsets;  // [count + 1]
  const int32_t* owner;    // the entity or relation each row names
  int32_t count;
};

// Tokens that extend a partially typed delimiter: for each byte position, the
// tokens whose bytes continue the delimiter from there, and where they end.
struct Delimiter {
  const int32_t* offsets;  // [length + 1], into tokens and next_pos
  const int32_t* tokens;
  const int32_t* next_pos;
  int32_t length;
};

// KG versions (docs/kg-versions-adr.md): for each version, the subjects it edits, sorted,
// each with its whole versioned edge list in the base adjacency's layout (built by
// src/xgraph/kgversions.py).
struct Versions {
  const int64_t* offsets;    // [count + 1]: version k's subjects are [offsets[k-1], offsets[k])
  const int32_t* subjects;   // entity positions, sorted within a version
  const int64_t* adjacency;  // [subjects + 1]: each edited subject's edges
  const int32_t* edge_relations;  // sorted within a subject
  const int32_t* edge_objects;    // label index rows, sorted within a subject and relation
  int32_t count;
};

// Device pointers of every table, passed to the kernels by value.
struct Tables {
  const int32_t* request_path_options;  // optional [slots, 10], -1 restores defaults
  const int32_t* request_menu_tokens;
  const int32_t* request_menu_offsets;
  const int32_t* request_terminal_tokens;
  const int32_t* request_terminal_offsets;
  const uint32_t* terminal_order_config;  // [order, seed], display rows only
  Rows entities;   // the label index
  Rows relations;  // the relation index
  const int32_t* relation_row_offsets;  // [relations + 1]: each relation's relation index rows
  const int32_t* relation_rows;
  const int64_t* adjacency;  // [entities + 1]: each subject's edges
  const int64_t* entity_row_offsets;  // [entities + 1]: each entity's label index rows
  const int32_t* entity_rows;         // in row order
  const int32_t* edge_relations;  // sorted within a subject
  const int32_t* edge_objects;    // label index rows, sorted within a subject and relation
  Versions versions;             // read through edges_of
  const uint32_t* root_mask;      // [words]
  const int64_t* wide_keys;       // sorted: row range start * 8 + depth of wide prefixes
  const uint32_t* wide_masks;     // [wide prefixes, words]: their next tokens
  int32_t wide_count;
  const int8_t* free_next;        // [opener length, vocab]
  const uint32_t* free_mask;      // [opener length, words]
  const uint32_t* eos_mask;       // [words], the served model's explicit EOS ids
  const int8_t* esc_next;         // [closer length, vocab]
  const uint32_t* esc_mask;       // [closer length, words]
  Delimiter closer;
  Delimiter separator;
  Delimiter escape;
  int32_t vocab;
  int32_t words;
  bool repeats;  // false: a fact span cannot close with the labels of an earlier one
  // Release (apply_mask only): inside a span, a row whose model puts less than
  // exp(release_log_mass) on the allowed set, or (release_top) whose own top token is
  // disallowed, may write only the escape marker. release_log_mass is -inf when off.
  float release_log_mass;
  bool release_top;
  bool derivation;  // spans bound by entity (see Memory, best_subject, record_reached)
  // apply_mask only: added to the logits of the tokens that commit to the escape marker
  // (they start it and can continue neither the label, the separator, nor the closer),
  // inside spans; 0 when off.
  float escape_bias;
  // Path mode: a span is a subject and relations only, each one the frontier (the current
  // entities, kept in the memory's reached rows as each entity's first label row) has.
  bool paths;
  bool end_after_answer;
  int32_t observe;  // maximum displayed labels; zero leaves every old mask/state unchanged
  int32_t observe_width;  // show the whole capped frontier when no larger than this
  int32_t observe_defer;
  bool menu_close_next;
  const int32_t* defer_tokens;
  int32_t defer_length;
  const int32_t* decision_tokens;
  const int32_t* decision_offsets;
  const int32_t* observation_tokens;  // independently tokenized pieces (see observations.py)
  const int32_t* observation_offsets;
  bool observe_select, observe_answer;
  bool observe_required, observe_no_loop;
  const int32_t* loop_label_groups;  // [relation label rows, 2], equal token labels
  const int32_t* selection_tokens;
  int32_t selection_length;
  const int32_t* answer_tokens;
  int32_t answer_length;
  bool audit;
  int32_t menu;
  int32_t audit_entities, audit_relations;
  int32_t observe_terminal;  // display only after commitment and the answer-label-row cap
  const int32_t* audit_tokens;
  const int32_t* audit_offsets;
  const int32_t* audit_command_tokens;
  const int32_t* audit_command_offsets;
  const int32_t* audit_scores;  // immutable [slots, 64, 2] (relation id, equal-score rank)
  const int32_t* audit_primary_rows;  // [relations], -1 when the canonical label was excluded
  const int32_t* audit_default_alias_ranks;  // [relation rows], lexical fallback without a question
  const int32_t* audit_alias_ranks;  // immutable [slots, relation rows], question-scored fallback
  const uint8_t* audit_alias_active;
  int32_t relation_mode;   // 0: frontier, 1: vocabulary, 2: successor
  const uint32_t* relation_successors;  // [base + versions, relation_count, relation_words]
  int32_t relation_count;
  int32_t relation_words;  // zero unless successor mode
  MemoryLayout layout;  // immutable allocation strides and optional offsets
  // False: a subject the derivation binding restricts cannot escape (the oracle's
  // bound_escapes).
  bool bound_escapes;
  // Path mode: a path resolves to what its best start entity reaches (the oracle's
  // committed_paths; see Memory::frontier_ranks, ranked_starts, next_frontier).
  bool committed_paths;
  // The sitelink rank (the oracle's start_rank "sitelinks"): best-connected order puts more
  // sitelinks first, then more adjacency entries (see rank_key).
  bool sitelinks_rank;
  const int32_t* entity_sitelinks;  // [entities]: each entity's sitelink count, if ranked
  // Unquoted starts (the oracle's unquoted_starts): an unbound subject cannot name a label
  // that starts with a quotation mark. unquoted_rows[r] counts the rows before r that it
  // may name; the root mask and the wide masks already hold only their tokens.
  bool unquoted_starts;
  const int32_t* unquoted_rows;  // [label index rows + 1], if unquoted_starts
  // Expressions (the oracle's expressions): a span may be a date comparison. The operators
  // (each after the label prefix), operator_lcp[k * kMaxOperators + j] the leading bytes
  // operators k and j share, and the operand separator. The date table: the date entities'
  // positions, sorted, and their dates' interval bounds (xgraph.index.date_lo, date_hi).
  bool expressions;
  Delimiter operators[kMaxOperators];
  int32_t operator_count;
  int32_t operator_lcp[kMaxOperators * kMaxOperators];
  Delimiter operand_separator;
  const int32_t* date_entities;
  const int32_t* date_lo;
  const int32_t* date_hi;
  const int32_t* date_keys;  // their dates, packed (xgraph.index.pack_date)
  int32_t date_count;
  // A value comparison's verdict entities (positions: yes, no), whose rows no unbound
  // subject names (they are left out of unquoted_rows); without them the third operator,
  // the value comparison's, is not offered.
  const int32_t* verdict_entities;
  // Residual-goal STOP: shared normalized start-label words, then immutable request
  // tables by slot. Each relation has head membership and a uint64 word-coverage mask.
  const int64_t* stop_label_offsets;
  const uint8_t* stop_label_words;
  const int32_t* stop_relations;  // [slots, stop_relation_count, 3]
  int32_t stop_relation_count;
  const uint8_t* stop_words;     // [slots, kMaxStopBytes]
  const int32_t* stop_offsets;   // [slots, kMaxStopWords + 2]: count, then word offsets
};

// Only the view changes. Allocation strides and menu/history offsets stay fixed.
__device__ Tables request_tables(Tables t, int32_t slot) {
  if (!t.request_path_options) return t;
  const int32_t* p = t.request_path_options + static_cast<int64_t>(slot) * 10;
  if (p[0] < 0) {
    // A mixed server's default is its first-call view, before any override.
    if (t.menu || t.observe) t.observe_terminal = 0;
    return t;
  }
  t.menu = p[0]; t.observe = p[1];
  t.observe_select = p[2]; t.observe_answer = p[3]; t.observe_required = p[4];
  t.observe_no_loop = p[5]; t.observe_width = p[6]; t.observe_defer = p[7];
  t.menu_close_next = p[8];
  t.observe_terminal = p[9];
  if (t.observe_terminal && t.request_terminal_tokens) {
    t.audit_tokens = t.request_terminal_tokens;
    t.audit_offsets = t.request_terminal_offsets;
  }
  if (t.menu && !t.observe && t.request_menu_tokens) {
    t.observation_tokens = t.request_menu_tokens;
    t.observation_offsets = t.request_menu_offsets;
  }
  return t;
}

__device__ int32_t token_at(const Rows& r, int32_t row, int32_t depth) {
  const int64_t at = r.offsets[row] + depth;
  return at < r.offsets[row + 1] ? r.tokens[at] : -1;
}

// The first row in [lo, hi) whose token at depth is >= value (> value if upper).
__device__ int32_t search(const Rows& r, int32_t lo, int32_t hi, int32_t depth, int32_t value,
                          bool upper) {
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    const int32_t token = token_at(r, mid, depth);
    if (upper ? token <= value : token < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The first index in [lo, hi) of a sorted array whose value is >= value (> if upper).
__device__ int64_t bound(const int32_t* a, int64_t lo, int64_t hi, int32_t value, bool upper) {
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (upper ? a[mid] <= value : a[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The byte position a delimiter reaches when the token follows pos matched bytes, or -1.
__device__ int32_t delimiter_step(const Delimiter& d, int32_t pos, int32_t token) {
  for (int32_t i = d.offsets[pos]; i < d.offsets[pos + 1]; ++i) {
    if (d.tokens[i] == token) return d.next_pos[i];
  }
  return -1;
}

// The edges of one subject as a sequence reads them: edges [begin, end) of relations and
// objects, from the base adjacency or, for a subject the sequence's KG version edits, from
// the version's table. Every read of the adjacency goes through here.
struct Edges {
  const int32_t* relations;
  const int32_t* objects;
  int64_t begin, end;
};

__device__ Edges edges_of(const Tables& t, int32_t version, int32_t entity) {
  if (version > 0) {
    const Versions& v = t.versions;
    int64_t a = v.offsets[version - 1], b = v.offsets[version];
    const int64_t stop = b;
    while (a < b) {
      const int64_t m = a + (b - a) / 2;
      if (v.subjects[m] < entity) a = m + 1; else b = m;
    }
    if (a < stop && v.subjects[a] == entity) {
      return Edges{v.edge_relations, v.edge_objects, v.adjacency[a], v.adjacency[a + 1]};
    }
  }
  return Edges{t.edge_relations, t.edge_objects, t.adjacency[entity], t.adjacency[entity + 1]};
}

__device__ bool has_triples(const Tables& t, const Memory& f, int32_t entity) {
  const Edges e = edges_of(t, f.version, entity);
  return e.end > e.begin;
}

// The edges of a subject with one relation: one object slice, [first, last) of the edges.
__device__ Edges object_slice(const Tables& t, const Memory& f, int32_t entity,
                              int32_t relation, int64_t& first, int64_t& last) {
  const Edges e = edges_of(t, f.version, entity);
  first = bound(e.relations, e.begin, e.end, relation, false);
  last = bound(e.relations, first, e.end, relation, true);
  return e;
}

// Some label index row in [lo, hi) names an entity with a triple.
__device__ bool any_subject(const Tables& t, const Memory& f, int32_t lo, int32_t hi) {
  for (int32_t row = lo; row < hi; ++row) {
    if (has_triples(t, f, t.entities.owner[row])) return true;
  }
  return false;
}

// Some candidate row of the state's segment lies in [lo, hi).
// Derivation mode with a fact span closed: spans may only name the rows it reached.
__device__ bool derivation_bound(const Tables& t, const Memory& f) {
  return t.derivation && *f.reached_count > 0;
}

// Unquoted starts: some row in [lo, hi) of the label index that an unbound subject may name.
__device__ bool unquoted_in(const Tables& t, int32_t lo, int32_t hi) {
  return !t.unquoted_starts || t.unquoted_rows[hi] > t.unquoted_rows[lo];
}

// The escape may start here: always with bound escapes, else not in a bound subject.
__device__ bool may_escape(const Tables& t, const Memory& f, const State& s) {
  if (s.selecting || ((t.observe_answer || t.audit || t.observe_terminal) && s.last.kind == kPathSpan)) return false;
  return t.bound_escapes || !(s.segment == kSubject && derivation_bound(t, f));
}

// The first reached row at or after lo (reached rows are sorted), or the count.
__device__ int32_t reached_from(const Memory& f, int32_t lo) {
  int32_t a = 0, b = *f.reached_count;
  while (a < b) {
    const int32_t m = a + (b - a) / 2;
    if (f.reached[m] < lo) a = m + 1; else b = m;
  }
  return a;
}

__device__ bool reached_in(const Memory& f, int32_t lo, int32_t hi) {
  const int32_t i = reached_from(f, lo);
  return i < *f.reached_count && f.reached[i] < hi;
}

// A path's frontier in the memory: operand 0, the path's (or a comparison's operand A's),
// operand 1, a comparison's operand B's.
struct Frontier {
  int32_t* rows;
  int32_t* ranks;
  int32_t* count;
};

__device__ Frontier frontier_of(const Memory& f, int32_t operand) {
  return operand == 0 ? Frontier{f.frontier, f.frontier_ranks, f.frontier_count}
                      : Frontier{f.frontier_b, f.frontier_b_ranks, f.frontier_b_count};
}

// The delimiter the separator branch matches: the operand separator after a comparison's
// operand A, the separator everywhere else.
__device__ const Delimiter& separator_of(const Tables& t, const State& s) {
  return s.op != 0 && s.segment == kSubject && s.operand == 0 ? t.operand_separator
                                                              : t.separator;
}

// Expressions: the entity's position in the date table, or -1 if it is no date.
__device__ int32_t date_index(const Tables& t, int32_t entity) {
  int32_t a = 0, b = t.date_count;
  while (a < b) {
    const int32_t m = a + (b - a) / 2;
    if (t.date_entities[m] < entity) a = m + 1; else b = m;
  }
  return a < t.date_count && t.date_entities[a] == entity ? a : -1;
}

// Expressions: whether the entity is a date, and if so its interval bounds.
__device__ bool date_of(const Tables& t, int32_t entity, int32_t& lo, int32_t& hi) {
  const int32_t at = date_index(t, entity);
  if (at < 0) return false;
  lo = t.date_lo[at];
  hi = t.date_hi[at];
  return true;
}

// Some entity of the frontier has the relation.
__device__ bool frontier_has(const Tables& t, const Memory& f, const Frontier& from,
                             int32_t relation) {
  for (int32_t i = 0; i < *from.count; ++i) {
    const Edges e = edges_of(t, f.version, t.entities.owner[from.rows[i]]);
    const int64_t at = bound(e.relations, e.begin, e.end, relation, false);
    if (at < e.end && e.relations[at] == relation) return true;
  }
  return false;
}

// Expressions: an operator may start here, at the start of an unbound span's subject
// segment (no label token yet: any token either extends the label or kills it).
__device__ bool may_operate(const Tables& t, const Memory& f, const State& s) {
  return t.expressions && s.segment == kSubject && s.op == 0 && s.depth == 0 && s.lo >= 0 &&
         s.op_pos == 0 && !derivation_bound(t, f);
}

// The operator branch after the token: its new byte position (and the operator, in kind),
// or -1 when the token does not continue it.
__device__ int32_t operator_step(const Tables& t, const Memory& f, const State& s,
                                 int32_t token, int32_t& kind) {
  if (s.op_pos == 0 && !may_operate(t, f, s)) return -1;
  for (int32_t j = 0; j < t.operator_count; ++j) {
    if (s.op_pos > 0 && t.operator_lcp[s.op_kind * kMaxOperators + j] < s.op_pos) continue;
    const int32_t pos = delimiter_step(t.operators[j], s.op_pos, token);
    if (pos >= 0) {
      kind = j;
      return pos;
    }
  }
  return -1;
}

// The subject rows of a relation or object segment: the subject range, or in path mode
// the frontier (the memory's reached rows) of a relation segment.
__device__ bool on_frontier(const Tables& t, const State& s) {
  return t.paths && t.relation_mode == 0 && s.segment == kRelation;
}

// Weaker relation supports never read an entity frontier. The first successor
// segment admits the full vocabulary; later ones read the preceding label's union.
__device__ bool supported_relation(const Tables& t, const Memory& f, const State& s,
                                   int32_t relation) {
  return t.relation_mode == 1 || s.hops == 1 ||
         (f.relation_support[relation >> 5] & (1u << (relation & 31))) != 0;
}

__device__ const uint32_t* successors_of(const Tables& t, const Memory& f, int32_t relation) {
  return t.relation_successors +
         (static_cast<int64_t>(f.version) * t.relation_count + relation) * t.relation_words;
}

__device__ bool relation_goes_on(const Tables& t, const Memory& f, const State& s,
                                 int32_t lo, int32_t hi) {
  if (t.relation_mode == 1) return t.relations.count > 0;
  for (int32_t row = lo; row < hi; ++row) {
    const int32_t relation = t.relations.owner[row];
    if (!supported_relation(t, f, s, relation)) continue;
    const uint32_t* next = successors_of(t, f, relation);
    for (int32_t w = 0; w < t.relation_words; ++w) if (next[w]) return true;
  }
  return false;
}

// Build the next support before replacing the old one, so a shared label keeps
// exactly its compatible ids. No cap on aliases and no truncation to MAX_REACHED.
__device__ void consume_relation(const Tables& t, const Memory& f, const State& s,
                                  int32_t lo, int32_t hi) {
  for (int32_t w = 0; w < t.relation_words; ++w) {
    uint32_t next = 0;
    for (int32_t row = lo; row < hi; ++row) {
      const int32_t relation = t.relations.owner[row];
      if (supported_relation(t, f, s, relation)) next |= successors_of(t, f, relation)[w];
    }
    f.next_support[w] = next;
  }
  for (int32_t w = 0; w < t.relation_words; ++w) f.relation_support[w] = f.next_support[w];
}
__device__ int32_t subject_count(const Tables& t, const Memory& f, const State& s) {
  return on_frontier(t, s) ? *f.frontier_count : s.subj_hi - s.subj_lo;
}
__device__ int32_t subject_row(const Tables& t, const Memory& f, const State& s, int32_t i) {
  return on_frontier(t, s) ? f.frontier[i] : s.subj_lo + i;
}

__device__ uint32_t frontier_hash(const int32_t* rows, int32_t n) {
  uint32_t hash = 0;
  for (int32_t i = 0; i < n; ++i) {
    uint32_t x = static_cast<uint32_t>(rows[i]) + 1u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    hash += x ^ (x >> 16);
  }
  return hash;
}

__device__ bool loop_label_allowed(const Tables& t, const Memory& f, const State& s,
                                    int32_t row) {
  if (!t.observe_no_loop || s.observed || s.loop_count == 0) return true;
  const int32_t n = *f.frontier_count;
  const uint32_t hash = frontier_hash(f.frontier, n);
  const int32_t lo = t.loop_label_groups[row * 2], hi = t.loop_label_groups[row * 2 + 1];
  for (int32_t i = 0; i < s.loop_count; ++i) {
    const int32_t* entry = f.loop_rows + i * kLoopEntryInts;
    if (entry[0] != n || static_cast<uint32_t>(entry[1]) != hash) continue;
    bool equal = true;
    for (int32_t j = 0; j < n && equal; ++j) {
      const int64_t at = bound(entry + 4, 0, n, f.frontier[j], false);
      equal = at < n && entry[4 + at] == f.frontier[j];
    }
    if (!equal) continue;  // hash collisions cannot affect admission
    for (int32_t r = lo; r < hi; ++r) {
      const int32_t relation = t.relations.owner[r];
      if (!frontier_has(t, f, frontier_of(f, 0), relation)) continue;
      for (int32_t old = entry[2]; old < entry[3]; ++old)
        if (t.relations.owner[old] == relation) return false;
    }
  }
  return true;
}

__device__ bool candidate_in(const Tables& t, const Memory& f, const State& s, int32_t lo,
                             int32_t hi) {
  if (lo >= hi) return false;
  if (s.segment == kSubject) {
    if (s.selecting) {
      for (int32_t i = 0; i < *f.frontier_b_count; ++i) {
        const int32_t owner = t.entities.owner[f.frontier_b[i]];
        const int64_t end = t.entity_row_offsets[owner + 1];
        const int64_t at = bound(t.entity_rows, t.entity_row_offsets[owner], end, lo, false);
        if (at < end && t.entity_rows[at] < hi) return true;
      }
      return false;
    }
    return derivation_bound(t, f) ? reached_in(f, lo, hi) : unquoted_in(t, lo, hi);
  }
  if (s.segment == kRelation && t.relation_mode != 0) {
    for (int32_t row = lo; row < hi; ++row) {
      if (supported_relation(t, f, s, t.relations.owner[row])) return true;
    }
    return false;
  }
  if (t.observe_no_loop && s.segment == kRelation && !s.observed) {
    for (int32_t row = lo; row < hi; ++row)
      if (loop_label_allowed(t, f, s, row) &&
          frontier_has(t, f, frontier_of(f, 0), t.relations.owner[row])) return true;
    return false;
  }
  for (int32_t i = 0; i < subject_count(t, f, s); ++i) {
    const int32_t entity = t.entities.owner[subject_row(t, f, s, i)];
    if (s.segment == kRelation) {
      // The subject's relations, and their relation index rows within [lo, hi). Edges
      // are sorted by relation: jump from one relation's run to the next.
      const Edges edges = edges_of(t, f.version, entity);
      const int64_t end = edges.end;
      int64_t e = edges.begin;
      while (e < end) {
        const int32_t relation = edges.relations[e];
        const int64_t first = bound(t.relation_rows, t.relation_row_offsets[relation],
                                    t.relation_row_offsets[relation + 1], lo, false);
        if (first < t.relation_row_offsets[relation + 1] && t.relation_rows[first] < hi &&
            (s.op == 0 || frontier_has(t, f, frontier_of(f, 1), relation))) {
          return true;  // a comparison's: operand B's frontier has the relation too
        }
        int64_t a = e + 1, b = end;
        while (a < b) {
          const int64_t m = a + (b - a) / 2;
          if (edges.relations[m] <= relation) a = m + 1; else b = m;
        }
        e = a;
      }
    } else {
      // The object slices of each candidate pair, within [lo, hi).
      for (int32_t rel = s.rel_lo; rel < s.rel_hi; ++rel) {
        int64_t first = 0, last = 0;
        const Edges e = object_slice(t, f, entity, t.relations.owner[rel], first, last);
        const int64_t at = bound(e.objects, first, last, lo, false);
        if (at < last && e.objects[at] < hi) return true;
      }
    }
  }
  return false;
}

__device__ State opened_segment(const State& s, int32_t segment, int32_t rows) {
  State next = s;
  next.segment = segment;
  next.depth = 0;
  next.lo = 0;
  next.hi = rows;
  next.close_pos = next.close_lo = next.close_hi = 0;
  next.sep_pos = next.sep_lo = next.sep_hi = 0;
  next.esc_pos = 0;
  next.op_pos = next.op_kind = 0;
  next.observe_pos = next.observe_lo = next.observe_hi = next.observe_depth = 0;
  next.observe_offset = next.observe_length = next.observed = 0;
  next.observe_count = next.observe_more = 0;
  next.select_pos = next.selecting = next.selected = 0;
  next.select_segment = next.select_lo = next.select_hi = next.select_depth = 0;
  next.defer_pos = next.decision_phase = next.decision_pos = next.decision_choices = 0;
  return next;
}

__device__ bool closed_before(const Memory& f, int32_t subj, int32_t rel, int32_t obj) {
  for (int32_t i = 0; i < *f.count; ++i) {
    const int32_t* row = f.rows + i * 3;
    if (row[0] == subj && row[1] == rel && row[2] == obj) return true;
  }
  return false;
}

__device__ int64_t entity_degree(const Tables& t, const Memory& f, int32_t row) {
  const Edges e = edges_of(t, f.version, t.entities.owner[row]);
  return e.end - e.begin;
}

// A row's entity in best-connected order, greater first: its adjacency entries (its degree,
// in the KG version the sequence reads), after its sitelink count with the sitelink rank.
struct EntityRank {
  int32_t sitelinks;
  int64_t degree;
};
__device__ EntityRank rank_key(const Tables& t, int32_t row, int64_t degree) {
  return EntityRank{t.sitelinks_rank ? t.entity_sitelinks[t.entities.owner[row]] : 0, degree};
}
__device__ bool rank_greater(const EntityRank& a, const EntityRank& b) {
  return a.sitelinks != b.sitelinks ? a.sitelinks > b.sitelinks : a.degree > b.degree;
}
__device__ bool rank_equal(const EntityRank& a, const EntityRank& b) {
  return a.sitelinks == b.sitelinks && a.degree == b.degree;
}

// Derivation mode: the row a span binds to, the best-connected (by rank_key, then the first
// row) among the complete rows [lo, hi) the span may name; a fact span's subject needs
// triples (-1 if none has any), an entity span's not.
__device__ int32_t best_row(const Tables& t, const Memory& f, int32_t lo, int32_t hi,
                            bool needs_triples) {
  int32_t best = -1;
  EntityRank most{-1, -1};
  const bool limited = derivation_bound(t, f);
  int32_t i = limited ? reached_from(f, lo) : 0;
  for (int32_t row = limited ? (i < *f.reached_count ? f.reached[i] : hi) : lo; row < hi;) {
    const int64_t degree = entity_degree(t, f, row);
    const EntityRank key = rank_key(t, row, degree);
    if ((degree > 0 || !needs_triples) && rank_greater(key, most)) {
      best = row;
      most = key;
    }
    if (limited) {
      ++i;
      row = i < *f.reached_count ? f.reached[i] : hi;
    } else {
      ++row;
    }
  }
  return best;
}

__device__ int32_t best_subject(const Tables& t, const Memory& f, int32_t lo, int32_t hi) {
  return best_row(t, f, lo, hi, true);
}

// Insert a row into a sorted list of at most kMaxReached rows, keeping the smallest.
__device__ void insert_sorted(int32_t* list, int32_t& n, int32_t row) {
  int32_t at = n;
  while (at > 0 && list[at - 1] > row) --at;
  if ((at > 0 && list[at - 1] == row) || at == kMaxReached) return;
  for (int32_t k = (n < kMaxReached ? n : kMaxReached - 1); k > at; --k) list[k] = list[k - 1];
  list[at] = row;
  if (n < kMaxReached) ++n;
}

__device__ void insert_reached(const Memory& f, int32_t& n, int32_t row) {
  insert_sorted(f.reached, n, row);
}

// Committed paths: insert an entity (its first label row) reached from a start of this rank
// into a list of at most kMaxReached, ordered by rank and then row, keeping the first. An
// entity already in the list keeps the lesser of its two ranks. An entity that fell off the
// end cannot come back with a greater rank, since the list's last entry only moves forward.
__device__ void insert_ranked(int32_t* rows, int32_t* ranks, int32_t& n, int32_t row,
                              int32_t rank) {
  for (int32_t i = 0; i < n; ++i) {
    if (rows[i] != row) continue;
    if (ranks[i] <= rank) return;
    for (int32_t k = i; k + 1 < n; ++k) {
      rows[k] = rows[k + 1];
      ranks[k] = ranks[k + 1];
    }
    --n;
    break;
  }
  int32_t at = n;
  while (at > 0 && (ranks[at - 1] > rank || (ranks[at - 1] == rank && rows[at - 1] > row))) --at;
  if (at == kMaxReached) return;
  for (int32_t k = (n < kMaxReached ? n : kMaxReached - 1); k > at; --k) {
    rows[k] = rows[k - 1];
    ranks[k] = ranks[k - 1];
  }
  rows[at] = row;
  ranks[at] = rank;
  if (n < kMaxReached) ++n;
}

// Committed paths: the first frontier, every entity the complete rows [lo, hi) may name that
// has triples, ranked from 0 in best-connected order (rank_key, then the row it is named
// by, as best_row), the first kMaxReached of them, into out; returns how many.
__device__ int32_t ranked_starts(const Tables& t, const Memory& f, int32_t lo, int32_t hi,
                                 const Frontier& out) {
  const bool bound = derivation_bound(t, f);
  int32_t n = 0;  // out.rows holds the rows the starts are named by, best first
  for (int32_t row = lo; row < hi; ++row) {
    const int32_t owner = t.entities.owner[row];
    if ((bound && !reached_in(f, row, row + 1)) || !has_triples(t, f, owner)) continue;
    bool named = false;  // by an earlier row, which ranks it better
    for (int32_t i = 0; i < n && !named; ++i) named = t.entities.owner[out.rows[i]] == owner;
    if (named) continue;
    const EntityRank key = rank_key(t, row, entity_degree(t, f, row));
    int32_t at = n;
    while (at > 0 &&
           rank_greater(key, rank_key(t, out.rows[at - 1], entity_degree(t, f, out.rows[at - 1])))) {
      --at;
    }
    if (at == kMaxReached) continue;
    for (int32_t k = (n < kMaxReached ? n : kMaxReached - 1); k > at; --k) {
      out.rows[k] = out.rows[k - 1];
    }
    out.rows[at] = row;
    if (n < kMaxReached) ++n;
  }
  for (int32_t i = 0; i < n; ++i) {
    out.rows[i] = t.entity_rows[t.entity_row_offsets[t.entities.owner[out.rows[i]]]];
    out.ranks[i] = i;
  }
  return n;
}

// Derivation mode: the label rows of the entities a fact span reaches, the objects in
// [lo, hi) (rows of the written object label) of its subject rows under its relation
// rows; every label row of those entities, the first kMaxReached of them in row order.
__device__ void record_reached(const Tables& t, const Memory& f, const State& s, int32_t lo,
                               int32_t hi) {
  int32_t n = 0;
  for (int32_t subj = s.subj_lo; subj < s.subj_hi; ++subj) {
    const int32_t entity = t.entities.owner[subj];
    for (int32_t rel = s.rel_lo; rel < s.rel_hi; ++rel) {
      int64_t first = 0, last = 0;
      const Edges edges = object_slice(t, f, entity, t.relations.owner[rel], first, last);
      for (int64_t e = bound(edges.objects, first, last, lo, false); e < last; ++e) {
        if (edges.objects[e] >= hi) break;
        const int32_t reached = t.entities.owner[edges.objects[e]];
        for (int64_t r = t.entity_row_offsets[reached]; r < t.entity_row_offsets[reached + 1];
             ++r) {
          insert_reached(f, n, t.entity_rows[r]);
        }
      }
    }
  }
  *f.reached_count = n;
}

// Path mode: the next frontier, the objects of a frontier's entities under the relation
// rows [rel_lo, rel_hi), as each entity's first label row, the first kMaxReached in row
// order; with committed paths each with the least rank of the entities it is reached from,
// the first kMaxReached by rank and then row (see insert_ranked). Returns how many.
__device__ int32_t next_frontier(const Tables& t, const Memory& f, const Frontier& from,
                                 int32_t rel_lo, int32_t rel_hi, int32_t* out,
                                 int32_t* out_ranks) {
  int32_t n = 0;
  for (int32_t i = 0; i < *from.count; ++i) {
    const int32_t entity = t.entities.owner[from.rows[i]];
    for (int32_t rel = rel_lo; rel < rel_hi; ++rel) {
      int64_t first = 0, last = 0;
      const Edges edges = object_slice(t, f, entity, t.relations.owner[rel], first, last);
      for (int64_t e = first; e < last; ++e) {
        const int32_t owner = t.entities.owner[edges.objects[e]];
        const int32_t row = t.entity_rows[t.entity_row_offsets[owner]];
        if (t.committed_paths) {
          insert_ranked(out, out_ranks, n, row, from.ranks[i]);
        } else {
          insert_sorted(out, n, row);
        }
      }
    }
  }
  return n;
}

// A selected boundary has already executed its relation. Never execute it again.
__device__ int32_t path_frontier(const Tables& t, const Memory& f, const State& s,
                                 int32_t lo, int32_t hi, int32_t* out, int32_t* ranks) {
  if (!s.selected) return next_frontier(t, f, frontier_of(f, 0), lo, hi, out, ranks);
  const int32_t n = *f.frontier_b_count;
  for (int32_t i = 0; i < n; ++i) {
    out[i] = f.frontier_b[i];
    ranks[i] = f.frontier_b_ranks[i];
  }
  return n;
}

// Path mode: some entity of a frontier's next frontier has triples, so the path can go on;
// with dated, some entity of it is a date instead (where a comparison may close).
__device__ bool frontier_goes_on(const Tables& t, const Memory& f, const Frontier& from,
                                 int32_t rel_lo, int32_t rel_hi, bool dated = false) {
  int32_t next[kMaxReached], ranks[kMaxReached];
  const int32_t n = next_frontier(t, f, from, rel_lo, rel_hi, next, ranks);
  for (int32_t i = 0; i < n; ++i) {
    const int32_t entity = t.entities.owner[next[i]];
    int32_t lo = 0, hi = 0;
    if (dated ? date_of(t, entity, lo, hi) : has_triples(t, f, entity)) return true;
  }
  return false;
}

__device__ uint64_t stop_used(const State& s) {
  return static_cast<uint32_t>(s.stop_used_lo) |
         (static_cast<uint64_t>(static_cast<uint32_t>(s.stop_used_hi)) << 32);
}

__device__ void set_stop_used(State& s, uint64_t mask) {
  s.stop_used_lo = static_cast<int32_t>(mask);
  s.stop_used_hi = static_cast<int32_t>(mask >> 32);
}

__device__ const int32_t* stop_relation(const Tables& t, const Memory& f, int32_t relation) {
  return t.stop_relations + (static_cast<int64_t>(f.slot) * t.stop_relation_count + relation) * 3;
}

__device__ uint64_t stop_coverage(const int32_t* relation) {
  return static_cast<uint32_t>(relation[1]) |
         (static_cast<uint64_t>(static_cast<uint32_t>(relation[2])) << 32);
}

// All indexed relations sharing the written label contribute their aliases and stems,
// including those this frontier lacks, exactly as stopping.residual_goal_stop reads them.
__device__ uint64_t stop_consumed(const Tables& t, const Memory& f, int32_t lo, int32_t hi) {
  uint64_t used = 0;
  for (int32_t row = lo; row < hi; ++row) {
    used |= stop_coverage(stop_relation(t, f, t.relations.owner[row]));
  }
  return used;
}

__device__ uint64_t stop_start_words(const Tables& t, const Memory& f, int32_t row) {
  const int32_t* offsets = t.stop_offsets + static_cast<int64_t>(f.slot) * (kMaxStopWords + 2);
  const uint8_t* words = t.stop_words + static_cast<int64_t>(f.slot) * kMaxStopBytes;
  uint64_t used = 0;
  int64_t at = t.stop_label_offsets[row], end = t.stop_label_offsets[row + 1];
  while (at < end) {
    int64_t stop = at;
    while (stop < end && t.stop_label_words[stop] != 0) ++stop;
    for (int32_t k = 0; k < offsets[0]; ++k) {
      const int32_t first = offsets[k + 1], last = offsets[k + 2];
      if (stop - at != last - first) continue;
      int32_t j = 0;
      while (j < last - first && words[first + j] == t.stop_label_words[at + j]) ++j;
      if (j == last - first) used |= uint64_t{1} << k;
    }
    at = stop + 1;
  }
  return used;
}

__device__ bool residual_goal_stop(const Tables& t, const Memory& f, const State& s,
                                   int32_t hi) {
  if (!f.stop || s.op != 0 || !on_frontier(t, s)) return false;
  bool head = false;
  for (int32_t row = s.lo; row < hi; ++row) {
    head |= stop_relation(t, f, t.relations.owner[row])[0] != 0;
  }
  if (!head) return false;
  const uint64_t used = stop_used(s) | stop_consumed(t, f, s.lo, hi);
  int32_t next[kMaxReached], ranks[kMaxReached];
  const int32_t n = path_frontier(t, f, s, s.lo, hi, next, ranks);
  for (int32_t i = 0; i < n; ++i) {  // the whole capped frontier, before start commitment
    const Edges edges = edges_of(t, f.version, t.entities.owner[next[i]]);
    for (int64_t e = edges.begin; e < edges.end;) {
      const int32_t relation = edges.relations[e];
      if ((stop_coverage(stop_relation(t, f, relation)) & ~used) != 0) return false;
      e = bound(edges.relations, e, edges.end, relation, true);
    }
  }
  return true;
}

// A value comparison (the oracle's _same) after the relation rows [rel_lo, rel_hi): each
// operand's committed values (its next frontier's entities of the least rank, which come
// first), into a and b their start rows, and the verdict: 1 yes (a shared entity, or an
// equal date), 0 no, -1 where it cannot close (an empty reach, or values of two kinds:
// entities on one side only).
__device__ int32_t value_verdict(const Tables& t, const Memory& f, int32_t rel_lo,
                                 int32_t rel_hi, int32_t& a, int32_t& b) {
  int32_t next_a[kMaxReached], ranks_a[kMaxReached], next_b[kMaxReached], ranks_b[kMaxReached];
  int32_t na = next_frontier(t, f, frontier_of(f, 0), rel_lo, rel_hi, next_a, ranks_a);
  int32_t nb = next_frontier(t, f, frontier_of(f, 1), rel_lo, rel_hi, next_b, ranks_b);
  if (na == 0 || nb == 0) return -1;
  int32_t n = 0;
  while (n < na && ranks_a[n] == ranks_a[0]) ++n;
  na = n;
  n = 0;
  while (n < nb && ranks_b[n] == ranks_b[0]) ++n;
  nb = n;
  bool entities_a = false, entities_b = false;
  for (int32_t i = 0; i < na; ++i) entities_a |= date_index(t, t.entities.owner[next_a[i]]) < 0;
  for (int32_t j = 0; j < nb; ++j) entities_b |= date_index(t, t.entities.owner[next_b[j]]) < 0;
  if (entities_a != entities_b) return -1;
  a = f.starts[ranks_a[0]];
  b = f.starts[kMaxReached + ranks_b[0]];
  for (int32_t i = 0; i < na; ++i) {
    const int32_t date_a = date_index(t, t.entities.owner[next_a[i]]);
    for (int32_t j = 0; j < nb; ++j) {
      const int32_t date_b = date_index(t, t.entities.owner[next_b[j]]);
      if (date_a < 0 && date_b < 0 && next_a[i] == next_b[j]) return 1;  // a shared entity
      if (date_a >= 0 && date_b >= 0 && t.date_keys[date_a] == t.date_keys[date_b]) return 1;
    }
  }
  return 0;
}

// Expressions: in a frontier ordered by rank (committed paths), the least rank of a date
// entity and the earliest date of that rank's dates (least lower bound, then least upper),
// as the oracle's _compare reads them; rank is -1 if none is a date.
__device__ void earliest_dated(const Tables& t, const int32_t* rows, const int32_t* ranks,
                               int32_t n, int32_t& rank, int32_t& lo, int32_t& hi) {
  rank = -1;
  for (int32_t i = 0; i < n; ++i) {
    int32_t date_lo = 0, date_hi = 0;
    if (!date_of(t, t.entities.owner[rows[i]], date_lo, date_hi)) continue;
    if (rank < 0) {
      rank = ranks[i];
      lo = date_lo;
      hi = date_hi;
    } else if (ranks[i] == rank && (date_lo < lo || (date_lo == lo && date_hi < hi))) {
      lo = date_lo;
      hi = date_hi;
    }
  }
}

__device__ void close_span(const Tables& t, State& s, const Memory& f, int32_t kind, int32_t lo,
                           int32_t hi) {
  if (kind == kFactSpan && f.record && t.derivation) record_reached(t, f, s, lo, hi);
  if (kind == kFactSpan && f.record && !t.repeats && *f.count < kMaxFacts) {
    int32_t* row = f.rows + *f.count * 3;
    row[0] = s.subj_key;
    row[1] = s.rel_lo;
    row[2] = lo;
    *f.count += 1;
  }
  State closed{};
  closed.phase = kFree;
  closed.answer_pos = (t.observe_answer || t.audit || t.observe_terminal) && kind == kPathSpan ? 1 : 0;
  closed.answer_complete = t.end_after_answer && kind == kEntitySpan &&
      (s.last.kind == kPathSpan || s.last.kind == kRelationPlan || s.last.kind == kComparison);
  closed.last = kind == kFactSpan ? LastSpan{kind, lo, hi, s.subj_lo, s.subj_hi, s.rel_lo, s.rel_hi}
                                  : LastSpan{kind, lo, hi, 0, 0, 0, 0};
  if (kind == kEntitySpan && t.verdict_entities && !derivation_bound(t, f))
    closed.last.subj_lo = -1;
  s = closed;
}

// Where a span state's complete rows end, and whether the closer and the separator
// may start after them (the oracle's rules).
struct Options {
  int32_t complete_hi;
  bool closes;
  bool separates;
  bool stop;
};

__device__ bool observes_label(const Tables& t, const State& s) {
  const bool answer = s.last.kind == kPathSpan || s.last.kind == kComparison ||
                      s.last.kind == kRelationPlan;
  return (t.observe > 0 || t.menu > 0) && !s.observed && s.op == 0 &&
         (s.selecting || s.segment == kRelation || (s.segment == kSubject && !answer));
}

__device__ int32_t observation_piece_size(const Tables& t, int32_t piece) {
  return t.observation_offsets[piece + 1] - t.observation_offsets[piece];
}

// Return one payload token without constructing a variable-length token buffer. The
// rows were chosen on the GPU at marker completion; the KG and all pieces stay there.
__device__ int32_t observation_token(const Tables& t, const Memory& f, const State& s) {
  int32_t at = s.observe_offset;
  if (s.observe_count == 0) return t.observation_tokens[t.observation_offsets[2] + at];
  for (int32_t i = 0; i < s.observe_count; ++i) {
    if (i > 0) {
      const int32_t n = observation_piece_size(t, 1);
      if (at < n) return t.observation_tokens[t.observation_offsets[1] + at];
      at -= n;
    }
    const int32_t row = f.observation_rows[i];
    const int64_t start = t.entities.offsets[row];
    const int32_t n = t.entities.offsets[row + 1] - start;
    if (at < n) return t.entities.tokens[start + at];
    at -= n;
  }
  return t.observation_tokens[t.observation_offsets[s.observe_more + 2] + at];
}

__device__ void prepare_step_menu(const Tables& t, const Memory& f, State& s,
                                 const Frontier& from);

__device__ void prepare_observation(const Tables& t, const Memory& f, State& s) {
  int32_t rows[kMaxReached], ranks[kMaxReached], n = 0;
  if (t.observe_no_loop && s.segment == kRelation && !s.selected) {
    // The eight-relation cap makes overflow unreachable for an admitted path.
    if (s.loop_count >= kMaxPathRelations) { s.phase = kError; return; }
    int32_t* entry = f.loop_rows + s.loop_count++ * kLoopEntryInts;
    entry[0] = *f.frontier_count;
    entry[1] = static_cast<int32_t>(frontier_hash(f.frontier, entry[0]));
    entry[2] = s.lo; entry[3] = s.hi;
    int32_t count = 0;
    for (int32_t i = 0; i < *f.frontier_count; ++i)
      insert_sorted(entry + 4, count, f.frontier[i]);
  }
  if (s.selected) {
    n = path_frontier(t, f, s, s.lo, s.hi, rows, ranks);
  } else if (s.segment == kRelation) {
    n = next_frontier(t, f, frontier_of(f, 0), s.lo, s.hi, rows, ranks);
  } else if (t.committed_paths) {
    int32_t count = 0;
    const Frontier scratch{rows, ranks, &count};
    n = ranked_starts(t, f, s.lo, s.hi, scratch);
  } else {
    const bool limited = derivation_bound(t, f);
    for (int32_t row = s.lo; row < s.hi; ++row) {
      const int32_t owner = t.entities.owner[row];
      if ((!limited || reached_in(f, row, row + 1)) && has_triples(t, f, owner)) {
        insert_sorted(rows, n, t.entity_rows[t.entity_row_offsets[owner]]);
      }
    }
  }
  if ((t.observe_select || t.observe_no_loop || t.menu_close_next) && !s.selected) {
    *f.frontier_b_count = n;
    for (int32_t i = 0; i < n; ++i) {
      f.frontier_b[i] = rows[i];
      f.frontier_b_ranks[i] = t.committed_paths ? ranks[i] : 0;
    }
  }
  if (t.menu && !t.menu_close_next) {
    prepare_step_menu(t, f, s, Frontier{rows, ranks, &n});
    if (!t.observe) {
      s.observe_offset = s.observe_length = s.observe_count = s.observe_more = 0;
      s.audit_phase = 6;
      return;
    }
  }
  // Display order is best-connected order over the already capped execution set.
  for (int32_t i = 1; i < n; ++i) {
    const int32_t row = rows[i];
    const EntityRank key = rank_key(t, row, entity_degree(t, f, row));
    int32_t j = i;
    while (j > 0) {
      const int32_t prior = rows[j - 1];
      const EntityRank prior_key = rank_key(t, prior, entity_degree(t, f, prior));
      if (rank_greater(prior_key, key) || (rank_equal(prior_key, key) && prior < row)) break;
      rows[j] = prior;
      --j;
    }
    rows[j] = row;
  }
  s.observe_count = n <= t.observe_width ? n : min(n, t.observe);
  s.observe_more = n - s.observe_count;
  s.observe_offset = 0;
  s.observe_length = n == 0 ? observation_piece_size(t, 2) : 0;
  for (int32_t i = 0; i < s.observe_count; ++i) {
    f.observation_rows[i] = rows[i];
    s.observe_length += t.entities.offsets[rows[i] + 1] - t.entities.offsets[rows[i]];
    if (i > 0) s.observe_length += observation_piece_size(t, 1);
  }
  if (s.observe_more > 0) s.observe_length += observation_piece_size(t, s.observe_more + 2);
}

// Audit payload descriptors: entity row, entity-count + relation row, or -(piece+1).
// Two existing frontiers retain the terminal (B) and penultimate (A) checkpoints.
__device__ int32_t audit_item_size(const Tables& t, int32_t item) {
  if (item < 0) { const int32_t p = -item - 1; return t.audit_offsets[p + 1] - t.audit_offsets[p]; }
  const Rows& rows = item < t.entities.count ? t.entities : t.relations;
  const int32_t row = item < t.entities.count ? item : item - t.entities.count;
  return rows.offsets[row + 1] - rows.offsets[row];
}

__device__ int32_t audit_token(const Tables& t, const Memory& f, const State& s) {
  const int32_t item = f.audit_items[s.audit_item];
  if (item < 0) return t.audit_tokens[t.audit_offsets[-item - 1] + s.audit_offset];
  const Rows& rows = item < t.entities.count ? t.entities : t.relations;
  const int32_t row = item < t.entities.count ? item : item - t.entities.count;
  return rows.tokens[rows.offsets[row] + s.audit_offset];
}

__device__ void audit_add(const Memory& f, State& s, int32_t item) {
  f.audit_items[s.audit_items++] = item;
}

__device__ void audit_list(const Tables& t, const Memory& f, State& s, int32_t heading,
                           const int32_t* rows, int32_t n, int32_t total) {
  if (heading >= 0) audit_add(f, s, -heading - 1);
  for (int32_t i = 0; i < n; ++i) {
    if (i > 0) audit_add(f, s, -5);  // join
    audit_add(f, s, rows[i]);
  }
  if (n == 0) audit_add(f, s, -6);  // prefixed empty
  if (total > n) audit_add(f, s, -(7 + total - n));  // count pieces start at 7
  audit_add(f, s, -7);  // closer
}

__device__ int32_t audit_relation_rank(const Tables& t, const Memory& f, int32_t relation) {
  const int32_t* scores = t.audit_scores + static_cast<int64_t>(f.slot) * kAuditScores * 2;
  for (int32_t i = 0; i < *f.audit_score_count; ++i)
    if (scores[2 * i] == relation) return scores[2 * i + 1];
  return kAuditScores;
}

__device__ int64_t audit_frequency(const Tables& t, const Memory& f, const Frontier& from,
                                   int32_t relation) {
  int64_t count = 0;
  for (int32_t i = 0; i < *from.count; ++i) {
    const Edges edges = edges_of(t, f.version, t.entities.owner[from.rows[i]]);
    count += bound(edges.relations, edges.begin, edges.end, relation, true) -
             bound(edges.relations, edges.begin, edges.end, relation, false);
  }
  return count;
}

__device__ void audit_menu(const Tables& t, const Memory& f, State& s, int32_t heading,
                           const Frontier& from, int32_t limit) {
  int32_t rows[kMaxReached], ranks[kMaxReached], n = 0, total = 0;
  int64_t frequencies[kMaxReached];
  for (int32_t r = 0; r < t.relation_count; ++r) {
    const int64_t frequency = audit_frequency(t, f, from, r);
    if (frequency == 0) continue;
    Memory next = f;
    next.frontier = from.rows; next.frontier_count = from.count;
    State relation = s;
    relation.segment = kRelation; relation.observed = 0;
    const bool guarded = t.menu && t.observe_no_loop;
    int32_t display_row = t.audit_primary_rows[r];
    if (display_row < 0 || (guarded && !loop_label_allowed(t, next, relation, display_row))) {
      const int32_t* priorities = t.audit_alias_active[f.slot]
          ? t.audit_alias_ranks + static_cast<int64_t>(f.slot) * t.relations.count
          : t.audit_default_alias_ranks;
      int32_t best = INT32_MAX;
      display_row = -1;
      for (int64_t at = t.relation_row_offsets[r]; at < t.relation_row_offsets[r + 1]; ++at) {
        const int32_t row = t.relation_rows[at];
        if (priorities[row] < best && (!guarded || loop_label_allowed(t, next, relation, row))) {
          display_row = row; best = priorities[row];
        }
      }
    }
    if (display_row < 0) continue;
    ++total;
    const int32_t rank = audit_relation_rank(t, f, r);
    int32_t at = n;
    while (at > 0 && (ranks[at - 1] > rank ||
           (ranks[at - 1] == rank && frequencies[at - 1] < frequency))) --at;
    if (at == limit) continue;
    for (int32_t k = min(n, limit - 1); k > at; --k) {
      rows[k] = rows[k - 1]; ranks[k] = ranks[k - 1]; frequencies[k] = frequencies[k - 1];
    }
    rows[at] = t.entities.count + display_row;
    ranks[at] = rank; frequencies[at] = frequency;
    n = min(n + 1, limit);
  }
  audit_list(t, f, s, heading, rows, n, total);
}

__device__ void prepare_step_menu(const Tables& t, const Memory& f, State& s,
                                 const Frontier& from) {
  s.audit_item = s.audit_offset = s.audit_items = 0;
  audit_menu(t, f, s, t.observe ? 0 : -1, from, t.menu);
}

__device__ void prepare_audit(const Tables& t, const Memory& f, State& s, bool menus,
                              bool deliverable = false) {
  s.audit_item = s.audit_offset = s.audit_items = 0;
  int32_t rows[kMaxReached], n = 0;
  if (deliverable) {
    // The reached rows already reflect start commitment and answer-row capping.
    // Retain each deliverable entity once, with its first surviving indexed row.
    for (int32_t i = 0; i < *f.reached_count; ++i) {
      const int32_t row = f.reached[i], owner = t.entities.owner[row];
      bool present = false;
      for (int32_t j = 0; j < n; ++j) if (t.entities.owner[rows[j]] == owner) present = true;
      if (!present) rows[n++] = row;
    }
    if (n <= 1) return;  // preserve singleton delivery without a display
  } else {
    n = *f.frontier_b_count;
    for (int32_t i = 0; i < n; ++i) rows[i] = f.frontier_b[i];
  }
  for (int32_t i = 1; i < n; ++i) {
    const int32_t row = rows[i], links = t.entity_sitelinks[t.entities.owner[row]];
    const int64_t degree = entity_degree(t, f, row);
    int32_t j = i;
    while (j > 0) {
      const int32_t prior = rows[j - 1], prior_links = t.entity_sitelinks[t.entities.owner[prior]];
      const int64_t prior_degree = entity_degree(t, f, prior);
      if (prior_links > links || (prior_links == links &&
          (prior_degree > degree || (prior_degree == degree && prior < row)))) break;
      rows[j] = prior; --j;
    }
    rows[j] = row;
  }
  const int32_t shown = min(n, deliverable ? t.observe_terminal : t.audit_entities);
  if (deliverable && t.terminal_order_config) {
    const uint32_t order = t.terminal_order_config[0];
    uint32_t seed = t.terminal_order_config[1];
    for (int32_t i = shown - 1; i > 0; --i) {
      if (order == 0) break;
      if (order == 1 && i < shown / 2) break;
      seed = 1664525u * seed + 1013904223u;
      const int32_t j = order == 1 ? shown - 1 - i : seed % (i + 1);
      const int32_t row = rows[i]; rows[i] = rows[j]; rows[j] = row;
    }
  }
  audit_list(t, f, s, 0, rows, shown, n);
  if (menus) {
    if (t.audit_relations) {
      audit_menu(t, f, s, 1, frontier_of(f, 1), t.audit_relations);
      audit_menu(t, f, s, 2, frontier_of(f, 0), t.audit_relations);
    }
    audit_add(f, s, -4);  // decision prefix
  }
}

__device__ void finish_audit(const Tables& t, const Memory& f, State& s,
                            int32_t lo = 0, int32_t hi = 0) {
  const int32_t action = s.audit_action;
  if (action == 1) {  // SELECT: every matching owner, without the answer-row cap
    int32_t n = 0;
    for (int32_t i = 0; i < *f.frontier_b_count; ++i) {
      const int32_t owner = t.entities.owner[f.frontier_b[i]];
      const int64_t end = t.entity_row_offsets[owner + 1];
      const int64_t at = bound(t.entity_rows, t.entity_row_offsets[owner], end, lo, false);
      if (at < end && t.entity_rows[at] < hi) {
        f.frontier_b[n] = f.frontier_b[i]; f.frontier_b_ranks[n++] = f.frontier_b_ranks[i];
      }
    }
    *f.frontier_b_count = n;
  } else if (action == 4) {  // DROP
    *f.frontier_b_count = *f.frontier_count;
    for (int32_t i = 0; i < *f.frontier_count; ++i) {
      f.frontier_b[i] = f.frontier[i]; f.frontier_b_ranks[i] = f.frontier_ranks[i];
    }
  } else if (action == 2 || action == 3) {  // APPEND / REPLACE (A holds the chosen source)
    int32_t rows[kMaxReached], ranks[kMaxReached];
    const int32_t n = next_frontier(t, f, frontier_of(f, 0), lo, hi, rows, ranks);
    *f.frontier_b_count = n;
    for (int32_t i = 0; i < n; ++i) {
      f.frontier_b[i] = rows[i]; f.frontier_b_ranks[i] = t.committed_paths ? ranks[i] : 0;
    }
  }
  int32_t count = 0;
  for (int32_t i = 0; i < *f.frontier_b_count; ++i) {
    if (t.committed_paths && f.frontier_b_ranks[i] != f.frontier_b_ranks[0]) break;
    const int32_t owner = t.entities.owner[f.frontier_b[i]];
    for (int64_t at = t.entity_row_offsets[owner]; at < t.entity_row_offsets[owner + 1]; ++at)
      insert_reached(f, count, t.entity_rows[at]);
  }
  *f.reached_count = count;
  close_span(t, s, f, kPathSpan, lo, hi);
  if (action >= 2) { s.audit_phase = 5; prepare_audit(t, f, s, false); }
}

__device__ void start_audit(const Tables& t, const Memory& f, State& s, int32_t lo, int32_t hi) {
  int32_t rows[kMaxReached], ranks[kMaxReached];
  const int32_t n = next_frontier(t, f, frontier_of(f, 0), lo, hi, rows, ranks);
  *f.frontier_b_count = n;
  for (int32_t i = 0; i < n; ++i) {
    f.frontier_b[i] = rows[i]; f.frontier_b_ranks[i] = t.committed_paths ? ranks[i] : 0;
  }
  s.audit_hops = s.hops;
  s.audit_phase = 1; s.audit_choices = (1 << 0) | (1 << 1) | (1 << 3);
  if (s.hops < kMaxPathRelations) {
    for (int32_t i = 0; i < n; ++i)
      if (has_triples(t, f, t.entities.owner[rows[i]])) s.audit_choices |= 1 << 2;
  }
  if (s.hops > 1) s.audit_choices |= 1 << 4;
  s.lo = s.hi = -1;
  s.close_pos = s.sep_pos = s.esc_pos = 0;
  prepare_audit(t, f, s, true);
}

__device__ void advance_audit(const Tables& t, const Memory& f, State& s, int32_t token) {
  if (s.audit_phase == 1 || s.audit_phase == 5 || s.audit_phase == 6) {
    if (token != audit_token(t, f, s)) { s.phase = kError; return; }
    if (++s.audit_offset == audit_item_size(t, f.audit_items[s.audit_item])) {
      s.audit_offset = 0;
      if (++s.audit_item == s.audit_items) s.audit_phase = s.audit_phase == 1 ? 2 : 0;
    }
    return;
  }
  if (s.audit_phase == 2) {
    int32_t choices = 0;
    for (int32_t c = 0; c < 5; ++c) {
      if (!(s.audit_choices & (1 << c))) continue;
      const int32_t begin = t.audit_command_offsets[c], end = t.audit_command_offsets[c + 1];
      if (token != t.audit_command_tokens[begin + s.audit_pos]) continue;
      choices |= 1 << c;
      if (begin + s.audit_pos + 1 == end) {
        s.audit_action = c;
        if (c == 0 || c == 4) { finish_audit(t, f, s); return; }
        const int32_t phase = c == 1 ? 3 : 4;
        s = opened_segment(s, c == 1 ? kSubject : kRelation,
                           c == 1 ? t.entities.count : t.relations.count);
        s.audit_phase = phase;
        s.selecting = c == 1;
        if (c == 2) {
          *f.frontier_count = *f.frontier_b_count;
          for (int32_t i = 0; i < *f.frontier_count; ++i) {
            f.frontier[i] = f.frontier_b[i]; f.frontier_ranks[i] = f.frontier_b_ranks[i];
          }
        }
        return;
      }
    }
    if (!choices) s.phase = kError;
    else { s.audit_choices = choices; ++s.audit_pos; }
    return;
  }
  const Rows& rows = s.audit_phase == 3 ? t.entities : t.relations;
  int32_t lo = -1, hi = -1;
  if (s.lo >= 0) {
    const int32_t a = search(rows, s.lo, s.hi, s.depth, token, false);
    const int32_t b = search(rows, a, s.hi, s.depth, token, true);
    if (candidate_in(t, f, s, a, b)) { lo = a; hi = b; }
  }
  int32_t pos = -1, close_lo = s.close_lo, close_hi = s.close_hi;
  if (s.close_pos) pos = delimiter_step(t.closer, s.close_pos, token);
  else if (s.lo >= 0) {
    const int32_t end = search(rows, s.lo, s.hi, s.depth, -1, true);
    if (candidate_in(t, f, s, s.lo, end)) {
      pos = delimiter_step(t.closer, 0, token); close_lo = s.lo; close_hi = end;
    }
  }
  if (pos == t.closer.length) { finish_audit(t, f, s, close_lo, close_hi); return; }
  if (lo < 0 && pos < 0) { s.phase = kError; return; }
  if (lo >= 0) ++s.depth;
  s.lo = lo; s.hi = hi;
  s.close_pos = pos > 0 ? pos : 0; s.close_lo = close_lo; s.close_hi = close_hi;
}

__device__ Options span_options(const Tables& t, const State& s, const Memory& f) {
  Options options{s.lo, false, false, false};
  if (s.lo < 0) return options;  // the label branch is dead
  const Rows& rows = s.segment == kRelation ? t.relations : t.entities;
  options.complete_hi = search(rows, s.lo, s.hi, s.depth, -1, true);
  if (s.selecting) return options;  // only the observation marker commits a selection
  if (s.segment == kSubject && derivation_bound(t, f)) {
    options.closes = reached_in(f, s.lo, options.complete_hi);
    // Path mode: the span right after a path or a comparison is its answer, an entity span.
    const bool answers_path =
        t.paths && (s.last.kind == kPathSpan || s.last.kind == kComparison ||
                    s.last.kind == kRelationPlan);
    options.separates = !answers_path && best_subject(t, f, s.lo, options.complete_hi) >= 0;
  } else if (s.segment == kSubject) {
    options.closes = s.op == 0 && s.lo < options.complete_hi &&
                     unquoted_in(t, s.lo, options.complete_hi);  // an operand cannot close
    options.separates = s.last.kind != kRelationPlan && any_subject(t, f, s.lo, options.complete_hi);
  } else if (s.segment == kRelation && t.relation_mode != 0) {
    options.closes = candidate_in(t, f, s, s.lo, options.complete_hi);
    options.separates = options.closes && s.hops < kMaxPathRelations &&
                        relation_goes_on(t, f, s, s.lo, options.complete_hi);
  } else if (s.op != 0) {  // a comparison's relation: both frontiers go on in lock step
    const bool complete = candidate_in(t, f, s, s.lo, options.complete_hi);
    const Frontier a = frontier_of(f, 0), b = frontier_of(f, 1);
    int32_t start_a = 0, start_b = 0;
    options.closes =
        complete &&
        (s.op == 3 ? value_verdict(t, f, s.lo, options.complete_hi, start_a, start_b) >= 0
                   : frontier_goes_on(t, f, a, s.lo, options.complete_hi, true) &&
                         frontier_goes_on(t, f, b, s.lo, options.complete_hi, true));
    options.separates = complete && s.hops < kMaxPathRelations &&
                        frontier_goes_on(t, f, a, s.lo, options.complete_hi) &&
                        frontier_goes_on(t, f, b, s.lo, options.complete_hi);
  } else {
    const bool complete = candidate_in(t, f, s, s.lo, options.complete_hi);
    options.closes = (s.segment == kObject && complete &&
                      (t.repeats || !closed_before(f, s.subj_key, s.rel_lo, s.lo))) ||
                     (on_frontier(t, s) && complete);
    bool goes_on = false;
    if (s.selected) {
      for (int32_t i = 0; i < *f.frontier_b_count; ++i)
        goes_on |= has_triples(t, f, t.entities.owner[f.frontier_b[i]]);
    } else if (t.paths) {
      goes_on = frontier_goes_on(t, f, frontier_of(f, 0), s.lo, options.complete_hi);
    }
    options.separates = s.segment == kRelation && complete &&
                        (!t.paths || (s.hops < kMaxPathRelations && goes_on));
  }
  options.stop = options.closes && residual_goal_stop(t, f, s, options.complete_hi);
  if (options.stop) options.separates = false;
  if (observes_label(t, s)) {
    options.closes = options.separates = options.stop = false;  // observe before STOP
  }
  if (t.observe_no_loop && s.observed && options.separates) {
    Memory next = f;
    next.frontier = f.frontier_b; next.frontier_count = f.frontier_b_count;
    State relation = s;
    relation.segment = kRelation; relation.observed = 0;
    options.separates = s.hops < kMaxPathRelations &&
                        candidate_in(t, next, relation, 0, t.relations.count);
  }
  return options;
}

__device__ bool requires_selection(const Tables& t, const Memory& f, const State& s) {
  return t.observe_required && s.observed && !s.selecting && !s.selected &&
         *f.frontier_b_count > 1;
}

__device__ void after_observation(const Tables& t, const Memory& f, State& s) {
  if (!t.menu) return;
  if (!t.menu_close_next) { s.audit_phase = 6; return; }
  if (requires_selection(t, f, s)) return;
  if (s.segment == kSubject) {
    prepare_step_menu(t, f, s, frontier_of(f, 1));
    s.audit_phase = 6;
    return;
  }
  s.decision_phase = 1; s.decision_pos = 0;
  s.decision_choices = span_options(t, s, f).separates ? 3 : 1;
}

__device__ void advance_span(const Tables& t, const Memory& f, State& s, int32_t token) {
  if (s.observe_offset < s.observe_length) {
    if (token != observation_token(t, f, s)) s.phase = kError;
    else if (++s.observe_offset == s.observe_length) after_observation(t, f, s);
    return;
  }
  const int32_t decision_phase = s.decision_phase, decision_pos = s.decision_pos;
  if (decision_phase == 2) {
    int32_t choices = 0;
    for (int32_t command = 0; command < 2; ++command) {
      if (!(s.decision_choices & (1 << command))) continue;
      const int32_t first = t.decision_offsets[command + 1];
      if (token != t.decision_tokens[first + decision_pos]) continue;
      choices |= 1 << command;
      if (first + decision_pos + 1 == t.decision_offsets[command + 2]) {
        s.decision_phase = 3 + command; s.decision_pos = 0;
        if (command) {
          prepare_step_menu(t, f, s, frontier_of(f, 1));
          s.audit_phase = 6;
        }
        return;
      }
    }
    if (!choices) s.phase = kError;
    else { s.decision_choices = choices; ++s.decision_pos; }
    return;
  }
  if (decision_phase) {
    const int32_t piece = decision_phase == 1 ? 0 : decision_phase;
    if (token != t.decision_tokens[t.decision_offsets[piece] + decision_pos]) {
      s.phase = kError; return;
    }
    if (decision_phase == 1) {
      if (++s.decision_pos == t.decision_offsets[1]) {
        s.decision_phase = 2; s.decision_pos = 0;
      }
      return;
    }
    s.decision_phase = s.decision_pos = 0;  // advance the forced closer/separator normally
  }
  bool defer_alive = false;
  if (s.selecting && s.defer_pos) {
    if (token == t.defer_tokens[s.defer_pos - 1]) {
      if (s.defer_pos == t.defer_length) {
        s.selecting = 0; s.selected = s.observed = 1;
        s.segment = s.select_segment;
        s.lo = s.select_lo; s.hi = s.select_hi; s.depth = s.select_depth;
        ++s.defer_count; s.defer_pos = 0;
        s.close_pos = s.sep_pos = s.esc_pos = s.observe_pos = 0;
        if (t.menu_close_next) after_observation(t, f, s);
        return;
      }
      ++s.defer_pos; defer_alive = true;
    } else s.defer_pos = 0;
  }
  const bool required = requires_selection(t, f, s);
  if (required && token != t.selection_tokens[s.select_pos]) { s.phase = kError; return; }
  const Rows& rows = s.segment == kRelation ? t.relations : t.entities;
  const bool alive = s.lo >= 0;
  Options options = span_options(t, s, f);
  // A required selection owns this token. Preserve legacy off-mode branch state.
  if ((t.menu || t.observe_defer) && required) options.separates = false;
  const bool owns_selection = required && (t.observe_defer || t.menu_close_next);
  if (owns_selection) options.closes = false;
  int32_t select_pos = 0;
  if (t.observe_select && !s.selecting && !s.selected &&
      (s.select_pos > 0 || (s.observed && (required || !options.stop) && *f.frontier_b_count > 0 &&
                           candidate_in(t, f, s, s.lo, options.complete_hi)))) {
    if (token == t.selection_tokens[s.select_pos]) {
      select_pos = s.select_pos + 1;
      if (select_pos == t.selection_length) {
        const bool can_defer = required && s.defer_count < t.observe_defer;
        const int32_t segment = s.segment, lo = s.select_pos ? s.select_lo : s.lo;
        const int32_t hi = s.select_pos ? s.select_hi : options.complete_hi;
        const int32_t depth = s.select_pos ? s.select_depth : s.depth;
        s = opened_segment(s, kSubject, t.entities.count);
        s.selecting = 1;
        s.defer_pos = can_defer ? 1 : 0;
        s.select_segment = segment;
        s.select_lo = lo; s.select_hi = hi; s.select_depth = depth;
        return;
      }
      if (s.select_pos == 0) {
        s.select_lo = s.lo; s.select_hi = options.complete_hi; s.select_depth = s.depth;
      }
    }
  }
  int32_t observe_pos = 0, observe_lo = s.observe_lo, observe_hi = s.observe_hi;
  int32_t observe_depth = s.observe_depth;
  if (observes_label(t, s) &&
      (s.observe_pos > 0 || candidate_in(t, f, s, s.lo, options.complete_hi))) {
    if (token == t.observation_tokens[t.observation_offsets[0] + s.observe_pos]) {
      observe_pos = s.observe_pos + 1;
      if (s.observe_pos == 0) {
        observe_lo = s.lo;
        observe_hi = options.complete_hi;
        observe_depth = s.depth;
      }
      if (observe_pos == observation_piece_size(t, 0)) {
        if (s.selecting) {
          int32_t n = 0;
          for (int32_t i = 0; i < *f.frontier_b_count; ++i) {
            const int32_t owner = t.entities.owner[f.frontier_b[i]];
            const int64_t end = t.entity_row_offsets[owner + 1];
            const int64_t at = bound(t.entity_rows, t.entity_row_offsets[owner], end, observe_lo, false);
            if (at < end && t.entity_rows[at] < observe_hi) {
              f.frontier_b[n] = f.frontier_b[i];
              f.frontier_b_ranks[n++] = f.frontier_b_ranks[i];
            }
          }
          *f.frontier_b_count = n;
          s.selecting = 0; s.selected = 1;
          s.defer_pos = 0;
          s.segment = s.select_segment;
          observe_lo = s.select_lo; observe_hi = s.select_hi; observe_depth = s.select_depth;
        }
        s.lo = observe_lo;
        s.hi = observe_hi;
        s.depth = observe_depth;
        s.observed = 1;
        s.observe_pos = s.observe_lo = s.observe_hi = s.observe_depth = 0;
        s.close_pos = s.close_lo = s.close_hi = 0;
        s.sep_pos = s.sep_lo = s.sep_hi = s.esc_pos = s.op_pos = s.op_kind = 0;
        prepare_observation(t, f, s);
        return;
      }
    }
  }
  int32_t lo = -1;
  int32_t hi = -1;
  if (alive && !s.observed) {
    const int32_t next_lo = search(rows, s.lo, s.hi, s.depth, token, false);
    const int32_t next_hi = search(rows, next_lo, s.hi, s.depth, token, true);
    if (candidate_in(t, f, s, next_lo, next_hi)) {
      lo = next_lo;
      hi = next_hi;
    }
  }

  int32_t close_pos = -1, close_lo = s.close_lo, close_hi = s.close_hi;
  if (s.close_pos > 0) {
    close_pos = delimiter_step(t.closer, s.close_pos, token);
  } else if (options.closes) {
    close_pos = delimiter_step(t.closer, 0, token);
    close_lo = s.lo;
    close_hi = options.complete_hi;
  }
  int32_t sep_pos = -1, sep_lo = s.sep_lo, sep_hi = s.sep_hi;
  const Delimiter& separator = separator_of(t, s);
  if (s.sep_pos > 0 && !options.stop) {
    sep_pos = delimiter_step(separator, s.sep_pos, token);
  } else if (options.separates) {
    sep_pos = delimiter_step(separator, 0, token);
    sep_lo = s.lo;
    sep_hi = options.complete_hi;
  }
  int32_t esc_pos = -1;
  if (s.esc_pos > 0 && !options.stop && !owns_selection) {
    esc_pos = delimiter_step(t.escape, s.esc_pos, token);
  } else if (alive && !options.stop && !owns_selection && may_escape(t, f, s)) {
    esc_pos = delimiter_step(t.escape, 0, token);
  }
  int32_t op_kind = s.op_kind;
  const int32_t op_pos = operator_step(t, f, s, token, op_kind);

  if (close_pos == t.closer.length && s.segment == kRelation && t.relation_mode != 0) {
    *f.reached_count = *f.frontier_count = 0;
    close_span(t, s, f, kRelationPlan, close_lo, close_hi);
  } else if (close_pos == t.closer.length && on_frontier(t, s) && s.op == 3) {  // a value comparison
    int32_t a = 0, b = 0;
    const int32_t verdict = value_verdict(t, f, close_lo, close_hi, a, b);
    const int32_t owner = t.verdict_entities[verdict == 1 ? 0 : 1];
    int32_t count = 0;  // the verdict's label rows bind the answer after the comparison
    for (int64_t r = t.entity_row_offsets[owner]; r < t.entity_row_offsets[owner + 1]; ++r) {
      insert_reached(f, count, t.entity_rows[r]);
    }
    *f.reached_count = count;
    const int32_t op = s.op;
    close_span(t, s, f, kComparison, a, b);
    s.last.subj_lo = verdict == 1 ? 3 : 4;
    s.last.subj_hi = op;
  } else if (close_pos == t.closer.length && on_frontier(t, s) && s.op != 0) {  // dates
    int32_t next_a[kMaxReached], ranks_a[kMaxReached], next_b[kMaxReached], ranks_b[kMaxReached];
    const int32_t na = next_frontier(t, f, frontier_of(f, 0), close_lo, close_hi, next_a, ranks_a);
    const int32_t nb = next_frontier(t, f, frontier_of(f, 1), close_lo, close_hi, next_b, ranks_b);
    int32_t rank_a = 0, lo_a = 0, hi_a = 0, rank_b = 0, lo_b = 0, hi_b = 0;
    earliest_dated(t, next_a, ranks_a, na, rank_a, lo_a, hi_a);
    earliest_dated(t, next_b, ranks_b, nb, rank_b, lo_b, hi_b);
    // Each operand commits to its best start that reaches a date.
    const int32_t a = f.starts[rank_a], b = f.starts[kMaxReached + rank_b];
    const bool a_first = hi_a < lo_b, b_first = hi_b < lo_a, earlier = s.op == 1;
    const int32_t result = a_first ? (earlier ? 0 : 1) : b_first ? (earlier ? 1 : 0) : 2;
    int32_t count = 0;  // the result's label rows bind the answer after the comparison
    for (int32_t k = 0; k < 2; ++k) {
      if (result == 1 - k) continue;  // 0: A only, 1: B only, 2: both
      const int32_t owner = t.entities.owner[k == 0 ? a : b];
      for (int64_t r = t.entity_row_offsets[owner]; r < t.entity_row_offsets[owner + 1]; ++r) {
        insert_reached(f, count, t.entity_rows[r]);
      }
    }
    *f.reached_count = count;
    const int32_t op = s.op;
    close_span(t, s, f, kComparison, a, b);
    s.last.subj_lo = result;
    s.last.subj_hi = op;
  } else if (close_pos == t.closer.length && (on_frontier(t, s) || s.selected)) {  // a path ends
    if (t.audit) { start_audit(t, f, s, close_lo, close_hi); return; }
    int32_t frontier[kMaxReached], ranks[kMaxReached];
    int32_t n = path_frontier(t, f, s, close_lo, close_hi, frontier, ranks);
    if (t.committed_paths) {  // what the best start with a non-empty path reaches
      int32_t best = 0;
      while (best < n && ranks[best] == ranks[0]) ++best;
      n = best;
    }
    int32_t count = 0;  // the frontier entities' label rows bind the spans after the path
    for (int32_t i = 0; i < n; ++i) {
      const int32_t owner = t.entities.owner[frontier[i]];
      for (int64_t r = t.entity_row_offsets[owner]; r < t.entity_row_offsets[owner + 1]; ++r) {
        insert_reached(f, count, t.entity_rows[r]);
      }
    }
    *f.reached_count = count;
    close_span(t, s, f, kPathSpan, close_lo, close_hi);
    if (t.observe_terminal) {
      prepare_audit(t, f, s, false, true);
      if (s.audit_items) s.audit_phase = 5;  // readback, then the forced bound answer; no decision
    }
  } else if (close_pos == t.closer.length) {
    if (s.segment == kSubject && derivation_bound(t, f)) {  // bound to one entity
      close_lo = best_row(t, f, close_lo, close_hi, false);
      close_hi = close_lo + 1;
    }
    close_span(t, s, f, s.segment == kObject ? kFactSpan : kEntitySpan, close_lo, close_hi);
  } else if (sep_pos == separator.length && s.segment == kSubject && s.op != 0) {
    // A comparison's operand: rank its starts into its frontier, and keep them.
    const int32_t operand = s.operand;
    const Frontier out = frontier_of(f, operand);
    const int32_t n = ranked_starts(t, f, sep_lo, sep_hi, out);
    *out.count = n;
    for (int32_t i = 0; i < n; ++i) f.starts[operand * kMaxReached + i] = out.rows[i];
    if (operand == 0) {  // operand B follows
      s = opened_segment(s, kSubject, t.entities.count);
      s.operand = 1;
    } else {  // both frontiers go on in lock step
      s = opened_segment(s, kRelation, t.relations.count);
      s.subj_lo = s.subj_key = sep_lo;
      s.subj_hi = sep_hi;
      s.hops = 1;
    }
  } else if (sep_pos == separator.length) {
    if (s.segment == kSubject) {
      const bool selected = s.selected;
      // Derivation mode binds the subject to one entity (see best_subject).
      const int32_t subject = t.derivation ? best_subject(t, f, sep_lo, sep_hi) : -1;
      s = opened_segment(s, kRelation, t.relations.count);
      s.subj_lo = t.derivation ? subject : sep_lo;
      s.subj_key = sep_lo;  // the label, whichever entity derivation mode binds
      s.subj_hi = t.derivation ? subject + 1 : sep_hi;
      if (f.stop) set_stop_used(s, stop_start_words(t, f, sep_lo));
      if (t.relation_mode != 0) {
        *f.frontier_count = 0;  // start admission/ranking above, without an execution frontier
        s.hops = 1;
      } else if (selected) {
        *f.frontier_count = *f.frontier_b_count;
        for (int32_t i = 0; i < *f.frontier_b_count; ++i) {
          f.frontier[i] = f.frontier_b[i]; f.frontier_ranks[i] = f.frontier_b_ranks[i];
        }
        s.hops = 1;
      } else if (t.paths && t.committed_paths) {  // every start, ranked
        *f.frontier_count = ranked_starts(t, f, sep_lo, sep_hi, frontier_of(f, 0));
        s.hops = 1;
      } else if (t.paths) {  // the frontier: every entity the label may name that has triples
        const bool bound = derivation_bound(t, f);
        int32_t n = 0;
        for (int32_t row = sep_lo; row < sep_hi; ++row) {
          const int32_t owner = t.entities.owner[row];
          if ((!bound || reached_in(f, row, row + 1)) && has_triples(t, f, owner)) {
            insert_sorted(f.frontier, n, t.entity_rows[t.entity_row_offsets[owner]]);
          }
        }
        *f.frontier_count = n;
        s.hops = 1;
      }
    } else if (t.relation_mode != 0) {
      if (t.relation_mode == 2) consume_relation(t, f, s, sep_lo, sep_hi);
      s = opened_segment(s, kRelation, t.relations.count);
      s.hops += 1;
    } else if (t.paths) {  // the next hop: the relation's objects become the frontier
      if (f.stop && s.op == 0) {
        set_stop_used(s, stop_used(s) | stop_consumed(t, f, sep_lo, sep_hi));
      }
      for (int32_t operand = 0; operand < (s.op != 0 ? 2 : 1); ++operand) {
        int32_t frontier[kMaxReached], ranks[kMaxReached];
        const Frontier current = frontier_of(f, operand);  // a comparison has two
        const int32_t n = s.selected ? path_frontier(t, f, s, sep_lo, sep_hi, frontier, ranks)
                                    : next_frontier(t, f, current, sep_lo, sep_hi, frontier, ranks);
        for (int32_t i = 0; i < n; ++i) {
          current.rows[i] = frontier[i];
          current.ranks[i] = ranks[i];
        }
        *current.count = n;
      }
      s = opened_segment(s, kRelation, t.relations.count);
      s.hops += 1;
    } else {
      s = opened_segment(s, kObject, t.entities.count);
      s.rel_lo = sep_lo;
      s.rel_hi = sep_hi;
    }
  } else if (esc_pos == t.escape.length) {
    State escaped{};
    escaped.phase = kEscaped;
    escaped.last = s.last;
    s = escaped;
  } else if (op_pos >= 0 && op_pos == t.operators[op_kind].length) {  // operand A follows
    s = opened_segment(s, kSubject, t.entities.count);
    s.op = op_kind + 1;
    s.operand = 0;
  } else if (!defer_alive && lo < 0 && close_pos < 0 && sep_pos < 0 && esc_pos < 0 && op_pos < 0 && observe_pos == 0 && select_pos == 0) {
    s.phase = kError;
  } else {
    if (lo >= 0) s.depth += 1;
    s.lo = lo;
    s.hi = hi;
    s.close_pos = close_pos > 0 ? close_pos : 0;
    s.close_lo = close_pos > 0 ? close_lo : 0;
    s.close_hi = close_pos > 0 ? close_hi : 0;
    s.sep_pos = sep_pos > 0 ? sep_pos : 0;
    s.sep_lo = sep_pos > 0 ? sep_lo : 0;
    s.sep_hi = sep_pos > 0 ? sep_hi : 0;
    s.esc_pos = esc_pos > 0 ? esc_pos : 0;
    s.op_pos = op_pos > 0 ? op_pos : 0;
    s.op_kind = op_pos > 0 ? op_kind : 0;
    s.observe_pos = observe_pos;
    s.observe_lo = observe_pos > 0 ? observe_lo : 0;
    s.observe_hi = observe_pos > 0 ? observe_hi : 0;
    s.observe_depth = observe_pos > 0 ? observe_depth : 0;
    s.select_pos = select_pos;
  }
  if (decision_phase >= 3 && s.phase == kSpan &&
      t.decision_offsets[decision_phase] + decision_pos + 1 < t.decision_offsets[decision_phase + 1]) {
    s.decision_phase = decision_phase; s.decision_pos = decision_pos + 1;
  }
}

// The state right after the span opener: a span's subject segment, every row in range.
__device__ void open_span(const Tables& t, State& s) {
  if (s.answer_complete) {
    s.phase = kError;
    return;
  }
  s = opened_segment(s, kSubject, t.entities.count);
  s.phase = kSpan;
  s.pos = 0;
  s.subj_lo = s.subj_hi = s.rel_lo = s.rel_hi = 0;
  s.op = s.operand = 0;
  s.stop_used_lo = s.stop_used_hi = 0;
}

__device__ void advance_state(const Tables& t, const Memory& f, State& s, int32_t token) {
  if (s.phase == kError) return;
  if (token < 0 || token >= t.vocab) {
    s.phase = kError;
  } else if (s.audit_phase) {
    advance_audit(t, f, s, token);
  } else if (s.answer_complete) {
    if ((t.eos_mask[token >> 5] & (1u << (token & 31))) == 0) s.phase = kError;
  } else if (s.answer_pos > 0) {
    if (token != t.answer_tokens[s.answer_pos - 1]) s.phase = kError;
    else if (s.answer_pos == t.answer_length) { s.answer_pos = 0; open_span(t, s); }
    else ++s.answer_pos;
  } else if (s.phase == kFree) {
    const int8_t code = t.free_next[static_cast<int64_t>(s.pos) * t.vocab + token];
    if (code == kReject) {
      s.phase = kError;
    } else if (code == kComplete) {
      open_span(t, s);
    } else {
      s.pos = code;
    }
  } else if (s.phase == kEscaped) {
    const int8_t code = t.esc_next[static_cast<int64_t>(s.pos) * t.vocab + token];
    if (code == kReject) {
      s.phase = kError;
    } else if (code == kComplete) {
      close_span(t, s, f, kUnresolved, 0, 0);
    } else {
      s.pos = code;
    }
  } else {
    advance_span(t, f, s, token);
  }
}

// The memory of a sequence; advancing updates it as spans close.
__device__ Memory memory_of(const Tables& t, int32_t* rows, int32_t* counts, int32_t seq, bool advancing) {
  return memory_at(rows + static_cast<int64_t>(seq) * t.layout.memory_ints,
                   counts + static_cast<int64_t>(seq) * kMemoryCounts, advancing, seq,
                   t.layout);
}

__global__ void advance_kernel(Tables t, State* states, int32_t* fact_rows, int32_t* fact_counts,
                               const int32_t* tokens, int32_t sequences) {
  const int32_t seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= sequences) return;
  t = request_tables(t, seq);
  State s = states[seq];
  advance_state(t, memory_of(t, fact_rows, fact_counts, seq, true), s, tokens[seq]);
  states[seq] = s;
}

// Put slots back in the initial state, with their cursors unset, and mark them
// active or not.
__global__ void begin_kernel(State* states, int32_t* fact_counts, int32_t* cursors,
                             uint8_t* active, uint8_t* alias_active, int32_t* path_options,
                             const int32_t* slots,
                             int32_t count, uint8_t on) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int32_t slot = slots[i];
  states[slot] = State{};
  for (int32_t k = 0; k < kMemoryCounts; ++k) {  // the counts, and the base KG
    fact_counts[static_cast<int64_t>(slot) * kMemoryCounts + k] = 0;
  }
  cursors[slot] = -1;
  active[slot] = on;
  if (alias_active) alias_active[slot] = 0;
  if (path_options) path_options[static_cast<int64_t>(slot) * 10] = -1;
}

// Start a slot inside a span, as if the span opener had just been written (a forced opener).
__global__ void open_kernel(Tables t, State* states, int32_t slot) {
  t = request_tables(t, slot);
  open_span(t, states[slot]);
}

// Let a slot read a KG version (0: the base KG) until its next begin.
__global__ void set_version_kernel(int32_t* fact_counts, int32_t slot, int32_t version) {
  fact_counts[static_cast<int64_t>(slot) * kMemoryCounts + 3] = version;
}

// Advance each listed active slot over its token history, from its cursor (the
// prompt length the first time) up to its total length.
__global__ void sync_kernel(Tables t, State* states, int32_t* fact_rows, int32_t* fact_counts,
                            int32_t* cursors, const uint8_t* active,
                            const int32_t* slots, int32_t count, const int32_t* history,
                            int64_t history_stride, const int32_t* prompt_len,
                            const int32_t* total_len) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int32_t slot = slots[i];
  if (!active[slot]) return;
  t = request_tables(t, slot);
  int32_t cursor = cursors[slot] < 0 ? prompt_len[slot] : cursors[slot];
  State s = states[slot];
  const int32_t* tokens = history + static_cast<int64_t>(slot) * history_stride;
  const Memory f = memory_of(t, fact_rows, fact_counts, slot, true);
  for (; cursor < total_len[slot]; ++cursor) advance_state(t, f, s, tokens[cursor]);
  states[slot] = s;
  cursors[slot] = cursor;
}

__device__ void set_token(uint32_t* out, int32_t token) {
  atomicOr(&out[token >> 5], 1u << (token & 31));
}

__device__ void add_delimiter_tokens(const Delimiter& d, int32_t pos, uint32_t* out) {
  for (int32_t i = d.offsets[pos]; i < d.offsets[pos + 1]; ++i) set_token(out, d.tokens[i]);
}

// The label tokens a span state allows: the tokens at the current depth of the
// candidate rows in its row range. The threads of the block share the work.
// Set the tokens at `depth` of the label rows that position(i) gives for i in
// [lo, hi). Those tokens must be non-decreasing in i, as they are for rows that share
// their first `depth` tokens, in label index order. Each thread takes a chunk and jumps
// from one token's run to the next by binary search, so the work grows with the number
// of distinct tokens rather than with the number of rows. With unquoted (label index rows,
// position the identity), a run's token is set only if the run holds a row an unbound
// subject may name (see Tables::unquoted_rows).
template <typename Position>
__device__ void add_distinct_tokens(const Rows& rows, int64_t lo, int64_t hi, int32_t depth,
                                    Position position, uint32_t* out,
                                    const int32_t* unquoted = nullptr) {
  const int64_t chunk = (hi - lo + blockDim.x - 1) / blockDim.x;
  int64_t i = lo + threadIdx.x * chunk;
  const int64_t stop = hi < i + chunk ? hi : i + chunk;
  while (i < stop) {
    const int32_t token = token_at(rows, position(i), depth);
    int64_t a = i + 1, b = stop;  // the first i past this token's run
    while (a < b) {
      const int64_t m = a + (b - a) / 2;
      if (token_at(rows, position(m), depth) <= token) a = m + 1; else b = m;
    }
    if (token >= 0 && (unquoted == nullptr || unquoted[a] > unquoted[i])) set_token(out, token);
    i = a;
  }
}

__device__ void add_label_tokens(const Tables& t, const Memory& f, const State& s,
                                 uint32_t* out) {
  if (s.selecting) {
    for (int32_t i = threadIdx.x; i < *f.frontier_b_count; i += blockDim.x) {
      const int32_t owner = t.entities.owner[f.frontier_b[i]];
      for (int64_t at = t.entity_row_offsets[owner]; at < t.entity_row_offsets[owner + 1]; ++at) {
        const int32_t row = t.entity_rows[at];
        if (row < s.lo || row >= s.hi) continue;
        const int32_t token = token_at(t.entities, row, s.depth);
        if (token >= 0) set_token(out, token);
      }
    }
    return;
  }
  if (s.segment == kSubject && derivation_bound(t, f)) {  // only the reached rows
    for (int32_t i = reached_from(f, s.lo) + threadIdx.x; i < *f.reached_count;
         i += blockDim.x) {
      if (f.reached[i] >= s.hi) break;
      const int32_t token = token_at(t.entities, f.reached[i], s.depth);
      if (token >= 0) set_token(out, token);
    }
    return;
  }
  if (s.segment == kSubject) {
    if (s.depth == 0) return;  // the root mask already holds them
    // A wide prefix's next tokens are precomputed (xgraph.engine.wide_masks).
    const int64_t key = static_cast<int64_t>(s.lo) * 8 + s.depth;
    int32_t a = 0, b = t.wide_count;
    while (a < b) {
      const int32_t m = a + (b - a) / 2;
      if (t.wide_keys[m] < key) a = m + 1; else b = m;
    }
    if (s.depth < 8 && a < t.wide_count && t.wide_keys[a] == key) {
      const uint32_t* wide = t.wide_masks + static_cast<int64_t>(a) * t.words;
      for (int32_t w = threadIdx.x; w < t.words; w += blockDim.x) {
        if (wide[w]) atomicOr(&out[w], wide[w]);
      }
      return;
    }
    add_distinct_tokens(t.entities, s.lo, s.hi, s.depth,
                        [](int64_t i) { return static_cast<int32_t>(i); }, out,
                        t.unquoted_starts ? t.unquoted_rows : nullptr);
    return;
  }
  if (s.segment == kRelation) {
    if (t.relation_mode != 0) {
      for (int32_t row = s.lo + threadIdx.x; row < s.hi; row += blockDim.x) {
        if (!supported_relation(t, f, s, t.relations.owner[row])) continue;
        const int32_t token = token_at(t.relations, row, s.depth);
        if (token >= 0) set_token(out, token);
      }
      return;
    }
    // One warp per candidate subject, so that a label many subjects share does not
    // serialize them. A subject's edges are sorted by relation: each lane takes a
    // chunk and jumps from one relation's run to the next, so hub subjects cost their
    // distinct relations, not their edges.
    const int32_t lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int32_t subjects = subject_count(t, f, s);
    for (int32_t i = threadIdx.x >> 5; i < subjects; i += warps) {
      const int32_t entity = t.entities.owner[subject_row(t, f, s, i)];
      const Edges edges = edges_of(t, f.version, entity);
      const int64_t begin = edges.begin, end = edges.end;
      const int64_t chunk = (end - begin + 31) / 32;
      int64_t e = begin + lane * chunk;
      const int64_t last = end < e + chunk ? end : e + chunk;
      while (e < last) {
        const int32_t relation = edges.relations[e];
        // A comparison's relations: those operand B's frontier has too.
        const bool shared = s.op == 0 || frontier_has(t, f, frontier_of(f, 1), relation);
        const int32_t stop = shared ? t.relation_row_offsets[relation + 1] : 0;
        for (int32_t i = t.relation_row_offsets[relation]; i < stop; ++i) {
          const int32_t row = t.relation_rows[i];
          if (row < s.lo || row >= s.hi) continue;
          if (!loop_label_allowed(t, f, s, row)) continue;
          const int32_t token = token_at(t.relations, row, s.depth);
          if (token >= 0) set_token(out, token);
        }
        int64_t a = e + 1, b = last;  // the first edge past this relation's run
        while (a < b) {
          const int64_t m = a + (b - a) / 2;
          if (edges.relations[m] <= relation) a = m + 1; else b = m;
        }
        e = a;
      }
    }
    return;
  }
  for (int32_t subj = s.subj_lo; subj < s.subj_hi; ++subj) {
    const int32_t entity = t.entities.owner[subj];
    for (int32_t rel = s.rel_lo; rel < s.rel_hi; ++rel) {
      int64_t first = 0, last = 0;
      const Edges edges = object_slice(t, f, entity, t.relations.owner[rel], first, last);
      const int64_t from = bound(edges.objects, first, last, s.lo, false);
      const int64_t to = bound(edges.objects, from, last, s.hi, false);
      const int32_t* objects = edges.objects;
      add_distinct_tokens(t.entities, from, to, s.depth,
                          [objects](int64_t e) { return objects[e]; }, out);
    }
  }
}

// The delimiter tokens a span state allows (one thread).
__device__ void add_span_delimiters(const Tables& t, const Memory& f, const State& s,
                                    uint32_t* out) {
  const Options options = span_options(t, s, f);
  if (t.observe_select && !s.selecting && !s.selected &&
      (s.select_pos > 0 || (s.observed && !options.stop && *f.frontier_b_count > 0 &&
                           candidate_in(t, f, s, s.lo, options.complete_hi)))) {
    const int32_t token = t.selection_tokens[s.select_pos];
    out[token >> 5] |= 1u << (token & 31);
  }
  if (observes_label(t, s) &&
      (s.observe_pos > 0 || candidate_in(t, f, s, s.lo, options.complete_hi))) {
    const int32_t token = t.observation_tokens[t.observation_offsets[0] + s.observe_pos];
    out[token >> 5] |= 1u << (token & 31);
  }
  if (s.close_pos > 0) {
    add_delimiter_tokens(t.closer, s.close_pos, out);
  } else if (options.closes) {
    add_delimiter_tokens(t.closer, 0, out);
  }
  const Delimiter& separator = separator_of(t, s);
  if (s.sep_pos > 0 && !options.stop) {
    add_delimiter_tokens(separator, s.sep_pos, out);
  } else if (options.separates) {
    add_delimiter_tokens(separator, 0, out);
  }
  if (s.esc_pos > 0 && !options.stop) {
    add_delimiter_tokens(t.escape, s.esc_pos, out);
  } else if (s.lo >= 0 && !options.stop && may_escape(t, f, s)) {  // the label branch is alive
    add_delimiter_tokens(t.escape, 0, out);
  }
  if (s.op_pos > 0 || may_operate(t, f, s)) {  // the operators the bytes so far allow
    for (int32_t j = 0; j < t.operator_count; ++j) {
      if (s.op_pos > 0 && t.operator_lcp[s.op_kind * kMaxOperators + j] < s.op_pos) continue;
      add_delimiter_tokens(t.operators[j], s.op_pos, out);
    }
  }
}

// One block per sequence: start from a precomputed mask (or zeros), then, inside a
// span, add the label tokens of the candidate rows and the delimiter tokens.
// Write the mask of a state into out (words of the block's own row). Every thread of
// the block must call it.
__device__ void write_mask(const Tables& t, const Memory& f, const State& s, uint32_t* out) {
  const uint32_t* base = nullptr;
  if (s.audit_phase) {
    for (int32_t w = threadIdx.x; w < t.words; w += blockDim.x) out[w] = 0u;
    __syncthreads();
    if (s.audit_phase == 1 || s.audit_phase == 5 || s.audit_phase == 6) {
      if (threadIdx.x == 0) set_token(out, audit_token(t, f, s));
    } else if (s.audit_phase == 2) {
      if (threadIdx.x == 0)
        for (int32_t c = 0; c < 5; ++c)
          if (s.audit_choices & (1 << c))
            set_token(out, t.audit_command_tokens[t.audit_command_offsets[c] + s.audit_pos]);
    } else {
      if (s.lo >= 0) add_label_tokens(t, f, s, out);
      if (threadIdx.x == 0) {
        const Rows& rows = s.audit_phase == 3 ? t.entities : t.relations;
        if (s.close_pos) add_delimiter_tokens(t.closer, s.close_pos, out);
        else if (s.lo >= 0) {
          const int32_t hi = search(rows, s.lo, s.hi, s.depth, -1, true);
          if (candidate_in(t, f, s, s.lo, hi)) add_delimiter_tokens(t.closer, 0, out);
        }
      }
    }
    return;
  }
  if (s.answer_pos > 0) {
    for (int32_t w = threadIdx.x; w < t.words; w += blockDim.x) out[w] = 0u;
    __syncthreads();
    if (threadIdx.x == 0) set_token(out, t.answer_tokens[s.answer_pos - 1]);
    return;
  }
  if (s.phase == kFree) {
    base = s.answer_complete ? t.eos_mask : t.free_mask + static_cast<int64_t>(s.pos) * t.words;
  } else if (s.phase == kEscaped) {
    base = t.esc_mask + static_cast<int64_t>(s.pos) * t.words;
  } else if (s.phase == kSpan && s.segment == kSubject && s.lo < s.hi && s.depth == 0 &&
             !s.selecting && !derivation_bound(t, f)) {
    base = t.root_mask;  // at depth 0 of the subject segment the range is every row
  }
  for (int32_t w = threadIdx.x; w < t.words; w += blockDim.x) out[w] = base ? base[w] : 0u;
  if (s.phase != kSpan) return;  // uniform across the block
  __syncthreads();
  if (s.observe_offset < s.observe_length) {
    if (threadIdx.x == 0) {
      const int32_t token = observation_token(t, f, s);
      out[token >> 5] |= 1u << (token & 31);
    }
    return;
  }
  if (s.decision_phase) {
    if (threadIdx.x == 0) {
      if (s.decision_phase == 2) {
        for (int32_t command = 0; command < 2; ++command)
          if (s.decision_choices & (1 << command))
            set_token(out, t.decision_tokens[t.decision_offsets[command + 1] + s.decision_pos]);
      } else {
        const int32_t piece = s.decision_phase == 1 ? 0 : s.decision_phase;
        set_token(out, t.decision_tokens[t.decision_offsets[piece] + s.decision_pos]);
      }
    }
    return;
  }
  if (requires_selection(t, f, s)) {
    if (threadIdx.x == 0) set_token(out, t.selection_tokens[s.select_pos]);
    return;
  }
  if (s.lo >= 0 && !s.observed) add_label_tokens(t, f, s, out);
  if (threadIdx.x == 0) {
    add_span_delimiters(t, f, s, out);
    if (s.selecting && s.defer_pos) set_token(out, t.defer_tokens[s.defer_pos - 1]);
  }
}

__global__ void mask_kernel(Tables t, const State* states, int32_t* fact_rows,
                            int32_t* fact_counts, uint32_t* mask) {
  const int32_t seq = blockIdx.x;
  t = request_tables(t, seq);
  write_mask(t, memory_of(t, fact_rows, fact_counts, seq, false), states[seq],
             mask + static_cast<int64_t>(seq) * t.words);
}

// The columns of 32-column word w that must be set to -inf: those the mask does not
// allow, including every column past the served vocabulary.
__device__ uint32_t blocked_bits(const uint32_t* words, int32_t w, int32_t mask_words,
                                 int32_t columns) {
  uint32_t blocked = ~(w < mask_words ? words[w] : 0u);
  const int32_t rest = columns - w * 32;
  if (rest < 32) blocked &= (1u << rest) - 1u;
  return blocked;
}

// One block per logits row: find where the mask of the row's slot lives. Outside a
// span it is a row of a precomputed table (free text, escaped span), used in place;
// inside a span it is computed into the row's mask buffer. Rows of inactive slots
// are skipped.
//
// Under speculative decoding a request owns consecutive rows, one per draft position
// (local_pos 0, 1, ...). Row k is masked with a scratch copy of the slot's state
// advanced over the input tokens of the request's rows 1..k, the drafts before it;
// the slot's state itself does not move. A row whose drafts leave the error phase
// (a disallowed draft, which verification rejects before reaching the row) is left
// unmasked rather than all -inf. local_pos and input_ids are null without drafts.
// The scratch state has a scratch copy of the slot's memory (facts and reached rows), so that
// the no-repeat rule also sees the facts its drafts close.

// Whether a token continues a span state's label branch (as advance_span reads it).
__device__ bool continues_label(const Tables& t, const Memory& f, const State& s,
                                int32_t token) {
  if (s.lo < 0) return false;
  const Rows& rows = s.segment == kRelation ? t.relations : t.entities;
  const int32_t lo = search(rows, s.lo, s.hi, s.depth, token, false);
  const int32_t hi = search(rows, lo, s.hi, s.depth, token, true);
  return candidate_in(t, f, s, lo, hi);
}

__global__ void row_mask_kernel(Tables t, const State* states, int32_t* fact_rows,
                                int32_t* fact_counts, const uint8_t* active,
                                const int32_t* row_slots, const int32_t* local_pos,
                                const int32_t* input_ids, uint32_t* mask,
                                const uint32_t** sources, int32_t* escape_pos,
                                int32_t* bias_tokens, int32_t* bias_count) {
  const int32_t row = blockIdx.x;
  const int32_t slot = row_slots[row];
  t = request_tables(t, slot);
  if (threadIdx.x == 0) {
    escape_pos[row] = -1;
    bias_count[row] = 0;
  }
  if (!active[slot]) return;  // uniform across the block
  __shared__ State s;
  extern __shared__ int32_t scratch_rows[];  // t.layout.memory_ints, including successor support
  __shared__ int32_t scratch_counts[kMemoryCounts];
  const int32_t* own = fact_rows + static_cast<int64_t>(slot) * t.layout.memory_ints;
  const int32_t* own_counts = fact_counts + static_cast<int64_t>(slot) * kMemoryCounts;
  for (int32_t i = threadIdx.x; i < t.layout.memory_ints; i += blockDim.x) scratch_rows[i] = own[i];
  for (int32_t i = threadIdx.x; i < kMemoryCounts; i += blockDim.x) {
    scratch_counts[i] = own_counts[i];
  }
  __syncthreads();
  const Memory f = memory_at(scratch_rows, scratch_counts, true, slot, t.layout);
  if (threadIdx.x == 0) {
    s = states[slot];
    const int32_t k = local_pos ? local_pos[row] : 0;
    for (int32_t j = 1; j <= k; ++j) advance_state(t, f, s, input_ids[row - k + j]);
  }
  __syncthreads();
  if (s.phase == kError) {
    if (threadIdx.x == 0) sources[row] = nullptr;
    return;
  }
  if ((s.phase == kFree || s.phase == kEscaped) && s.answer_pos == 0) {
    if (threadIdx.x == 0) {
      const uint32_t* table = s.phase == kFree ? t.free_mask : t.esc_mask;
      sources[row] = s.answer_complete ? t.eos_mask : table + static_cast<int64_t>(s.pos) * t.words;
    }
    return;
  }
  uint32_t* words = mask + static_cast<int64_t>(row) * t.words;
  write_mask(t, f, s, words);
  if (threadIdx.x == 0) {
    sources[row] = words;
    if (s.phase != kSpan || s.audit_phase) return;
    // Where the escape marker can go on (started, or startable while the label lives),
    // if some token takes it further from there.
    const int32_t pos = s.esc_pos;
    const bool goes_on = t.escape.offsets[pos + 1] > t.escape.offsets[pos];
    const Options options = span_options(t, s, f);
    const bool observing = s.observe_offset < s.observe_length || s.observe_pos > 0 ||
                           s.select_pos > 0 || requires_selection(t, f, s) ||
                           s.defer_pos > 0 || s.decision_phase > 0;
    const bool startable = !observing && s.lo >= 0 && !options.stop && may_escape(t, f, s);
    if (!observing && !options.stop && goes_on && (pos > 0 || startable)) escape_pos[row] = pos;
    if (t.escape_bias != 0.0f && pos == 0 && startable) {  // tokens that start the escape
      int32_t n = 0;
      const bool separator = s.sep_pos > 0 || options.separates;
      const bool closer = s.close_pos > 0 || options.closes;
      for (int32_t i = t.escape.offsets[0]; i < t.escape.offsets[1]; ++i) {
        const int32_t token = t.escape.tokens[i];
        int32_t kind = 0;
        const bool other_branch =
            continues_label(t, f, s, token) ||
            (observes_label(t, s) && candidate_in(t, f, s, s.lo, options.complete_hi) &&
             token == t.observation_tokens[t.observation_offsets[0]]) ||
            (t.observe_select && s.observed && !s.selected && *f.frontier_b_count > 0 &&
             candidate_in(t, f, s, s.lo, options.complete_hi) && token == t.selection_tokens[0]) ||
            (separator && delimiter_step(separator_of(t, s), s.sep_pos, token) >= 0) ||
            (closer && delimiter_step(t.closer, s.close_pos, token) >= 0) ||
            operator_step(t, f, s, token, kind) >= 0;
        if (!other_branch)
          bias_tokens[static_cast<int64_t>(row) * t.layout.bias_capacity + n++] = token;
      }
      bias_count[row] = n;
    }
  }
}

// Add the escape bias to each row's tokens that commit to the escape marker.
__global__ void escape_bias_kernel(Tables t, const int32_t* bias_tokens,
                                   const int32_t* bias_count, float* logits,
                                   int64_t logits_stride, int32_t rows) {
  const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  for (int32_t k = 0; k < bias_count[row]; ++k) {
    logits[static_cast<int64_t>(row) * logits_stride +
           bias_tokens[static_cast<int64_t>(row) * t.layout.bias_capacity + k]] += t.escape_bias;
  }
}

// The log allowed mass of a logits row under its mask, and whether its top column is
// allowed, from one block's reductions over the row.
struct RowMass {
  float log_mass;
  bool top_allowed;
};

__device__ bool column_allowed(const uint32_t* words, int32_t mask_words, int32_t column) {
  const int32_t w = column >> 5;
  return w < mask_words && ((words[w] >> (column & 31)) & 1u);
}

constexpr int32_t kReleaseThreads = 256;  // release_kernel's block size, a power of two

__device__ RowMass row_mass(const float* logits, const uint32_t* words, int32_t mask_words,
                            int32_t columns) {
  __shared__ float best[kReleaseThreads];
  __shared__ int32_t best_at[kReleaseThreads];
  __shared__ float sums[kReleaseThreads];
  __shared__ float allowed_sums[kReleaseThreads];
  float m = -INFINITY;
  int32_t at = 0;
  for (int32_t c = threadIdx.x; c < columns; c += blockDim.x) {
    if (logits[c] > m) {
      m = logits[c];
      at = c;
    }
  }
  best[threadIdx.x] = m;
  best_at[threadIdx.x] = at;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = best[threadIdx.x + stride];
      const int32_t other_at = best_at[threadIdx.x + stride];
      if (other > best[threadIdx.x] ||
          (other == best[threadIdx.x] && other_at < best_at[threadIdx.x])) {
        best[threadIdx.x] = other;
        best_at[threadIdx.x] = other_at;
      }
    }
    __syncthreads();
  }
  const float top = best[0];
  if (!isfinite(top)) return RowMass{0.0f, true};  // no defined distribution: never release
  float sum = 0.0f, allowed_sum = 0.0f;
  for (int32_t c = threadIdx.x; c < columns; c += blockDim.x) {
    const float e = expf(logits[c] - top);
    sum += e;
    if (column_allowed(words, mask_words, c)) allowed_sum += e;
  }
  sums[threadIdx.x] = sum;
  allowed_sums[threadIdx.x] = allowed_sum;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      allowed_sums[threadIdx.x] += allowed_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return RowMass{logf(allowed_sums[0]) - logf(sums[0]),
                 column_allowed(words, mask_words, best_at[0])};
}

// One block of kReleaseThreads per logits row: a row inside a span whose rejected mass
// is too high (see Tables) gets the mask of the escape marker alone, from where it
// stands, in its own mask buffer. The mass is over the served vocabulary, at the logits'
// own temperature (vLLM hands logits processors the raw logits). Release is per step: a
// started escape marker goes on only while the rejected mass stays high or the model
// chooses it.
__global__ void release_kernel(Tables t, const int32_t* escape_pos, const uint32_t** sources,
                               uint32_t* mask, const float* logits, int64_t logits_stride,
                               int32_t columns) {
  const int32_t row = blockIdx.x;
  const int32_t pos = escape_pos[row];
  if (pos < 0) return;  // uniform across the block
  const RowMass mass = row_mass(logits + static_cast<int64_t>(row) * logits_stride,
                                sources[row], t.words, min(columns, t.vocab));
  const bool release = (t.release_top && !mass.top_allowed) ||
                       mass.log_mass < t.release_log_mass;
  if (!release) return;  // uniform: every thread reads the same reductions
  uint32_t* words = mask + static_cast<int64_t>(row) * t.words;
  __syncthreads();
  for (int32_t w = threadIdx.x; w < t.words; w += blockDim.x) words[w] = 0u;
  __syncthreads();
  if (threadIdx.x == 0) add_delimiter_tokens(t.escape, pos, words);
}

constexpr int32_t kApplyChunks = 16;  // blocks per logits row in apply_kernel

// Set every logit the row's mask disallows (and every column past the served
// vocabulary) to -inf, with kApplyChunks blocks per row (blockIdx.y is the row), so
// that a row inside a span, where nearly every column is written, uses the whole GPU.
// Each thread handles four consecutive columns and writes them with one 16-byte store
// when all four are blocked; free-text rows, where nearly every column is allowed,
// write almost nothing. Rows of inactive slots are left alone.
__global__ void apply_kernel(const uint8_t* active, const int32_t* row_slots,
                             const uint32_t* const* sources, int32_t mask_words, float* logits,
                             int64_t logits_stride, int32_t columns) {
  const int32_t row = blockIdx.y;
  if (!active[row_slots[row]]) return;  // uniform across the block
  const uint32_t* words = sources[row];
  if (words == nullptr) return;  // a row after a disallowed draft
  float* out = logits + static_cast<int64_t>(row) * logits_stride;
  const bool aligned = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
  const int32_t quads = (columns + 3) / 4;
  const int32_t per_chunk = (quads + kApplyChunks - 1) / kApplyChunks;
  const int32_t first = blockIdx.x * per_chunk;
  const int32_t stop = min(quads, first + per_chunk);
  for (int32_t q = first + threadIdx.x; q < stop; q += blockDim.x) {
    const int32_t c = q * 4;
    const uint32_t bits = (blocked_bits(words, c >> 5, mask_words, columns) >> (c & 31)) & 15u;
    if (bits == 0u) continue;
    if (bits == 15u && aligned && c + 4 <= columns) {
      reinterpret_cast<float4*>(out)[q] = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      continue;
    }
    for (int32_t k = 0; k < 4; ++k) {
      if ((bits >> k) & 1u) out[c + k] = -INFINITY;
    }
  }
}

template <typename T>
using HostArray = nb::ndarray<const T, nb::device::cpu, nb::c_contig>;
using Slots = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using Anywhere1D = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig>;
using Anywhere2D = nb::ndarray<const int32_t, nb::ndim<2>>;

cudaStream_t as_stream(uintptr_t stream) { return reinterpret_cast<cudaStream_t>(stream); }

class Engine {
 public:
  // tables: the named host arrays that src/xgraph/engine.py builds.
  Engine(nb::dict tables, int32_t vocab, int32_t words, int32_t max_sequences, bool repeats,
         float release_log_mass, bool release_top, bool derivation, float escape_bias,
         bool paths, bool bound_escapes, bool committed_paths, bool sitelinks_rank,
         bool unquoted_starts, bool expressions, bool end_after_answer, int32_t relations,
         int32_t observe, bool observe_select, bool observe_answer,
         bool observe_required, bool observe_no_loop, bool audit, int32_t audit_entities,
         int32_t audit_relations, int32_t menu, int32_t observe_width,
         int32_t observe_defer, bool menu_close_next, int32_t observe_terminal,
         bool request_path_options)
      : layout_(relation_words(tables, relations), observe, observe_no_loop, audit, menu,
                observe_terminal, escape_start_count(tables)),
        states_(max_sequences),
        fact_rows_(static_cast<size_t>(max_sequences) * layout_.memory_ints),
        fact_counts_(static_cast<size_t>(max_sequences) * kMemoryCounts),
        audit_scores_((audit || menu) ? static_cast<size_t>(max_sequences) * kAuditScores * 2 : 0),
        audit_alias_ranks_((audit || menu) ? static_cast<size_t>(max_sequences) *
                           size(tables, "relation_owners") : 0),
        audit_alias_active_((audit || menu) ? max_sequences : 0),
        path_options_(request_path_options ? static_cast<size_t>(max_sequences) * 10 : 0),
        cursors_(max_sequences),
        active_(max_sequences),
        tokens_(max_sequences),
        mask_(static_cast<size_t>(max_sequences) * words),
        row_masks_(static_cast<size_t>(max_sequences) * words),
        sources_(max_sequences),
        escape_pos_(max_sequences),
        bias_tokens_(static_cast<size_t>(max_sequences) * layout_.bias_capacity),
        bias_count_(max_sequences),
        row_capacity_(max_sequences),
        max_sequences_(max_sequences) {
    tables_.entities = Rows{upload<int32_t>(tables, "row_tokens"),
                            upload<int64_t>(tables, "row_offsets"),
                            upload<int32_t>(tables, "row_entities"),
                            static_cast<int32_t>(size(tables, "row_entities"))};
    if (tables.contains("terminal_order_config"))
      tables_.terminal_order_config = upload<uint32_t>(tables, "terminal_order_config");
    tables_.relations = Rows{upload<int32_t>(tables, "relation_tokens"),
                             upload<int64_t>(tables, "relation_offsets"),
                             upload<int32_t>(tables, "relation_owners"),
                             static_cast<int32_t>(size(tables, "relation_owners"))};
    tables_.relation_row_offsets = upload<int32_t>(tables, "relation_row_offsets");
    tables_.relation_rows = upload<int32_t>(tables, "relation_rows");
    tables_.adjacency = upload<int64_t>(tables, "adjacency_offsets");
    tables_.entity_row_offsets = upload<int64_t>(tables, "entity_row_offsets");
    tables_.entity_rows = upload<int32_t>(tables, "entity_rows");
    tables_.edge_relations = upload<int32_t>(tables, "edge_relations");
    tables_.edge_objects = upload<int32_t>(tables, "edge_objects");
    tables_.versions = Versions{upload<int64_t>(tables, "version_offsets"),
                                upload<int32_t>(tables, "version_subjects"),
                                upload<int64_t>(tables, "version_adjacency"),
                                upload<int32_t>(tables, "version_edge_relations"),
                                upload<int32_t>(tables, "version_edge_objects"),
                                static_cast<int32_t>(size<int64_t>(tables, "version_offsets") - 1)};
    tables_.root_mask = upload<uint32_t>(tables, "root_mask");
    tables_.wide_keys = upload<int64_t>(tables, "wide_keys");
    tables_.wide_masks = upload<uint32_t>(tables, "wide_masks");
    tables_.wide_count = static_cast<int32_t>(size<int64_t>(tables, "wide_keys"));
    tables_.free_next = upload<int8_t>(tables, "free_next");
    tables_.free_mask = upload<uint32_t>(tables, "free_mask");
    tables_.eos_mask = upload<uint32_t>(tables, "eos_mask");
    tables_.esc_next = upload<int8_t>(tables, "esc_next");
    tables_.esc_mask = upload<uint32_t>(tables, "esc_mask");
    tables_.closer = delimiter(tables, "closer");
    tables_.separator = delimiter(tables, "separator");
    tables_.escape = delimiter(tables, "escape");
    tables_.vocab = vocab;
    tables_.words = words;
    tables_.repeats = repeats;
    tables_.release_log_mass = release_log_mass;
    tables_.release_top = release_top;
    tables_.derivation = derivation;
    tables_.escape_bias = escape_bias;
    tables_.paths = paths;
    tables_.end_after_answer = end_after_answer;
    tables_.observe = observe;
    tables_.request_path_options = path_options_.get();
    if (request_path_options && menu && observe_terminal) {
      tables_.request_terminal_tokens = upload<int32_t>(tables, "request_terminal_tokens");
      tables_.request_terminal_offsets = upload<int32_t>(tables, "request_terminal_offsets");
    }
    if (request_path_options && menu && observe) {
      tables_.request_menu_tokens = upload<int32_t>(tables, "request_menu_tokens");
      tables_.request_menu_offsets = upload<int32_t>(tables, "request_menu_offsets");
    }
    tables_.observe_select = observe_select;
    tables_.observe_answer = observe_answer;
    tables_.observe_required = observe_required;
    tables_.observe_width = observe_width;
    tables_.observe_defer = observe_defer;
    tables_.menu_close_next = menu_close_next;
    if (observe_defer) {
      tables_.defer_tokens = upload<int32_t>(tables, "defer_tokens");
      tables_.defer_length = static_cast<int32_t>(size(tables, "defer_tokens"));
    }
    if (menu_close_next) {
      tables_.decision_tokens = upload<int32_t>(tables, "decision_tokens");
      tables_.decision_offsets = upload<int32_t>(tables, "decision_offsets");
    }
    tables_.observe_no_loop = observe_no_loop;
    tables_.audit = audit; tables_.audit_entities = audit_entities; tables_.audit_relations = audit_relations;
    tables_.observe_terminal = observe_terminal;
    tables_.menu = menu;
    tables_.layout = layout_;
    tables_.audit_scores = audit_scores_.get();
    if (audit || menu || observe_terminal) {
      tables_.audit_tokens = upload<int32_t>(tables, "audit_tokens");
      tables_.audit_offsets = upload<int32_t>(tables, "audit_offsets");
    }
    if (audit || menu) {
      tables_.audit_primary_rows = upload<int32_t>(tables, "audit_primary_rows");
      tables_.audit_default_alias_ranks = upload<int32_t>(tables, "audit_default_alias_ranks");
      tables_.audit_alias_ranks = audit_alias_ranks_.get();
      tables_.audit_alias_active = audit_alias_active_.get();
    }
    if (audit) {
      tables_.audit_command_tokens = upload<int32_t>(tables, "audit_command_tokens");
      tables_.audit_command_offsets = upload<int32_t>(tables, "audit_command_offsets");
    }
    if (observe_no_loop) tables_.loop_label_groups = upload<int32_t>(tables, "loop_label_groups");
    if (observe_select) {
      tables_.selection_length = nb::len(nb::handle(tables["selection_tokens"]));
      tables_.selection_tokens = upload<int32_t>(tables, "selection_tokens");
    }
    if (observe_answer || audit || observe_terminal) {
      tables_.answer_length = nb::len(nb::handle(tables["answer_tokens"]));
      tables_.answer_tokens = upload<int32_t>(tables, "answer_tokens");
    }
    if (observe > 0 || menu > 0) {
      tables_.observation_tokens = upload<int32_t>(tables, "observation_tokens");
      tables_.observation_offsets = upload<int32_t>(tables, "observation_offsets");
    }
    tables_.relation_mode = relations;
    tables_.relation_count = static_cast<int32_t>(size(tables, "relation_row_offsets") - 1);
    tables_.relation_words = layout_.relation_words;
    tables_.relation_successors = upload<uint32_t>(tables, "relation_successors");
    tables_.bound_escapes = bound_escapes;
    tables_.committed_paths = committed_paths;
    tables_.sitelinks_rank = sitelinks_rank;
    tables_.entity_sitelinks = upload<int32_t>(tables, "entity_sitelinks");
    tables_.unquoted_starts = unquoted_starts;
    tables_.unquoted_rows = upload<int32_t>(tables, "unquoted_rows");
    if (unquoted_starts && size(tables, "unquoted_rows") != size(tables, "row_entities") + 1) {
      throw std::invalid_argument("unquoted starts need one unquoted_rows entry per row and one");
    }
    tables_.expressions = expressions;
    tables_.operand_separator = delimiter(tables, "operand");
    int32_t operators = 0;
    while (tables.contains(("operator" + std::to_string(operators) + "_offsets").c_str())) {
      if (operators == kMaxOperators) {
        throw std::invalid_argument("the engine supports at most " +
                                    std::to_string(kMaxOperators) + " operators");
      }
      tables_.operators[operators] = delimiter(tables, "operator" + std::to_string(operators));
      ++operators;
    }
    tables_.operator_count = operators;
    const auto lcp = nb::cast<HostArray<int32_t>>(tables["operator_lcp"]);
    if (lcp.size() != static_cast<size_t>(operators * operators)) {
      throw std::invalid_argument("operator_lcp needs one entry per pair of operators");
    }
    for (int32_t k = 0; k < operators; ++k) {
      for (int32_t j = 0; j < operators; ++j) {
        tables_.operator_lcp[k * kMaxOperators + j] = lcp.data()[k * operators + j];
      }
    }
    tables_.date_entities = upload<int32_t>(tables, "date_entities");
    tables_.date_lo = upload<int32_t>(tables, "date_lo");
    tables_.date_hi = upload<int32_t>(tables, "date_hi");
    tables_.date_count = static_cast<int32_t>(size(tables, "date_entities"));
    tables_.date_keys = upload<int32_t>(tables, "date_keys");
    tables_.verdict_entities = upload<int32_t>(tables, "verdict_entities");
    tables_.stop_label_offsets = upload<int64_t>(tables, "stop_label_offsets");
    tables_.stop_label_words = upload<uint8_t>(tables, "stop_label_words");
    if (size<int64_t>(tables, "stop_label_offsets") > 0) {
      if (!committed_paths ||
          size<int64_t>(tables, "stop_label_offsets") != size(tables, "row_entities") + 1) {
        throw std::invalid_argument("STOP needs committed paths and word offsets for every label row");
      }
      tables_.stop_relation_count = static_cast<int32_t>(size(tables, "relation_row_offsets") - 1);
      stop_relations_ = DeviceArray<int32_t>(static_cast<size_t>(max_sequences) *
                                            tables_.stop_relation_count * 3);
      stop_words_ = DeviceArray<uint8_t>(static_cast<size_t>(max_sequences) * kMaxStopBytes);
      stop_offsets_ = DeviceArray<int32_t>(static_cast<size_t>(max_sequences) * (kMaxStopWords + 2));
      tables_.stop_relations = stop_relations_.get();
      tables_.stop_words = stop_words_.get();
      tables_.stop_offsets = stop_offsets_.get();
    }
    if (operators > 2 && size(tables, "verdict_entities") != 2) {
      throw std::invalid_argument("a value comparison needs the verdict entities, yes and no");
    }
    if (size(tables, "date_lo") != size(tables, "date_entities") ||
        size(tables, "date_hi") != size(tables, "date_entities") ||
        size(tables, "date_keys") != size(tables, "date_entities")) {
      throw std::invalid_argument("the date table needs both bounds of every date entity");
    }
    reset(0);
    check(cudaDeviceSynchronize(), "engine setup");
  }

  void reset(uintptr_t stream) {
    if (path_options_.size()) check(cudaMemsetAsync(
        path_options_.get(), 0xff, path_options_.size() * sizeof(int32_t), as_stream(stream)),
        "reset path options");
    if (audit_alias_active_.size())
      check(cudaMemsetAsync(audit_alias_active_.get(), 0, audit_alias_active_.size(),
                            as_stream(stream)), "reset alias priorities");
    check(cudaMemsetAsync(states_.get(), 0, states_.size() * sizeof(State), as_stream(stream)),
          "reset");
    check(cudaMemsetAsync(cursors_.get(), 0xff, cursors_.size() * sizeof(int32_t),
                          as_stream(stream)),
          "reset");
    check(cudaMemsetAsync(active_.get(), 0, active_.size(), as_stream(stream)), "reset");
    check(cudaMemsetAsync(fact_counts_.get(), 0, fact_counts_.size() * sizeof(int32_t),
                          as_stream(stream)),
          "reset");
  }

  // Start (or, with on false, stop) constraining the listed slots: each goes back to
  // the initial state and will follow its history from the prompt length.
  void begin(Slots slots, bool on, uintptr_t stream) {
    const int32_t count = slot_count(slots);
    if (count == 0) return;
    begin_kernel<<<(count + 127) / 128, 128, 0, as_stream(stream)>>>(
        states_.get(), fact_counts_.get(), cursors_.get(), active_.get(),
        audit_alias_active_.get(), path_options_.get(), slots.data(), count,
        on ? 1 : 0);
    check(cudaGetLastError(), "begin kernel");
  }

  // Derivation mode: a slot's reached rows (sorted, at most kMaxReached), as if a fact span
  // had reached them: its spans then start from those entities. Call after begin.
  void set_path_options(int32_t slot, HostArray<int32_t> options, uintptr_t stream) {
    if (!path_options_.size() || slot < 0 || slot >= max_sequences_ || options.size() != 10)
      throw std::invalid_argument("invalid request path options/slot");
    check(cudaMemcpyAsync(path_options_.get() + static_cast<int64_t>(slot) * 10,
                          options.data(), 10 * sizeof(int32_t), cudaMemcpyHostToDevice,
                          as_stream(stream)), "set path options");
    check(cudaStreamSynchronize(as_stream(stream)), "set path options");
  }

  void set_reached(int32_t slot, HostArray<int32_t> rows, uintptr_t stream) {
    if (slot < 0 || slot >= max_sequences_) {
      throw std::invalid_argument("slot must be between 0 and max_sequences - 1, got " +
                                  std::to_string(slot));
    }
    if (rows.size() == 0) {
      throw std::invalid_argument("anchor has no writable label rows");
    }
    const int32_t count = static_cast<int32_t>(rows.size() < kMaxReached ? rows.size() : kMaxReached);
    int32_t* memory = fact_rows_.get() + static_cast<int64_t>(slot) * tables_.layout.memory_ints + kMaxFacts * 3;
    int32_t* counts = fact_counts_.get() + static_cast<int64_t>(slot) * kMemoryCounts;
    if (count > 0) {
      check(cudaMemcpyAsync(memory, rows.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice,
                            as_stream(stream)),
            "set_reached");
    }
    check(cudaMemcpyAsync(counts + 1, &count, sizeof(int32_t), cudaMemcpyHostToDevice,
                          as_stream(stream)),
          "set_reached");
    check(cudaStreamSynchronize(as_stream(stream)), "set_reached");
  }

  // Let a slot read KG version `version` (1 to the versions registered; 0: the base KG)
  // until its next begin. Call after begin.
  void set_version(int32_t slot, int32_t version, uintptr_t stream) {
    if (slot < 0 || slot >= max_sequences_) {
      throw std::invalid_argument("slot must be between 0 and max_sequences - 1, got " +
                                  std::to_string(slot));
    }
    if (version < 0 || version > tables_.versions.count) {
      throw std::invalid_argument("KG version must be between 0 and " +
                                  std::to_string(tables_.versions.count) + ", got " +
                                  std::to_string(version));
    }
    set_version_kernel<<<1, 1, 0, as_stream(stream)>>>(fact_counts_.get(), slot, version);
    check(cudaGetLastError(), "set_version kernel");
  }

  void set_alias_ranks(int32_t slot, HostArray<int32_t> priorities, uintptr_t stream) {
    if (!(tables_.audit || tables_.menu) || slot < 0 || slot >= max_sequences_ ||
        priorities.size() != static_cast<size_t>(tables_.relations.count))
      throw std::invalid_argument("invalid audit alias priorities/slot");
    if (priorities.size()) check(cudaMemcpyAsync(
        audit_alias_ranks_.get() + static_cast<int64_t>(slot) * priorities.size(),
        priorities.data(), priorities.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
        as_stream(stream)), "alias priorities");
    check(cudaMemsetAsync(audit_alias_active_.get() + slot, 1, 1, as_stream(stream)),
          "alias priorities active");
    check(cudaStreamSynchronize(as_stream(stream)), "alias priorities");
  }

  // Install immutable per-request relation scores. begin/end/reset clear the active count.
  void set_relation_scores(int32_t slot, HostArray<int32_t> scores, uintptr_t stream) {
    if (!(tables_.audit || tables_.menu) || slot < 0 || slot >= max_sequences_ || scores.size() % 2 ||
        scores.size() > kAuditScores * 2) throw std::invalid_argument("invalid audit relation scores/slot");
    for (size_t i = 0; i < scores.size(); i += 2) {
      if (scores.data()[i] < 0 || scores.data()[i] >= tables_.relation_count ||
          scores.data()[i + 1] < 0 || scores.data()[i + 1] >= kAuditScores)
        throw std::invalid_argument("invalid audit relation id/rank");
    }
    const int32_t count = scores.size() / 2;
    if (count) check(cudaMemcpyAsync(audit_scores_.get() + static_cast<int64_t>(slot) * kAuditScores * 2,
        scores.data(), scores.size() * sizeof(int32_t), cudaMemcpyHostToDevice, as_stream(stream)), "audit scores");
    check(cudaMemcpyAsync(fact_counts_.get() + static_cast<int64_t>(slot) * kMemoryCounts + 6,
        &count, sizeof(count), cudaMemcpyHostToDevice, as_stream(stream)), "audit score count");
    check(cudaStreamSynchronize(as_stream(stream)), "audit scores");
  }

  // speculative rows keep the real slot's tables and copy only the used-word state.
  void set_stop(int32_t slot, HostArray<int32_t> relations, HostArray<uint8_t> words,
                 HostArray<int32_t> offsets, uintptr_t stream) {
    if (slot < 0 || slot >= max_sequences_) {
      throw std::invalid_argument("STOP slot out of range: " + std::to_string(slot));
    }
    if (tables_.stop_offsets == nullptr) throw std::invalid_argument("STOP is not enabled");
    if (relations.size() != static_cast<size_t>(tables_.stop_relation_count) * 3 ||
        offsets.size() < 1 || offsets.size() > kMaxStopWords + 1 || words.size() > kMaxStopBytes) {
      throw std::invalid_argument("STOP needs relation masks and bounded question words");
    }
    const int32_t count = static_cast<int32_t>(offsets.size()) - 1;
    if (offsets.data()[0] != 0 || offsets.data()[count] != static_cast<int32_t>(words.size())) {
      throw std::invalid_argument("STOP word offsets must cover the word bytes");
    }
    for (int32_t k = 0; k < count; ++k) {
      if (offsets.data()[k] >= offsets.data()[k + 1]) {
        throw std::invalid_argument("STOP words must be nonempty, with increasing offsets");
      }
    }
    std::vector<int32_t> packed(kMaxStopWords + 2, 0);
    packed[0] = count;
    for (int32_t k = 0; k <= count; ++k) packed[k + 1] = offsets.data()[k];
    const cudaStream_t cuda_stream = as_stream(stream);
    if (relations.size() > 0) {
      check(cudaMemcpyAsync(stop_relations_.get() + static_cast<int64_t>(slot) * relations.size(),
                            relations.data(), relations.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cuda_stream), "set_stop relations");
    }
    if (words.size() > 0) {
      check(cudaMemcpyAsync(stop_words_.get() + static_cast<int64_t>(slot) * kMaxStopBytes,
                            words.data(), words.size(), cudaMemcpyHostToDevice, cuda_stream),
            "set_stop words");
    }
    check(cudaMemcpyAsync(stop_offsets_.get() + static_cast<int64_t>(slot) * packed.size(),
                          packed.data(), packed.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, cuda_stream), "set_stop offsets");
    const int32_t enabled = 1;
    check(cudaMemcpyAsync(fact_counts_.get() + static_cast<int64_t>(slot) * kMemoryCounts + 5,
                          &enabled, sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream),
          "set_stop enabled");
    check(cudaStreamSynchronize(cuda_stream), "set_stop");
  }

  // Start a slot inside a span, as if its prompt's last tokens had written the span opener
  // (a forced opener): its first token continues the span. Call after begin, which
  // returns the slot to free text; the memory (anchor, KG version) is kept.
  void open_span(int32_t slot, uintptr_t stream) {
    if (slot < 0 || slot >= max_sequences_) {
      throw std::invalid_argument("slot must be between 0 and max_sequences - 1, got " +
                                  std::to_string(slot));
    }
    open_kernel<<<1, 1, 0, as_stream(stream)>>>(tables_, states_.get(), slot);
    check(cudaGetLastError(), "open kernel");
  }

  // Advance the listed active slots over their token histories (one row per slot of
  // history, prompt_len and total_len per slot), all read in place.
  void sync(Slots slots, Anywhere2D history, Anywhere1D prompt_len, Anywhere1D total_len,
            uintptr_t stream) {
    const int32_t count = slot_count(slots);
    for (const auto* lengths : {&prompt_len, &total_len}) {
      if (static_cast<int64_t>(lengths->shape(0)) < max_sequences_) {
        throw std::invalid_argument("prompt_len and total_len need one entry per slot");
      }
      on_device(*lengths, "lengths");
    }
    on_device(history, "history");
    if (static_cast<int64_t>(history.shape(0)) < max_sequences_ || history.stride(1) != 1) {
      throw std::invalid_argument("history needs one contiguous row per slot");
    }
    if (count == 0) return;
    sync_kernel<<<(count + 127) / 128, 128, 0, as_stream(stream)>>>(
        tables_, states_.get(), fact_rows_.get(), fact_counts_.get(), cursors_.get(),
        active_.get(), slots.data(), count,
        history.data(), history.stride(0), prompt_len.data(), total_len.data());
    check(cudaGetLastError(), "sync kernel");
  }

  // Mask the rows of fp32 logits in place: row i belongs to slot row_slots[i].
  // local_pos and input_ids (one entry per row, or both empty) describe draft rows
  // under speculative decoding; see row_mask_kernel.
  void apply_mask(nb::ndarray<float, nb::ndim<2>, nb::device::cuda> logits, Slots row_slots,
                  Slots local_pos, Slots input_ids, uintptr_t stream) {
    const int64_t rows64 = static_cast<int64_t>(row_slots.shape(0));
    if (static_cast<int64_t>(logits.shape(0)) != rows64 || logits.stride(1) != 1) {
      throw std::invalid_argument("logits need one contiguous row per entry of row_slots");
    }
    if (rows64 > std::numeric_limits<int32_t>::max() / 65535) {
      throw std::invalid_argument("too many logits rows");
    }
    const int32_t rows = static_cast<int32_t>(rows64);
    const bool drafts = local_pos.shape(0) > 0;
    if (drafts && (static_cast<int64_t>(local_pos.shape(0)) != rows64 ||
                   static_cast<int64_t>(input_ids.shape(0)) != rows64)) {
      throw std::invalid_argument("local_pos and input_ids need one entry per logits row");
    }
    if (rows == 0) return;
    if (rows > row_capacity_) {  // more rows than before: drafts; grow once per new size
      check(cudaStreamSynchronize(as_stream(stream)), "row buffer growth");
      row_masks_ = DeviceArray<uint32_t>(static_cast<size_t>(rows) * tables_.words);
      sources_ = DeviceArray<const uint32_t*>(rows);
      escape_pos_ = DeviceArray<int32_t>(rows);
      bias_tokens_ = DeviceArray<int32_t>(static_cast<size_t>(rows) * tables_.layout.bias_capacity);
      bias_count_ = DeviceArray<int32_t>(rows);
      row_capacity_ = rows;
    }
    row_mask_kernel<<<rows, 256, tables_.layout.memory_ints * sizeof(int32_t), as_stream(stream)>>>(
        tables_, states_.get(), fact_rows_.get(), fact_counts_.get(), active_.get(),
        row_slots.data(),
        drafts ? local_pos.data() : nullptr, drafts ? input_ids.data() : nullptr,
        row_masks_.get(), sources_.get(), escape_pos_.get(), bias_tokens_.get(),
        bias_count_.get());
    check(cudaGetLastError(), "row mask kernel");
    if (tables_.escape_bias != 0.0f) {
      escape_bias_kernel<<<(rows + 127) / 128, 128, 0, as_stream(stream)>>>(
          tables_, bias_tokens_.get(), bias_count_.get(), logits.data(), logits.stride(0), rows);
      check(cudaGetLastError(), "escape bias kernel");
    }
    if (tables_.release_top || tables_.release_log_mass > -INFINITY) {
      release_kernel<<<rows, kReleaseThreads, 0, as_stream(stream)>>>(
          tables_, escape_pos_.get(), sources_.get(), row_masks_.get(), logits.data(),
          logits.stride(0), static_cast<int32_t>(logits.shape(1)));
      check(cudaGetLastError(), "release kernel");
    }
    apply_kernel<<<dim3(kApplyChunks, rows), 256, 0, as_stream(stream)>>>(
        active_.get(), row_slots.data(), sources_.get(), tables_.words, logits.data(),
        logits.stride(0), static_cast<int32_t>(logits.shape(1)));
    check(cudaGetLastError(), "apply kernel");
  }

  // Advance the first len(tokens) sequences. Tokens on the GPU are read in place;
  // tokens in host memory are copied in first.
  void advance(nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> tokens, uintptr_t stream) {
    const int32_t sequences = static_cast<int32_t>(tokens.shape(0));
    check_sequences(sequences);
    const int32_t* device_tokens = tokens.data();
    if (tokens.device_type() != nb::device::cuda::value) {
      check(cudaMemcpyAsync(tokens_.get(), tokens.data(), sequences * sizeof(int32_t),
                            cudaMemcpyHostToDevice, as_stream(stream)),
            "token upload");
      device_tokens = tokens_.get();
    }
    if (sequences == 0) return;
    advance_kernel<<<(sequences + 127) / 128, 128, 0, as_stream(stream)>>>(
        tables_, states_.get(), fact_rows_.get(), fact_counts_.get(), device_tokens, sequences);
    check(cudaGetLastError(), "advance kernel");
  }

  void fill_mask(int32_t sequences, uintptr_t stream) {
    check_sequences(sequences);
    if (sequences == 0) return;
    mask_kernel<<<sequences, 256, 0, as_stream(stream)>>>(
        tables_, states_.get(), fact_rows_.get(), fact_counts_.get(), mask_.get());
    check(cudaGetLastError(), "mask kernel");
  }

  // The mask buffer as a device array: int32 words, bit i of word w is token 32w + i.
  nb::ndarray<int32_t, nb::ndim<2>, nb::device::cuda> mask(nb::handle self) {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    const size_t shape[2] = {static_cast<size_t>(max_sequences_),
                             static_cast<size_t>(tables_.words)};
    return nb::ndarray<int32_t, nb::ndim<2>, nb::device::cuda>(
        reinterpret_cast<int32_t*>(mask_.get()), 2, shape, self, nullptr, nb::dtype<int32_t>(),
        nb::device::cuda::value, device);
  }

  nb::ndarray<nb::numpy, int32_t, nb::ndim<2>> mask_host(int32_t sequences) {
    check_sequences(sequences);
    return copy_to_host(reinterpret_cast<const int32_t*>(mask_.get()), sequences, tables_.words);
  }

  nb::ndarray<nb::numpy, int32_t, nb::ndim<2>> states_host(int32_t sequences) {
    check_sequences(sequences);
    return copy_to_host(reinterpret_cast<const int32_t*>(states_.get()), sequences, kStateFields);
  }

  int32_t max_sequences() const { return max_sequences_; }

 private:
  static int32_t escape_start_count(const nb::dict& tables) {
    const auto offsets = nb::cast<HostArray<int32_t>>(tables["escape_offsets"], false);
    return offsets.size() > 1 ? offsets.data()[1] - offsets.data()[0] : 0;
  }
  static int32_t relation_words(const nb::dict& tables, int32_t mode) {
    return mode == 2 ? static_cast<int32_t>((size(tables, "relation_row_offsets") - 1 + 31) / 32) : 0;
  }
  template <typename T = int32_t>
  static size_t size(const nb::dict& tables, const char* name) {
    return nb::cast<HostArray<T>>(tables[name], false).size();
  }

  // Copy one named host array to the GPU; the engine keeps it until it is destroyed.
  template <typename T>
  const T* upload(const nb::dict& tables, const char* name) {
    if (!tables.contains(name)) {
      throw std::invalid_argument(std::string("missing engine table ") + name);
    }
    const auto host = nb::cast<HostArray<T>>(tables[name]);
    auto device = std::make_unique<DeviceArray<T>>(host.data(), host.size());
    const T* ptr = device->get();
    owned_.push_back(std::shared_ptr<void>(device.release(), [](void* p) {
      delete static_cast<DeviceArray<T>*>(p);
    }));
    return ptr;
  }

  Delimiter delimiter(const nb::dict& tables, const std::string& name) {
    const std::string offsets = name + "_offsets";
    return Delimiter{upload<int32_t>(tables, offsets.c_str()),
                     upload<int32_t>(tables, (name + "_tokens").c_str()),
                     upload<int32_t>(tables, (name + "_next").c_str()),
                     static_cast<int32_t>(size(tables, offsets.c_str()) - 1)};
  }

  int32_t slot_count(const Slots& slots) const {
    const int64_t count = static_cast<int64_t>(slots.shape(0));
    if (count > max_sequences_) {
      throw std::invalid_argument("more slots than max_sequences");
    }
    return static_cast<int32_t>(count);
  }

  template <typename Array>
  static void on_device(const Array& array, const char* name) {
    const int type = array.device_type();
    if (type != nb::device::cuda::value && type != nb::device::cuda_host::value) {
      throw std::invalid_argument(std::string(name) + " must be on the GPU or in pinned host memory");
    }
  }

  void check_sequences(int32_t sequences) const {
    if (sequences < 0 || sequences > max_sequences_) {
      throw std::invalid_argument("sequences must be between 0 and max_sequences (" +
                                  std::to_string(max_sequences_) + "), got " +
                                  std::to_string(sequences));
    }
  }

  static nb::ndarray<nb::numpy, int32_t, nb::ndim<2>> copy_to_host(const int32_t* device,
                                                                   int32_t sequences,
                                                                   int32_t columns) {
    check(cudaDeviceSynchronize(), "engine step");
    const size_t count = static_cast<size_t>(sequences) * columns;
    int32_t* host = new int32_t[count > 0 ? count : 1];
    nb::capsule owner(host, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
    if (count > 0) {
      check(cudaMemcpy(host, device, count * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "copy to host");
    }
    const size_t shape[2] = {static_cast<size_t>(sequences), static_cast<size_t>(columns)};
    return nb::ndarray<nb::numpy, int32_t, nb::ndim<2>>(host, 2, shape, owner);
  }

  std::vector<std::shared_ptr<void>> owned_;
  const MemoryLayout layout_;
  DeviceArray<State> states_;
  DeviceArray<int32_t> fact_rows_;    // [max_sequences, tables_.layout.memory_ints]: see Memory
  DeviceArray<int32_t> fact_counts_;  // [max_sequences, kMemoryCounts]
  DeviceArray<int32_t> audit_scores_;
  DeviceArray<int32_t> audit_alias_ranks_;
  DeviceArray<uint8_t> audit_alias_active_;
  DeviceArray<int32_t> path_options_;
  DeviceArray<int32_t> stop_relations_;  // optional immutable request tables (see Tables)
  DeviceArray<uint8_t> stop_words_;
  DeviceArray<int32_t> stop_offsets_;
  DeviceArray<int32_t> cursors_;
  DeviceArray<uint8_t> active_;
  DeviceArray<int32_t> tokens_;
  DeviceArray<uint32_t> mask_;
  DeviceArray<uint32_t> row_masks_;       // apply_mask: computed masks, one per logits row
  DeviceArray<const uint32_t*> sources_;  // apply_mask: where each row's mask lives
  DeviceArray<int32_t> escape_pos_;       // apply_mask: each row's escape position, or -1
  DeviceArray<int32_t> bias_tokens_;      // apply_mask: each row's escape-committing tokens
  DeviceArray<int32_t> bias_count_;       // apply_mask: how many
  int32_t row_capacity_;                  // logits rows the two buffers above hold
  int32_t max_sequences_;
  Tables tables_{};
};

}  // namespace

NB_MODULE(_engine, m) {
  m.doc() = "CUDA kernels of the xgraph engine; use xgraph.engine.Engine.";
  m.attr("PHASES") = nb::make_tuple("free text", "span", "escaped", "error");
  m.attr("SEGMENTS") = nb::make_tuple("subject", "relation", "object");
  m.attr("CLOSED") = nb::make_tuple("none", "unresolved", "entity span", "fact span", "path span",
                                    "comparison", "relation plan");
  m.attr("MAX_OPERATORS") = kMaxOperators;
  m.attr("MAX_FACTS") = kMaxFacts;
  m.attr("MAX_REACHED") = kMaxReached;
  m.attr("STATE_FIELDS") = nb::make_tuple(
      "phase", "pos", "segment", "depth", "lo", "hi", "close_pos", "close_lo", "close_hi",
      "sep_pos", "sep_lo", "sep_hi", "esc_pos", "subj_lo", "subj_hi", "rel_lo", "rel_hi", "subj_key", "hops",
      "op_pos", "op_kind", "op", "operand",
      "stop_used_lo", "stop_used_hi", "answer_complete",
      "last_kind", "last_lo", "last_hi", "last_subj_lo", "last_subj_hi", "last_rel_lo",
      "last_rel_hi", "observe_pos", "observe_lo", "observe_hi", "observe_depth",
      "observe_offset", "observe_length", "observed", "observe_count", "observe_more",
      "select_pos", "selecting", "selected", "select_segment", "select_lo", "select_hi",
      "select_depth", "answer_pos", "loop_count", "audit_phase", "audit_item", "audit_offset",
      "audit_items", "audit_choices", "audit_pos", "audit_action", "audit_hops",
      "defer_count", "defer_pos", "decision_phase", "decision_pos", "decision_choices");
  nb::class_<Engine>(m, "Engine")
      .def(nb::init<nb::dict, int32_t, int32_t, int32_t, bool, float, bool, bool, float, bool,
                    bool, bool, bool, bool, bool, bool, int32_t, int32_t, bool, bool, bool, bool,
                    bool, int32_t, int32_t, int32_t, int32_t, int32_t, bool, int32_t, bool>(),
           nb::arg("tables"),
           nb::arg("vocab"), nb::arg("words"), nb::arg("max_sequences"), nb::arg("repeats") = true,
           nb::arg("release_log_mass") = -INFINITY, nb::arg("release_top") = false,
           nb::arg("derivation") = false, nb::arg("escape_bias") = 0.0f,
           nb::arg("paths") = false, nb::arg("bound_escapes") = true,
           nb::arg("committed_paths") = false, nb::arg("sitelinks_rank") = false,
           nb::arg("unquoted_starts") = false, nb::arg("expressions") = false,
           nb::arg("end_after_answer") = false, nb::arg("relations") = 0,
           nb::arg("observe") = 0, nb::arg("observe_select") = false,
           nb::arg("observe_answer") = false, nb::arg("observe_required") = false,
           nb::arg("observe_no_loop") = false, nb::arg("audit") = false,
           nb::arg("audit_entities") = 20, nb::arg("audit_relations") = 8, nb::arg("menu") = 0,
           nb::arg("observe_width") = 0, nb::arg("observe_defer") = 0,
           nb::arg("menu_close_next") = false, nb::arg("observe_terminal") = 0,
           nb::arg("request_path_options") = false)
      .def("set_path_options", &Engine::set_path_options, nb::arg("slot"), nb::arg("options"),
           nb::arg("stream"))
      .def("reset", &Engine::reset, nb::arg("stream"))
      .def("advance", &Engine::advance, nb::arg("tokens"), nb::arg("stream"))
      .def("set_reached", &Engine::set_reached, nb::arg("slot"), nb::arg("rows"),
           nb::arg("stream"))
      .def("set_version", &Engine::set_version, nb::arg("slot"), nb::arg("version"),
           nb::arg("stream"))
      .def("set_relation_scores", &Engine::set_relation_scores, nb::arg("slot"), nb::arg("scores"),
           nb::arg("stream"))
      .def("set_alias_ranks", &Engine::set_alias_ranks, nb::arg("slot"), nb::arg("priorities"),
           nb::arg("stream"))
      .def("set_stop", &Engine::set_stop, nb::arg("slot"), nb::arg("relations"),
           nb::arg("words"), nb::arg("offsets"), nb::arg("stream"))
      .def("open_span", &Engine::open_span, nb::arg("slot"), nb::arg("stream"))
      .def("fill_mask", &Engine::fill_mask, nb::arg("sequences"), nb::arg("stream"))
      .def("begin", &Engine::begin, nb::arg("slots"), nb::arg("on"), nb::arg("stream"))
      .def("sync", &Engine::sync, nb::arg("slots"), nb::arg("history"), nb::arg("prompt_len"),
           nb::arg("total_len"), nb::arg("stream"))
      .def("apply_mask", &Engine::apply_mask, nb::arg("logits"), nb::arg("row_slots"),
           nb::arg("local_pos"), nb::arg("input_ids"), nb::arg("stream"))
      .def("mask", [](nb::handle_t<Engine> self) { return nb::cast<Engine&>(self).mask(self); })
      .def("mask_host", &Engine::mask_host, nb::arg("sequences"))
      .def("states_host", &Engine::states_host, nb::arg("sequences"))
      .def_prop_ro("max_sequences", &Engine::max_sequences);
}
