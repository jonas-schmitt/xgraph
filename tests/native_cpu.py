"""Sequential extraction of production native helpers, never CUDA evidence.

Only CUDA's mask clear/barrier is replaced: clear once, then run each virtual
thread. Device helper bodies and Python native tables come from this checkout.
"""

from __future__ import annotations

import ctypes as ct
import re
import subprocess
from pathlib import Path

import numpy as np

from xgraph.engine import Engine, closed_span, mask_words, native_tables

SOURCE = Path(__file__).resolve().parents[1] / "csrc/engine.cu"


def function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def compile_cpp(directory, code, *, shared=False):
    cpp, binary = directory / "native.cpp", directory / ("native.so" if shared else "native")
    cpp.write_text(code)
    result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O1",
            *(["-shared", "-fPIC"] if shared else []),
            str(cpp),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def compile_helpers(directory):
    source = SOURCE.read_text()
    definitions = source[source.index("// Table codes") : source.index("void check(cudaError_t")]
    helpers = source[
        source.index("// Rows of the label index") : source.index("__global__ void advance_kernel")
    ]
    mask = source[
        source.index("__device__ void set_token") : source.index("__global__ void mask_kernel")
    ]
    mask, clears = re.subn(
        r"for \(int32_t w = threadIdx.x; w < t.words; w \+= blockDim.x\) "
        r"out\[w\] = (?:0u|base \? base\[w\] : 0u);",
        "",
        mask,
    )
    assert clears == 3, "update the sequential mask seam for the changed CUDA initialization"
    header = """
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
using std::min; using std::max;
#define __device__
#define __syncthreads() ((void)0)
struct Dim { int x; }; static Dim blockDim{32}, threadIdx{0};
inline void atomicOr(uint32_t* p, uint32_t x) { *p |= x; }
"""
    declarations = helpers[
        helpers.index("struct Tables {") + len("struct Tables {") : helpers.index(
            "// Only the view changes"
        )
    ]
    declarations = re.sub(r"//[^\n]*", "", declarations)
    pointers, integers = {}, []
    for declaration in declarations.split(";"):
        match = re.fullmatch(
            r"\s*(?:const\s+)?(int32_t|int64_t|uint8_t|uint32_t|int8_t|bool|float)"
            r"\s*(\*)?\s*(.*?)\s*",
            declaration,
            re.S,
        )
        if not match or "[" in match[3]:
            continue
        for name in match[3].split(","):
            if match[2]:
                pointers[name.strip()] = match[1]
            else:
                integers.append(name.strip())
    for prefix in ("entities", "relations"):
        for name, dtype in (("tokens", "int32_t"), ("offsets", "int64_t"), ("owner", "int32_t")):
            pointers[f"{prefix}.{name}"] = dtype
        integers.append(f"{prefix}.count")
    for name, dtype in (
        ("offsets", "int64_t"),
        ("subjects", "int32_t"),
        ("adjacency", "int64_t"),
        ("edge_relations", "int32_t"),
        ("edge_objects", "int32_t"),
    ):
        pointers[f"versions.{name}"] = dtype
    integers.append("versions.count")
    for prefix in (
        "closer",
        "separator",
        "escape",
        "operand_separator",
        *(f"operators[{k}]" for k in range(4)),
    ):
        for name in ("offsets", "tokens", "next_pos"):
            pointers[f"{prefix}.{name}"] = "int32_t"
        integers.append(f"{prefix}.length")
    wrapper = """
static Tables t{}; static State state{};
static int32_t mem[10000]{}, counts[kMemoryCounts]{};
extern "C" void configure_layout(int words, int observe, int loop, int audit, int menu,
                                 int terminal) {
 t.layout=MemoryLayout(words,observe,loop,audit,menu,terminal,t.escape.offsets[1]-t.escape.offsets[0]);
}
extern "C" void reset() {
 t = Tables{}; state = State{};
 memset(mem,0,sizeof(mem)); memset(counts,0,sizeof(counts));
}
extern "C" void set_ptr(const char* name, void* p) {
"""
    wrapper += "\n".join(
        f'if (!strcmp(name,"{name}")) t.{name}=static_cast<{dtype}*>(p);'
        for name, dtype in pointers.items()
    )
    wrapper += '\n}\nextern "C" void set_number(const char* name, double value) {\n'
    wrapper += "\n".join(f'if (!strcmp(name,"{name}")) t.{name}=value;' for name in integers)
    wrapper += """
}
extern "C" void set_lcp(const int32_t* source, int n) {
 for(int k=0;k<n;++k) for(int j=0;j<n;++j)
  t.operator_lcp[k*kMaxOperators+j]=source[k*n+j];
}
extern "C" void step(int token) {
 advance_state(t,memory_at(mem,counts,true,0,t.layout),state,token);
}
extern "C" int phase() { return state.phase; }
extern "C" void last(int32_t* out) { memcpy(out,&state.last,sizeof(state.last)); }
extern "C" void allowed(uint32_t* out) {
 const Memory f=memory_at(mem,counts,false,0,t.layout);
 const uint32_t* base=nullptr;
 if (!state.audit_phase && !state.answer_pos) {
  if(state.phase==kFree) base=state.answer_complete?t.eos_mask:
      t.free_mask+static_cast<int64_t>(state.pos)*t.words;
  else if(state.phase==kEscaped) base=t.esc_mask+static_cast<int64_t>(state.pos)*t.words;
  else if(state.phase==kSpan && state.segment==kSubject && state.lo<state.hi &&
          state.depth==0 && !state.selecting && !derivation_bound(t,f)) base=t.root_mask;
 }
 for(int w=0;w<t.words;++w) out[w]=base?base[w]:0u;
 for(int i=0;i<32;++i) { threadIdx.x=i; write_mask(t,f,state,out); }
}
"""
    return compile_cpp(directory, header + definitions + helpers + mask + wrapper, shared=True)


class NativeCPU:
    """Sequential unobserved frontier spans with actual host tables and export.

    Successor/observation capabilities get layout tests, not CPU execution agreement."""

    def __init__(self, library, index, tokenizer, syntax, **options):
        supported = {
            "derivation",
            "paths",
            "committed_paths",
            "expressions",
            "escapes",
            "escape_bias",
            "start_rank",
            "relations",
            "versions",
        }
        if set(options) - supported or options.get("relations", "frontier") != "frontier":
            raise ValueError(
                "CPU helper supports unobserved frontier spans only; layout checks are separate"
            )
        self.lib = ct.CDLL(str(library))
        self.lib.set_ptr.argtypes = [ct.c_char_p, ct.c_void_p]
        self.lib.set_number.argtypes = [ct.c_char_p, ct.c_double]
        self.lib.allowed.argtypes = self.lib.last.argtypes = [ct.c_void_p]
        self.lib.set_lcp.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.reset()
        self.index, self.vocab = index, tokenizer.vocab_size
        self.words = mask_words(self.vocab)
        table_options = {
            k: v for k, v in options.items() if k in ("expressions", "escapes", "versions")
        }
        self.arrays = native_tables(index, tokenizer, syntax, **table_options)
        mapping = {
            "row_tokens": "entities.tokens",
            "row_offsets": "entities.offsets",
            "row_entities": "entities.owner",
            "relation_tokens": "relations.tokens",
            "relation_offsets": "relations.offsets",
            "relation_owners": "relations.owner",
            "adjacency_offsets": "adjacency",
            "version_offsets": "versions.offsets",
            "version_subjects": "versions.subjects",
            "version_adjacency": "versions.adjacency",
            "version_edge_relations": "versions.edge_relations",
            "version_edge_objects": "versions.edge_objects",
        }
        for prefix in (
            "closer",
            "separator",
            "escape",
            "operand",
            *(f"operator{k}" for k in range(4)),
        ):
            destination = (
                "operand_separator"
                if prefix == "operand"
                else f"operators[{prefix[-1]}]"
                if prefix.startswith("operator")
                else prefix
            )
            for suffix in ("offsets", "tokens", "next"):
                mapping[f"{prefix}_{suffix}"] = (
                    f"{destination}.{'next_pos' if suffix == 'next' else suffix}"
                )
            if f"{prefix}_offsets" in self.arrays:
                self.set(f"{destination}.length", len(self.arrays[f"{prefix}_offsets"]) - 1)
        for key, array in self.arrays.items():
            self.lib.set_ptr(
                mapping.get(key, key).encode(), array.ctypes.data if array.size else None
            )
        operators = sum(f"operator{k}_offsets" in self.arrays for k in range(4))
        self.lib.set_lcp(self.arrays["operator_lcp"].ctypes.data, operators)
        for key, value in {
            "vocab": self.vocab,
            "words": self.words,
            "entities.count": index.labels.num_rows,
            "relations.count": index.relations.num_rows,
            "wide_count": len(self.arrays["wide_keys"]),
            "repeats": True,
            "release_log_mass": float("-inf"),
            "bound_escapes": True,
            "unquoted_starts": bool(len(self.arrays["unquoted_rows"])),
            "operator_count": operators,
            "date_count": len(self.arrays["date_entities"]),
            "versions.count": len(self.arrays["version_offsets"]) - 1,
            "derivation": options.get("paths", False) or options.get("derivation", False),
            "sitelinks_rank": options.get("start_rank") == "sitelinks",
        }.items():
            self.set(key, value)
        for key, value in options.items():
            if isinstance(value, (bool, int, float)):
                self.set(key, value)

        self.lib.configure_layout.argtypes = [ct.c_int] * 6
        self.lib.configure_layout(
            0,
            options.get("observe", 0),
            options.get("observe_no_loop", False),
            options.get("audit", False),
            options.get("menu", 0),
            options.get("observe_terminal", 0),
        )

    def set(self, key, value):
        self.lib.set_number(key.encode(), float(value))

    def allowed(self):
        out = np.zeros(self.words, np.uint32)
        self.lib.allowed(out.ctypes.data)
        return frozenset(t for t in range(self.vocab) if int(out[t >> 5]) & (1 << (t & 31)))

    def advance(self, token):
        self.lib.step(int(token))
        assert self.lib.phase() != 3, f"native error after token {token}"

    def closed(self, index=None):
        last = np.zeros(7, np.int32)
        self.lib.last(last.ctypes.data)
        names = ("kind", "lo", "hi", "subj_lo", "subj_hi", "rel_lo", "rel_hi")
        return closed_span(
            self.index if index is None else index,
            {f"last_{name}": np.array([v]) for name, v in zip(names, last, strict=True)},
            0,
        )


def compile_lifecycle_helpers(directory):
    """Extract host allocation/reset/launch code and serialize its CUDA kernels.

    CUDA runtime calls become host copies. Block copies and mask initialization
    finish before virtual threads run; barriers/scheduling and DLPack are untested.
    Release reductions are deliberately unsupported.
    """
    compile_helpers(directory)
    code = (directory / "native.cpp").read_text()
    source = SOURCE.read_text()
    code = code.replace(
        "struct Dim { int x; }; static Dim blockDim{32}, threadIdx{0};",
        "struct dim3 { int x,y; dim3(int x=1,int y=1):x(x),y(y){} }; "
        "static dim3 blockDim{32},threadIdx{0},blockIdx{0};",
    )
    code += re.search(r"constexpr int32_t kReleaseThreads = .*?;", source)[0]
    code += r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#define __global__
#define __shared__
struct alignas(16) float4 { float x,y,z,w; };
float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
static std::vector<int32_t> dynamic_shared;
void sequential_mask(const Tables& tables,const Memory& f,const State& s,uint32_t* out) {
 const auto saved=threadIdx;
 const uint32_t* base=nullptr;
 if (!s.audit_phase && !s.answer_pos) {
  if(s.phase==kFree) base=s.answer_complete?tables.eos_mask:
      tables.free_mask+static_cast<int64_t>(s.pos)*tables.words;
  else if(s.phase==kEscaped) base=tables.esc_mask+static_cast<int64_t>(s.pos)*tables.words;
  else if(s.phase==kSpan && s.segment==kSubject && s.lo<s.hi && s.depth==0 &&
          !s.selecting && !derivation_bound(tables,f)) base=tables.root_mask;
 }
 for(int w=0;w<tables.words;++w) out[w]=base?base[w]:0u;
 for(int i=0;i<blockDim.x;++i) { threadIdx.x=i; write_mask(tables,f,s,out); }
 threadIdx=saved;
}
"""
    for name in (
        "advance_kernel",
        "begin_kernel",
        "set_version_kernel",
        "sync_kernel",
        "mask_kernel",
        "row_mask_kernel",
        "escape_bias_kernel",
        "apply_kernel",
    ):
        kernel = function(source, f"__global__ void {name}")
        if name == "row_mask_kernel":
            kernel = kernel.replace(
                "extern __shared__ int32_t scratch_rows[];",
                "assert(dynamic_shared.size()==static_cast<size_t>(t.layout.memory_ints)); "
                "int32_t* scratch_rows=dynamic_shared.data();",
            )
            kernel = kernel.replace("i = threadIdx.x", "i = 0").replace("i += blockDim.x", "++i")
        if name in ("mask_kernel", "row_mask_kernel"):
            kernel = kernel.replace("write_mask(", "sequential_mask(")
        if name == "row_mask_kernel":
            code += function(source, "__device__ bool continues_label")
        if name == "apply_kernel":
            code += function(source, "__device__ uint32_t blocked_bits")
            code += re.search(r"constexpr int32_t kApplyChunks = .*?;", source)[0]
        code += kernel

    # Keep launch geometry and host row-buffer growth verbatim, changing only
    # dispatch to a serial scheduler. Guards surround every host-backed allocation.
    code += r"""
template<class T> struct DeviceArray {
 size_t n=0; std::unique_ptr<unsigned char[]> storage;
 DeviceArray()=default;
 explicit DeviceArray(size_t n):n(n),storage(n?new unsigned char[n*sizeof(T)+32]:nullptr) {
  if(n) memset(storage.get(),0xa5,n*sizeof(T)+32);
 }
 T* get() const { return n?reinterpret_cast<T*>(storage.get()+16):nullptr; }
 size_t size() const { return n; }
 void guard() const {
  for(int i=0;n && i<16;++i) {
   assert(storage[i]==0xa5); assert(storage[16+n*sizeof(T)+i]==0xa5);
  }
 }
};
template<class T> struct Array {
 T* p; int64_t rows,cols=1,step=1;
 T* data() const { return p; }
 size_t shape(int d) const { return d?cols:rows; }
 int64_t stride(int d) const { return d?1:step; }
 size_t size() const { return rows*cols; }
 int device_type() const { return 2; }
};
using Slots=Array<const int32_t>; using Anywhere1D=Slots; using Anywhere2D=Slots;
template<class T> using HostArray=Array<const T>;
namespace nb { namespace device { struct cuda { static constexpr int value=2; }; } }
using cudaStream_t=uintptr_t;
cudaStream_t as_stream(uintptr_t s) { return s; }
constexpr int cudaMemcpyHostToDevice=0;
int cudaMemsetAsync(void* p,int value,size_t n,uintptr_t) { if(n) memset(p,value,n); return 0; }
int cudaMemcpyAsync(void* p,const void* q,size_t n,int,uintptr_t) {
 if(n) memcpy(p,q,n); return 0;
}
int cudaStreamSynchronize(uintptr_t) { return 0; }
int cudaGetLastError() { return 0; }
void check(int error,const char*) { assert(!error); }
template<class F> void launch(dim3 grid,int threads,size_t shared,bool serial_block,F run) {
 blockDim=dim3(threads); dynamic_shared.resize(shared/sizeof(int32_t));
 for(int y=0;y<grid.y;++y) for(int x=0;x<grid.x;++x) {
  blockIdx=dim3(x,y);
  for(int i=0;i<(serial_block?1:threads);++i) { threadIdx=dim3(i); run(); }
 }
}
// Release needs block-wide reductions; this seam never enables it.
template<class... A> void release_kernel(A...) { assert(false); }
struct HostEngine {
"""
    members = source[source.index("  const MemoryLayout layout_;") : source.index("\n};\n\n}  //")]
    code += members
    initializers = source[
        source.index("        states_(max_sequences),") : source.index(
            "    tables_.entities = Rows{"
        )
    ]
    initializers = initializers.replace('size(tables, "relation_owners")', "tables.relations.count")
    initializers = initializers.replace("* words", "* tables.words")
    code += "\nHostEngine(Tables tables,int max_sequences):layout_(tables.layout),\n"
    code += initializers
    code += "tables_=tables; reset(0); }\n"
    # Constructor capability names referenced by the real initializer list.
    code = code.replace(
        "struct HostEngine {",
        "struct HostEngine {\nstatic constexpr bool audit=false,menu=false,"
        "request_path_options=false;",
    )
    code += function(source, "  void check_sequences(")
    code += "\nint slot_count(const Slots& s) const { return s.rows; }\n"
    code += "template<class A> void on_device(const A&,const char*) {}\n"
    for signature in (
        "  void reset(",
        "  void begin(",
        "  void set_reached(",
        "  void set_version(",
        "  void sync(",
        "  void apply_mask(",
        "  void advance(",
        "  void fill_mask(",
    ):
        method = function(source, signature)
        method = re.sub(r"nb::ndarray<float,.*?> logits", "Array<float> logits", method)
        method = method.replace("nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig>", "Slots")

        def dispatch(match):
            name, grid, threads, shared, arguments = match.groups()
            serial = str(name in ("row_mask_kernel", "mask_kernel")).lower()
            return f"launch(dim3({grid}),{threads},{shared},{serial},[&]{{{name}({arguments});}});"

        method = re.sub(
            r"(\w+)<<<(dim3\([^)]*\)|[^,]+), ([^,]+), ([^,]+), as_stream\(stream\)>>>"
            r"\((.*?)\);",
            dispatch,
            method,
            flags=re.S,
        )
        assert "<<<" not in method, signature
        code += method
    allocations = re.findall(r"DeviceArray<[^;]+> (\w+);", members)
    code += "void guards() const {" + "".join(f"{n}.guard();" for n in allocations) + "}\n};\n"
    code += r"""
extern "C" HostEngine* host_new(int slots) { return new HostEngine(t,slots); }
extern "C" void host_delete(HostEngine* e) { e->guards(); delete e; }
extern "C" void host_reset(HostEngine* e) { e->reset(0); }
extern "C" void host_begin(HostEngine* e,const int32_t* p,int n,int on) {
 e->begin({p,n},on,0);
}
extern "C" void host_advance(HostEngine* e,const int32_t* p,int n) { e->advance({p,n},0); }
extern "C" void host_version(HostEngine* e,int slot,int version) { e->set_version(slot,version,0); }
extern "C" void host_anchor(HostEngine* e,int slot,const int32_t* p,int n) {
 e->set_reached(slot,{p,n},0);
}
extern "C" void host_sync(HostEngine* e,const int32_t* slots,int n,const int32_t* history,
 int width,const int32_t* prompt,const int32_t* total) {
 e->sync({slots,n},{history,e->max_sequences_,width,width},
         {prompt,e->max_sequences_},{total,e->max_sequences_},0);
}
extern "C" void host_masks(HostEngine* e,uint32_t* out) {
 e->fill_mask(e->max_sequences_,0);
 memcpy(out,e->mask_.get(),e->mask_.size()*sizeof(uint32_t)); e->guards();
}
extern "C" void host_states(HostEngine* e,int32_t* out) {
 memcpy(out,e->states_.get(),e->states_.size()*sizeof(State)); e->guards();
}
extern "C" void host_counts(HostEngine* e,int32_t* out) {
 memcpy(out,e->fact_counts_.get(),e->fact_counts_.size()*sizeof(int32_t));
}
extern "C" void host_apply(HostEngine* e,float* out,int rows,int columns,int stride,
 const int32_t* slots,const int32_t* local,const int32_t* inputs) {
 e->apply_mask({out,rows,columns,stride},{slots,rows},{local,local?rows:0},
               {inputs,inputs?rows:0},0); e->guards();
}
"""
    return compile_cpp(directory, code, shared=True)


class SequentialEngine(Engine):
    """Existing Engine API over extracted host methods and serialized kernels."""

    def __init__(self, library, index, tokenizer, syntax, max_sequences, **options):
        from types import SimpleNamespace

        self.tables = NativeCPU(library, index, tokenizer, syntax, **options)
        self.lib = self.tables.lib
        signatures = {
            "new": [ct.c_int],
            "delete": [],
            "reset": [],
            "begin": [ct.c_void_p, ct.c_int, ct.c_int],
            "advance": [ct.c_void_p, ct.c_int],
            "version": [ct.c_int, ct.c_int],
            "anchor": [ct.c_int, ct.c_void_p, ct.c_int],
            "sync": [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p],
            "masks": [ct.c_void_p],
            "states": [ct.c_void_p],
            "counts": [ct.c_void_p],
            "apply": [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int] + [ct.c_void_p] * 3,
        }
        for name, args in signatures.items():
            getattr(self.lib, f"host_{name}").argtypes = (
                args if name == "new" else [ct.c_void_p, *args]
            )
        self.lib.host_new.restype = ct.c_void_p
        self.handle = self.lib.host_new(max_sequences)
        self._count_slots = max_sequences
        self._index, self._versions = index, options.get("versions")
        self._entity_index = None
        self._native = SimpleNamespace(
            set_reached=lambda slot, rows, stream: self.lib.host_anchor(
                self.handle, slot, rows.ctypes.data, len(rows)
            ),
            set_version=lambda slot, version, stream: self.lib.host_version(
                self.handle, slot, version
            ),
        )
        source = SOURCE.read_text().split('m.attr("STATE_FIELDS") = nb::make_tuple(', 1)[1]
        self.fields = re.findall(r'"(\w+)"', source.split(");", 1)[0])

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.host_delete(self.handle)

    @property
    def max_sequences(self):
        return self._count_slots

    def reset(self, stream=0):
        self.lib.host_reset(self.handle)

    def begin(self, slots, stream=0):
        slots = np.asarray(slots, dtype=np.int32)
        self.lib.host_begin(self.handle, slots.ctypes.data, len(slots), True)

    def end(self, slots, stream=0):
        slots = np.asarray(slots, dtype=np.int32)
        self.lib.host_begin(self.handle, slots.ctypes.data, len(slots), False)

    def advance(self, tokens, stream=0):
        tokens = np.asarray(tokens, dtype=np.int32)
        self.lib.host_advance(self.handle, tokens.ctypes.data, len(tokens))

    def sync(self, slots, history, prompt_len, total_len, stream=0):
        slots, history, prompt_len, total_len = (
            np.asarray(v, dtype=np.int32) for v in (slots, history, prompt_len, total_len)
        )
        self.lib.host_sync(
            self.handle,
            slots.ctypes.data,
            len(slots),
            history.ctypes.data,
            history.shape[1],
            prompt_len.ctypes.data,
            total_len.ctypes.data,
        )

    def fill_mask(self, sequences=None, stream=0):
        self._masks = np.empty((self.max_sequences, self.tables.words), np.uint32)
        self.lib.host_masks(self.handle, self._masks.ctypes.data)

    def mask_host(self, sequences=None):
        return self._masks[:sequences]

    def states_host(self, sequences=None):
        states = np.empty((self.max_sequences, len(self.fields)), np.int32)
        self.lib.host_states(self.handle, states.ctypes.data)
        # Only unobserved frontier capabilities are supported here.
        fields = self.fields[: self.fields.index("observe_pos")]
        return {name: states[:sequences, i] for i, name in enumerate(fields)}

    def counts_host(self):
        counts = np.empty((self.max_sequences, 7), np.int32)
        self.lib.host_counts(self.handle, counts.ctypes.data)
        return counts

    def apply_mask(self, logits, row_slots, stream=0, *, local_pos=None, input_ids=None):
        logits = np.asarray(logits)
        assert logits.dtype == np.float32 and logits.strides[1] == 4
        slots = np.asarray(row_slots, dtype=np.int32)
        drafts = [None if v is None else np.asarray(v, np.int32) for v in (local_pos, input_ids)]
        self.lib.host_apply(
            self.handle,
            logits.ctypes.data,
            *logits.shape,
            logits.strides[0] // 4,
            slots.ctypes.data,
            *(None if v is None else v.ctypes.data for v in drafts),
        )
