# vdb

An **embedded vector database** written in C++20, designed for .NET consumption.
Think *SQLite, but for similarity search*: no server, no service, no network —
a small native engine that lives inside your process and searches millions of
vectors in microseconds.

> **Status: alpha.** The core engine (HNSW index, durable segment format,
> C ABI, .NET package) is implemented and tested on Windows and Linux.
> WAL/memtable, deletes, filters, compaction and quantization are on the
> roadmap. Not production-ready yet.

---

## What is a vector database? (start here if this is new to you)

Modern ML models (OpenAI, llama.cpp, sentence-transformers...) can turn a piece
of text, an image or a product into an **embedding**: a list of floats — say
768 of them — where *similar things end up close together geometrically*.
"Invoice for cloud services" and "AWS billing receipt" share almost no words,
but their embeddings are near neighbors.

A vector database stores those embeddings and answers one question fast:

> *"Here is a query vector — which of my million stored vectors are closest
> to it?"*

That question powers semantic search, RAG (retrieval-augmented generation),
recommendations, deduplication and anomaly detection. Answering it by brute
force means comparing against every stored vector — fine for 10,000 items,
painful for 10 million. vdb answers it approximately in ~100 microseconds
using an **HNSW graph index** (see [Concepts](#concepts) below).

**Embedded** means there is nothing to deploy: like SQLite, the engine is a
library inside your app. Your vectors never leave your process or your disk.

---

## Quick start (.NET)

```bash
dotnet new console
dotnet add package Vdb        # from your feed; nuget.org publication pending
```

```csharp
using Vdb;

// 1. Build an index. Dim = your embedding size (384, 768, 1536...).
//    Cosine is the right metric for text embeddings.
using var builder = new VdbIndexBuilder(dim: 768, VdbMetric.Cosine);

foreach (var (id, embedding) in myDocuments)   // your own ids (u64) + float[]
    builder.Add(id, embedding);

using var index = builder.Freeze();            // builder is consumed here

// 2. Search: top-10 most similar to a query embedding.
SearchHit[] hits = index.Search(queryEmbedding, k: 10);
foreach (var hit in hits)
    Console.WriteLine($"doc {hit.ExternalId}: similarity {hit.Score:F3}");

// 3. Persist as an immutable segment file, atomically.
index.WriteSegment("docs.vdb");

// 4. Later (or in another process): open via memory-mapping. Zero copies —
//    searches run directly over the file. Opening 100k vectors is instant.
using var segment = VdbIndex.Open("docs.vdb");
var same = segment.Search(queryEmbedding, k: 10);   // bit-identical results
```

Everything the engine can throw surfaces as a `VdbException` with an error
code and message — never a native crash.

## From documents to search results (the RAG recipe)

vdb stores and searches **vectors** — it does not read PDFs or call embedding
models for you. This section shows the missing glue, end to end, because it
is where most people get stuck.

The pipeline is always the same four steps:

```
your documents → 1. chunk → 2. embed → 3. index (vdb) → 4. search & map back
```

### 1. Chunk your documents

Embedding a whole document dilutes its meaning into one average-of-everything
vector. Split it into passages first — a few hundred words with some overlap
is a solid default:

```csharp
static IEnumerable<string> Chunk(string text, int maxWords = 250, int overlap = 50)
{
    var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    for (var start = 0; start < words.Length; start += maxWords - overlap)
    {
        yield return string.Join(' ',
            words.Skip(start).Take(maxWords));
        if (start + maxWords >= words.Length) yield break;
    }
}
```

### 2. Embed each chunk

Any embedding model works. This example uses [Ollama](https://ollama.com)
running locally (`ollama pull nomic-embed-text`) — free, private, no API key.
The same code shape applies to OpenAI/Azure/Cohere: POST text, get floats.

```csharp
static async Task<float[][]> EmbedAsync(HttpClient http, IReadOnlyList<string> texts)
{
    var resp = await http.PostAsJsonAsync("http://localhost:11434/api/embed",
        new { model = "nomic-embed-text", input = texts });
    var body = await resp.Content.ReadFromJsonAsync<EmbedResponse>();
    return body!.embeddings;
}
record EmbedResponse(float[][] embeddings);
```

### 3. Index the chunks

vdb only keeps `(external_id, vector)`. **You own the mapping from id back to
text** — a `Dictionary`, a SQLite table, a JSON file, whatever your app
already has. (The segment format supports typed metadata columns in the C++
core; exposing them through the .NET binding is on the roadmap.)

```csharp
var chunks = new Dictionary<ulong, (string DocName, string Text)>();
using var builder = new VdbIndexBuilder(dim: 768, VdbMetric.Cosine); // nomic = 768

ulong nextId = 0;
foreach (var doc in myDocuments)
{
    var pieces = Chunk(doc.Text).ToList();
    var vectors = await EmbedAsync(http, pieces);
    for (var i = 0; i < pieces.Count; i++)
    {
        chunks[nextId] = (doc.Name, pieces[i]);
        builder.Add(nextId, vectors[i]);
        nextId++;
    }
}
using var index = builder.Freeze();
index.WriteSegment("kb.vdb");   // and persist your chunks dictionary alongside
```

### 4. Search and map back to text

```csharp
var queryVec = (await EmbedAsync(http, new[] { userQuestion }))[0];
var hits = index.Search(queryVec, k: 5);

foreach (var hit in hits)
{
    var (docName, text) = chunks[hit.ExternalId];
    Console.WriteLine($"[{hit.Score:F3}] {docName}: {text[..Math.Min(120, text.Length)]}...");
}
// For RAG: concatenate the top chunks into your LLM prompt as context,
// with an instruction like "answer ONLY from the passages below".
```

> **Runnable version:** this whole recipe, including the final LLM answer via
> a local Ollama model, lives in [`dotnet/samples/RagDemo`](dotnet/samples/RagDemo).
> With Ollama running (`ollama pull nomic-embed-text qwen3-coder:30b`):
> `cd dotnet/samples/RagDemo && dotnet run` — optionally
> `dotnet run -- "your question" your-chat-model`.

### The gotchas that bite everyone

- **Same model on both sides.** Documents and queries must be embedded with
  the *same* model. Mixing models produces garbage results with no error —
  the vectors simply live in different spaces.
- **`dim` is fixed by the model.** nomic-embed-text = 768, MiniLM = 384,
  OpenAI text-embedding-3-small = 1536. The builder's `dim` must match, and
  changing models means rebuilding the index.
- **Cosine is the metric for text embeddings.** Unless your model's docs say
  otherwise, don't overthink it.
- **No deletes yet.** Segments are immutable snapshots: when your corpus
  changes, rebuild and atomically replace the segment (that's what the
  tmp+rename write is for). Incremental updates arrive with the WAL/memtable
  milestone.
- **Ids are yours.** `external_id` is an opaque `u64` to vdb — encode
  whatever you want in it (row id, doc id × 1000 + chunk index, hash...).

### What the package contains

```
Vdb.nupkg
├── lib/net10.0/Vdb.dll                  ← managed wrapper (SafeHandle, Span)
└── runtimes/win-x64/native/vdb_c.dll    ← the C++ engine (linux/macOS: planned via CI)
```

The .NET runtime picks the right native binary automatically; `DllImport`
plumbing, handle lifetimes and pinning are already handled by the wrapper.

---

## Quick start (C++)

```cpp
#include "vdb/builder.hpp"
#include "vdb/search.hpp"
#include "vdb/segment.hpp"

auto builder = std::move(
    vdb::IndexBuilder::Create(768, vdb::Metric::kCosine).value());
for (auto& [id, vec] : docs)
    builder.Add(id, vec);                       // Result<> — no exceptions

auto index = std::move(std::move(builder).Freeze().value());
auto hits  = vdb::Search(index.View(), query, /*k=*/10).value();

vdb::WriteSegment("docs.vdb", index.View());    // atomic: tmp + fsync + rename
auto seg = std::move(vdb::Segment::Open("docs.vdb").value());
auto same = vdb::Search(seg.View(), query, 10).value();  // same code path, mmap'd
```

Every fallible call returns `vdb::Result<T>` (`.ok()` / `.value()` /
`.error()`). The library throws no exceptions across its public surface —
that is what makes the C ABI and the .NET binding safe.

---

## Concepts

### HNSW: how search stays fast

Comparing your query against every vector is O(n). HNSW (*Hierarchical
Navigable Small World*) instead builds a **graph**: every vector becomes a
node connected to a few well-chosen neighbors, organized in layers like a
skip list — the top layers are sparse "highways" for coarse navigation, the
bottom layer contains everything for fine-grained search. A query greedily
descends: a few long hops to land in the right region, then a careful local
search. Cost drops from O(n) to roughly O(log n) graph hops.

The price is **approximation**: HNSW can miss a true neighbor. Quality is
measured as *recall@k* (what fraction of the true top-k it finds) and tuned
at query time with a single knob (`ef`, below). In our benchmarks vdb reaches
recall 0.997–1.0 while being 55–124x faster than exact search.

### Parameters you actually need to know

| Parameter | What it does | Practical guidance |
|---|---|---|
| `dim` | Embedding size | Fixed by your embedding model (384, 768, 1536...) |
| `metric` | Distance definition | **Cosine** for text embeddings (the usual choice). **L2** for raw feature vectors. **Dot** for recommendation-style scores (caveat: weaker guarantees). |
| `M` | Graph degree (build-time) | Default 16 is a good start; 24–32 buys recall for large/hard datasets at more memory. Cannot change without rebuilding. |
| `ef_construction` | Build-time effort | Default 200. Higher = better graph, slower build. Rarely needs touching. |
| `ef` (search) | **The runtime knob** | Higher = better recall, ~linearly slower. 50–100 covers most uses; sweep it and measure recall on *your* data. Must be ≥ k. |

Score semantics: cosine and dot return *similarity* (higher = closer); L2
returns *squared distance* (lower = closer). Results always come best-first.

### Segments: durability without a server

A segment (`.vdb` file) is an **immutable snapshot** of an index:

- Written atomically (`.tmp` + fsync + rename — a crash leaves either the old
  file or the complete new one, never a torn write).
- Opened with **memory-mapping**: the OS pages data in on demand, so opening
  is instant regardless of size, and searches read the file directly with
  zero deserialization. The in-memory layout **is** the on-disk format.
- Self-validating: a 128-byte header + per-section CRC32C checksums + a
  footer. Flip any byte and `Open` rejects the file; truncated or
  half-written files are rejected too.
- Typed metadata columns (int64 / bool / string) travel inside the segment,
  also read zero-copy.

Immutable segments are the foundation of the LSM-style architecture this
project is heading toward (memtable + WAL + background compaction).

---

## Performance

Measured on a Ryzen 9 9900X3D, single thread, clustered corpus (the regime
real embeddings live in), n = 100,000 vectors, dim = 768, cosine:

| ef | recall@10 | latency | vs exact scan |
|----|-----------|---------|---------------|
| 10  | 0.794 | 52 µs  | 124x |
| 50  | 0.997 | 94 µs  | 67x  |
| 100 | 1.000 | 117 µs | 55x  |

Exact brute-force baseline: 6,373 µs/query. Build: ~2,900 inserts/s
(single-threaded); freezing to the immutable layout is sub-millisecond.
From C# through the NuGet package (P/Invoke included): ~29,000 queries/s.

SIMD: AVX2+FMA kernels selected at runtime via CPUID (scalar fallback on CPUs
without AVX2; NEON for arm64 planned). Vectors are stored padded to 64-byte
alignment so kernels use aligned loads with no scalar tails.

---

## Building from source

Requires CMake ≥ 3.20 and a C++20 compiler. No external dependencies — the
test framework, CRC32C and SIMD kernels are all in-tree.

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Verified on Linux (GCC 13) and Windows (MSVC, Visual Studio 2026). macOS
arm64 compiles the scalar kernel path (untested until CI lands).

.NET package (Windows, after the MSVC build):

```bash
cd dotnet/Vdb && dotnet pack -c Release -o ../packages
cd ../VdbNugetTest && dotnet run -c Release    # consumer test via local feed
```

## Project layout

```
include/vdb/        public API: builder, search, segment, meta, Result, C ABI
src/kernels/        L2/dot distance kernels (scalar + AVX2) + CPU dispatch
src/hnsw/           graph construction, neighbor-selection heuristic, search core
src/segment/        .vdb format: writer, mmap reader, CRC32C
src/io/             platform layer: atomic write, memory mapping
src/capi/           C ABI (the P/Invoke surface)
dotnet/Vdb/         managed wrapper + NuGet packaging
tests/  bench/      zero-dependency test suites and benchmarks
```

## Design principles

- **One search path.** In-memory index and mmap'd segment expose the same
  non-owning `IndexView`; the search code cannot tell them apart. Round-trip
  tests require bit-identical results, and get them.
- **No exceptions on the surface.** `Result<T>` everywhere; the C ABI is a
  mechanical translation, and .NET sees clean typed exceptions.
- **Layout is the format.** Contiguous, padded, aligned buffers — the same
  bytes serve SIMD kernels in RAM and zero-copy readers from disk.
- **Corruption is detected, not discovered.** CRCs + structural validation at
  open; a segment file is treated as untrusted input.

## Roadmap

- [ ] CI matrix (Linux / Windows / macOS arm64) + packaged native binaries
- [ ] WAL + memtable (live inserts on top of immutable segments)
- [ ] Deletes (tombstones), metadata filters, background compaction
- [ ] int8 quantization (header field already reserved)
- [ ] NuGet publication with all runtimes

## License

TBD.
