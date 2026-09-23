// End-to-end RAG over vdb + Ollama, fully local:
//   documents -> chunk -> embed (nomic-embed-text) -> vdb index
//   question  -> embed -> top-k retrieval -> grounded prompt -> LLM answer
//
// Prerequisites: Ollama running locally with:
//   ollama pull nomic-embed-text
//   ollama pull qwen3-coder:30b   (or pass another chat model as 2nd arg)
//
// Usage: dotnet run [-- "your question" [chat-model]]
using System.Net.Http.Json;
using Vdb;

const string EmbedModel = "nomic-embed-text";
const uint Dim = 768;  // nomic-embed-text output size
const uint TopK = 3;

var question = args.Length > 0
    ? args[0]
    : "How does the Kalman filter estimate the hedge ratio in pairs trading?";
var chatModel = args.Length > 1 ? args[1] : "qwen3-coder:30b";

var docs = new (string Name, string Text)[]
{
    ("pairs_trading.md",
     "Pairs trading is a market-neutral strategy that exploits relative " +
     "mispricing between two cointegrated assets. The spread between the two " +
     "legs is modeled as a mean-reverting process. A Kalman filter estimates " +
     "the hedge ratio dynamically, treating it as a hidden state that evolves " +
     "over time; each new price observation updates the state estimate and its " +
     "uncertainty. Entries trigger when the z-score of the spread exceeds two " +
     "standard deviations, and positions close when the spread reverts to its " +
     "mean. Transaction costs are the main constraint on live profitability."),
    ("options_pricing.md",
     "The Black-Scholes model prices European options assuming geometric " +
     "Brownian motion and constant volatility. The Greeks measure the " +
     "sensitivities of the option price: delta to the underlying, gamma to " +
     "delta itself, theta to time decay, vega to volatility and rho to " +
     "interest rates. Implied volatility is the volatility that makes the " +
     "model price match the market price; its surface across strikes and " +
     "maturities reveals where the model's assumptions break down."),
    ("hnsw_notes.md",
     "HNSW builds a multilayer navigable small-world graph for approximate " +
     "nearest neighbor search. Upper layers act as highways for greedy " +
     "routing while layer zero holds every vector. The neighbor selection " +
     "heuristic keeps edges directionally diverse, preserving bridges between " +
     "clusters. Recall is tuned at query time with the ef parameter, without " +
     "rebuilding the graph."),
    ("swimming_notes.md",
     "Freestyle swimming efficiency is measured with SWOLF, the sum of " +
     "strokes and seconds per pool length. A long glide phase after each " +
     "stroke reduces stroke count, and bilateral breathing every three " +
     "strokes balances the body roll. Kick from the hips with relaxed ankles " +
     "rather than from the knees."),
};

using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

// ---- 1. chunk + embed + index --------------------------------------------
var chunks = new Dictionary<ulong, (string Doc, string Text)>();
using var builder = new VdbIndexBuilder(Dim, VdbMetric.Cosine);

ulong id = 0;
foreach (var (name, text) in docs)
{
    var pieces = Chunk(text).ToList();
    var vectors = await EmbedAsync(pieces);
    for (var i = 0; i < pieces.Count; i++)
    {
        chunks[id] = (name, pieces[i]);
        builder.Add(id, vectors[i]);
        id++;
    }
    Console.WriteLine($"ingested {name,-22} -> {pieces.Count} chunk(s)");
}
using var index = builder.Freeze();
Console.WriteLine($"vdb index ready: {index.Count} chunks (dim {Dim})\n");

// ---- 2. retrieve -----------------------------------------------------------
Console.WriteLine($"Q: {question}\n");
var qvec = (await EmbedAsync(new[] { question }))[0];
var hits = index.Search(qvec, TopK);

Console.WriteLine("-- retrieved --");
foreach (var h in hits)
    Console.WriteLine($"[{h.Score:F3}] {chunks[h.ExternalId].Doc}");

// ---- 3. grounded prompt + generation ----------------------------------------
var context = string.Join("\n\n", hits.Select((h, i) =>
    $"[{i + 1}] (source: {chunks[h.ExternalId].Doc})\n{chunks[h.ExternalId].Text}"));
var prompt =
    "Answer the question using ONLY the context passages below. " +
    "If the answer is not in the context, say so plainly. Cite passages by [number].\n\n" +
    $"Context:\n{context}\n\nQuestion: {question}";

Console.WriteLine($"\n-- asking {chatModel} --");
var resp = await http.PostAsJsonAsync("http://localhost:11434/api/chat", new
{
    model = chatModel,
    messages = new[] { new { role = "user", content = prompt } },
    stream = false,
});
resp.EnsureSuccessStatusCode();
var chat = await resp.Content.ReadFromJsonAsync<ChatResponse>();

Console.WriteLine($"\n-- answer --\n{chat!.message.content.Trim()}");
return 0;

// ---- helpers ---------------------------------------------------------------
static IEnumerable<string> Chunk(string text, int maxWords = 80, int overlap = 20)
{
    var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    for (var start = 0; ; start += maxWords - overlap)
    {
        yield return string.Join(' ', words.Skip(start).Take(maxWords));
        if (start + maxWords >= words.Length) yield break;
    }
}

async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts)
{
    var r = await http.PostAsJsonAsync("http://localhost:11434/api/embed",
        new { model = EmbedModel, input = texts });
    r.EnsureSuccessStatusCode();
    return (await r.Content.ReadFromJsonAsync<EmbedResponse>())!.embeddings;
}

record EmbedResponse(float[][] embeddings);
record ChatMessage(string role, string content);
record ChatResponse(ChatMessage message);
