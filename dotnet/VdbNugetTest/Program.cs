// Consumes the Vdb NuGet package exactly as an end user would:
// PackageReference + the managed API. The native vdb_c.dll arrives inside
// the package (runtimes/win-x64/native) — nothing is copied by hand.
using Vdb;

const uint dim = 48;
const int count = 800;

var rng = new Random(7);
var vectors = new float[count][];
for (var i = 0; i < count; i++)
{
    vectors[i] = new float[dim];
    for (var d = 0; d < dim; d++)
        vectors[i][d] = (float)(rng.NextDouble() * 2.0 - 1.0);
}

using var builder = new VdbIndexBuilder(dim, VdbMetric.Cosine);
for (var i = 0; i < count; i++)
    builder.Add((ulong)(500 + i), vectors[i]);

using var memIndex = builder.Freeze();
Console.WriteLine($"index built from NuGet API: {memIndex.Count} vectors");

var hits = memIndex.Search(vectors[123], k: 5);
if (hits[0].ExternalId != 623 || Math.Abs(hits[0].Score - 1.0f) > 1e-4f)
{
    Console.WriteLine($"FAIL: self-search -> {hits[0]}");
    return 1;
}
Console.WriteLine($"self-search: {hits[0]}");

var path = Path.Combine(Path.GetTempPath(), "vdb_nuget_test.vdb");
memIndex.WriteSegment(path);
using var diskIndex = VdbIndex.Open(path);

var mismatches = 0;
for (var qi = 0; qi < 200; qi++)
{
    var q = new float[dim];
    for (var d = 0; d < dim; d++) q[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
    var a = memIndex.Search(q, 10);
    var b = diskIndex.Search(q, 10);
    if (a.Length != b.Length) { mismatches++; continue; }
    for (var j = 0; j < a.Length; j++)
        if (a[j] != b[j]) { mismatches++; break; }
}
File.Delete(path);

Console.WriteLine(mismatches == 0
    ? "memory vs segment: 200/200 identical through the package API"
    : $"FAIL: {mismatches} mismatches");

// Error surface: exceptions with native context, not crashes.
try
{
    VdbIndex.Open("no_such_segment.vdb");
    Console.WriteLine("FAIL: open of missing file did not throw");
    return 1;
}
catch (VdbException e)
{
    Console.WriteLine($"expected failure surfaced cleanly: {e.Message}");
}

if (mismatches != 0) return 1;
Console.WriteLine("NUGET PACKAGE TEST PASSED");
return 0;
