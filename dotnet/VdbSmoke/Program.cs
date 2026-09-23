// VdbSmoke: end-to-end smoke test of the vdb C ABI from .NET.
// Build index -> search -> write segment -> open (mmap) -> verify searches
// over the segment are bit-identical to the in-memory index.
using System.Diagnostics;
using System.Runtime.InteropServices;

internal static class Native
{
    private const string Lib = "vdb_c";

    [DllImport(Lib)] public static extern IntPtr vdb_last_error();

    [DllImport(Lib)]
    public static extern int vdb_builder_create(uint dim, byte metric, uint m,
        uint efConstruction, ulong seed, out IntPtr builder);

    [DllImport(Lib)]
    public static extern int vdb_builder_add(IntPtr builder, ulong externalId,
        float[] vec, uint len);

    [DllImport(Lib)]
    public static extern int vdb_builder_freeze(IntPtr builder, out IntPtr index);

    [DllImport(Lib)] public static extern void vdb_builder_destroy(IntPtr builder);

    [DllImport(Lib)]
    public static extern int vdb_index_write(IntPtr index,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib)]
    public static extern int vdb_index_open(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int validateCrcs,
        out IntPtr index);

    [DllImport(Lib)] public static extern uint vdb_index_count(IntPtr index);

    [DllImport(Lib)]
    public static extern int vdb_index_search(IntPtr index, float[] query, uint len,
        uint k, uint ef, ulong[] outIds, float[] outScores, out uint found);

    [DllImport(Lib)] public static extern void vdb_index_destroy(IntPtr index);

    public static string LastError()
        => Marshal.PtrToStringUTF8(vdb_last_error()) ?? "";

    public static void Check(int code, string what)
    {
        if (code != 0)
            throw new InvalidOperationException($"{what} failed ({code}): {LastError()}");
    }
}

internal static class Program
{
    private const uint Dim = 64;
    private const int Count = 2000;
    private const uint K = 10;
    private const uint Ef = 100;

    private static int Main()
    {
        var rng = new Random(42);
        var vectors = new float[Count][];
        for (var i = 0; i < Count; i++)
        {
            vectors[i] = new float[Dim];
            for (var d = 0; d < Dim; d++)
                vectors[i][d] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // ---- build in memory ------------------------------------------------
        Native.Check(Native.vdb_builder_create(Dim, metric: 1 /* cosine */, m: 16,
            efConstruction: 200, seed: 42, out var builder), "builder_create");
        var sw = Stopwatch.StartNew();
        for (var i = 0; i < Count; i++)
            Native.Check(Native.vdb_builder_add(builder, (ulong)(10_000 + i),
                vectors[i], Dim), "builder_add");
        Native.Check(Native.vdb_builder_freeze(builder, out var memIndex),
            "builder_freeze");
        Console.WriteLine(
            $"built {Native.vdb_index_count(memIndex)} vectors in {sw.ElapsedMilliseconds} ms");

        // ---- sanity: self-search --------------------------------------------
        var ids = new ulong[K];
        var scores = new float[K];
        Native.Check(Native.vdb_index_search(memIndex, vectors[42], Dim, K, Ef,
            ids, scores, out var found), "search(mem)");
        if (found != K || ids[0] != 10_042 || Math.Abs(scores[0] - 1.0f) > 1e-4f)
        {
            Console.WriteLine(
                $"FAIL: self-search returned id={ids[0]} score={scores[0]} found={found}");
            return 1;
        }
        Console.WriteLine($"self-search ok: id={ids[0]} score={scores[0]:F6}");

        // ---- segment roundtrip -----------------------------------------------
        var path = Path.Combine(Path.GetTempPath(), "vdb_smoke_seg.vdb");
        Native.Check(Native.vdb_index_write(memIndex, path), "index_write");
        Native.Check(Native.vdb_index_open(path, validateCrcs: 1, out var diskIndex),
            "index_open");
        Console.WriteLine(
            $"segment written and opened: {new FileInfo(path).Length / 1024.0 / 1024.0:F1} MB, count={Native.vdb_index_count(diskIndex)}");

        var mismatches = 0;
        var idsB = new ulong[K];
        var scoresB = new float[K];
        for (var qi = 0; qi < 500; qi++)
        {
            var q = new float[Dim];
            for (var d = 0; d < Dim; d++) q[d] = (float)(rng.NextDouble() * 2.0 - 1.0);

            Native.Check(Native.vdb_index_search(memIndex, q, Dim, K, Ef, ids, scores,
                out var fa), "search(mem)");
            Native.Check(Native.vdb_index_search(diskIndex, q, Dim, K, Ef, idsB,
                scoresB, out var fb), "search(disk)");
            if (fa != fb) { mismatches++; continue; }
            for (var j = 0; j < fa; j++)
            {
                // Bit-identical requirement: same ids AND same float scores.
                if (ids[j] != idsB[j] || !scores[j].Equals(scoresB[j]))
                {
                    mismatches++;
                    break;
                }
            }
        }
        Console.WriteLine(mismatches == 0
            ? "memory vs mmap segment: 500/500 queries bit-identical"
            : $"FAIL: {mismatches} mismatched queries");

        // ---- throughput from managed code -------------------------------------
        var queries = new float[200][];
        for (var i = 0; i < queries.Length; i++)
        {
            queries[i] = new float[Dim];
            for (var d = 0; d < Dim; d++)
                queries[i][d] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        sw.Restart();
        const int reps = 10;
        for (var r = 0; r < reps; r++)
            foreach (var q in queries)
                Native.vdb_index_search(diskIndex, q, Dim, K, Ef, ids, scores, out _);
        var totalQueries = reps * queries.Length;
        Console.WriteLine(
            $"search over mmap segment from C#: {totalQueries / sw.Elapsed.TotalSeconds:F0} queries/s (k={K}, ef={Ef})");

        Native.vdb_index_destroy(memIndex);
        Native.vdb_index_destroy(diskIndex);
        File.Delete(path);

        if (mismatches != 0) return 1;
        Console.WriteLine("SMOKE TEST PASSED");
        return 0;
    }
}
