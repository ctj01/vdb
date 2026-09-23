// Managed wrapper over the vdb C ABI.
//
// Design notes:
//  - SafeHandle for every native handle: deterministic release even under
//    finalization, immune to double-free (the C side's *_destroy is
//    NULL-safe, and Freeze marks the builder handle invalid because the
//    native call consumes it).
//  - ReadOnlySpan<float> + fixed: zero-copy across the P/Invoke boundary.
//  - Native error codes become VdbException with the thread-local message.
using System.Runtime.InteropServices;

namespace Vdb;

public enum VdbMetric : byte
{
    L2 = 0,
    Cosine = 1,
    Dot = 2,
}

public sealed class VdbException(int code, string message)
    : Exception($"vdb error {code}: {message}")
{
    public int Code { get; } = code;
}

internal sealed class BuilderHandle : SafeHandle
{
    public BuilderHandle() : base(IntPtr.Zero, ownsHandle: true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle()
    {
        NativeMethods.vdb_builder_destroy(handle);
        return true;
    }
}

internal sealed class IndexHandle : SafeHandle
{
    public IndexHandle() : base(IntPtr.Zero, ownsHandle: true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle()
    {
        NativeMethods.vdb_index_destroy(handle);
        return true;
    }
}

internal static unsafe partial class NativeMethods
{
    private const string Lib = "vdb_c";

    [DllImport(Lib)] internal static extern IntPtr vdb_last_error();

    [DllImport(Lib)]
    internal static extern int vdb_builder_create(uint dim, byte metric, uint m,
        uint efConstruction, ulong seed, out BuilderHandle builder);

    [DllImport(Lib)]
    internal static extern int vdb_builder_add(BuilderHandle builder,
        ulong externalId, float* vec, uint len);

    [DllImport(Lib)]
    internal static extern int vdb_builder_freeze(BuilderHandle builder,
        out IndexHandle index);

    [DllImport(Lib)] internal static extern void vdb_builder_destroy(IntPtr builder);

    [DllImport(Lib)]
    internal static extern int vdb_index_write(IndexHandle index,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib)]
    internal static extern int vdb_index_open(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, int validateCrcs,
        out IndexHandle index);

    [DllImport(Lib)] internal static extern uint vdb_index_count(IndexHandle index);

    [DllImport(Lib)]
    internal static extern int vdb_index_search(IndexHandle index, float* query,
        uint len, uint k, uint ef, ulong* outIds, float* outScores, out uint found);

    [DllImport(Lib)] internal static extern void vdb_index_destroy(IntPtr index);

    internal static void Check(int code)
    {
        if (code == 0) return;
        var msg = Marshal.PtrToStringUTF8(vdb_last_error()) ?? "(no message)";
        throw new VdbException(code, msg);
    }
}

public readonly record struct SearchHit(ulong ExternalId, float Score);

/// <summary>Builds an HNSW index in memory. Freeze() consumes the builder.</summary>
public sealed class VdbIndexBuilder : IDisposable
{
    private readonly BuilderHandle _handle;
    public uint Dim { get; }

    public VdbIndexBuilder(uint dim, VdbMetric metric, uint m = 16,
                           uint efConstruction = 200, ulong seed = 42)
    {
        Dim = dim;
        NativeMethods.Check(NativeMethods.vdb_builder_create(dim, (byte)metric, m,
            efConstruction, seed, out _handle));
    }

    public unsafe void Add(ulong externalId, ReadOnlySpan<float> vector)
    {
        fixed (float* p = vector)
        {
            NativeMethods.Check(NativeMethods.vdb_builder_add(_handle, externalId, p,
                (uint)vector.Length));
        }
    }

    /// <summary>Freezes into a searchable index. The builder becomes unusable.</summary>
    public VdbIndex Freeze()
    {
        NativeMethods.Check(NativeMethods.vdb_builder_freeze(_handle, out var index));
        // The native call freed the builder on success; make sure our
        // SafeHandle never calls destroy on the dangling pointer.
        _handle.SetHandleAsInvalid();
        return new VdbIndex(index);
    }

    public void Dispose() => _handle.Dispose();
}

/// <summary>A searchable index: frozen in memory or opened from a segment.</summary>
public sealed class VdbIndex : IDisposable
{
    private readonly IndexHandle _handle;
    internal VdbIndex(IndexHandle handle) => _handle = handle;

    public uint Count => NativeMethods.vdb_index_count(_handle);

    /// <summary>Writes an immutable .vdb segment (atomic: tmp + fsync + rename).</summary>
    public void WriteSegment(string path)
        => NativeMethods.Check(NativeMethods.vdb_index_write(_handle, path));

    /// <summary>Opens a segment via mmap; searches run directly over the file.</summary>
    public static VdbIndex Open(string path, bool validateCrcs = true)
    {
        NativeMethods.Check(NativeMethods.vdb_index_open(path, validateCrcs ? 1 : 0,
            out var handle));
        return new VdbIndex(handle);
    }

    public unsafe SearchHit[] Search(ReadOnlySpan<float> query, uint k, uint ef = 100)
    {
        var ids = new ulong[k];
        var scores = new float[k];
        uint found;
        fixed (float* q = query)
        fixed (ulong* pi = ids)
        fixed (float* ps = scores)
        {
            NativeMethods.Check(NativeMethods.vdb_index_search(_handle, q,
                (uint)query.Length, k, ef, pi, ps, out found));
        }
        var hits = new SearchHit[found];
        for (var i = 0; i < found; i++) hits[i] = new SearchHit(ids[i], scores[i]);
        return hits;
    }

    public void Dispose() => _handle.Dispose();
}
