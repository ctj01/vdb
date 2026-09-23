/**
 * @file test_segment.cpp
 * @brief Phase-2 tests: roundtrip bit-exactness, per-section corruption
 *        detection, truncation/footer rejection, version gating, metadata
 *        edge cases (empty/unicode strings), empty segments.
 */
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "harness.hpp"
#include "segment/crc32c.hpp"
#include "segment/format.hpp"
#include "vdb/builder.hpp"
#include "vdb/search.hpp"
#include "vdb/segment.hpp"

using namespace vdb;

namespace {

const char* kPath = "test_segment.vdb";

std::vector<float> RandomVec(std::mt19937& rng, uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::string ReadAll(const char* path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteAll(const char* path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

FrozenIndex BuildSample(std::mt19937& rng, uint32_t n, uint32_t dim) {
    auto b = std::move(IndexBuilder::Create(dim, Metric::kCosine).value());
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(b.Add(5000 + i, RandomVec(rng, dim)).ok());
    }
    return std::move(std::move(b).Freeze().value());
}

std::vector<MetaColumn> SampleMeta(uint32_t n) {
    MetaColumn ids{"doc_id", MetaType::kInt64, {}, {}, {}};
    MetaColumn flags{"published", MetaType::kBool, {}, {}, {}};
    MetaColumn titles{"title", MetaType::kString, {}, {}, {}};
    for (uint32_t i = 0; i < n; ++i) {
        ids.i64.push_back(static_cast<int64_t>(i) * 37 - 1000);
        flags.bools.push_back(i % 3 == 0 ? 1 : 0);
        if (i % 5 == 0) {
            titles.strings.emplace_back();  // empty string edge case
        } else if (i % 5 == 1) {
            titles.strings.push_back("título #" + std::to_string(i) + " — čžß日本語🎯");
        } else {
            titles.strings.push_back("doc " + std::to_string(i));
        }
    }
    return {ids, flags, titles};
}

}  // namespace

int main() {
    std::mt19937 rng(321);
    constexpr uint32_t kN = 3000;
    constexpr uint32_t kDim = 100;  // stride pads 100 -> 112
    constexpr uint32_t kK = 10;

    const FrozenIndex mem = BuildSample(rng, kN, kDim);
    const auto meta = SampleMeta(kN);

    // ---- roundtrip: write -> open -> bit-identical search -------------------
    {
        auto w = WriteSegment(kPath, mem.View(), meta);
        CHECK(w.ok());
        auto ro = Segment::Open(kPath);
        CHECK(ro.ok());
        if (!ro.ok()) {
            std::printf("open error: %s\n", ro.error().message.c_str());
            return TestSummary("test_segment");
        }
        const Segment seg = std::move(ro).value();
        const IndexView& disk = seg.View();
        CHECK(disk.count == kN);
        CHECK(disk.dim == kDim);
        CHECK(disk.stride == PaddedStride(kDim));
        CHECK(disk.metric == Metric::kCosine);

        int mismatches = 0;
        for (int qi = 0; qi < 2000; ++qi) {
            const auto q = RandomVec(rng, kDim);
            const auto a = Search(mem.View(), q, kK, 100).value();
            const auto b = Search(disk, q, kK, 100).value();
            if (a.size() != b.size()) { ++mismatches; continue; }
            for (std::size_t j = 0; j < a.size(); ++j) {
                if (a[j].internal_id != b[j].internal_id ||
                    a[j].score != b[j].score ||           // bit-exact scores
                    a[j].external_id != b[j].external_id) {
                    ++mismatches;
                    break;
                }
            }
        }
        CHECK(mismatches == 0);

        // Metadata roundtrip.
        const auto& cols = seg.Meta();
        CHECK(cols.size() == 3);
        CHECK(cols[0].name == "doc_id");
        CHECK(cols[0].type == MetaType::kInt64);
        CHECK(cols[0].i64[7] == 7 * 37 - 1000);
        CHECK(cols[1].name == "published");
        CHECK(cols[1].bools[0] == 1);
        CHECK(cols[1].bools[1] == 0);
        CHECK(cols[2].name == "title");
        CHECK(cols[2].String(0).empty());
        CHECK(cols[2].String(1) == "título #1 — čžß日本語🎯");
        CHECK(cols[2].String(2) == "doc 2");
    }

    // ---- corruption: flip one byte inside each section + header -------------
    {
        const std::string good = ReadAll(kPath);
        CHECK(good.size() > fmt::kHeaderSize + fmt::kFooterSize);

        // Header corruption (byte inside the covered 128 bytes).
        {
            std::string bad = good;
            bad[13] = static_cast<char>(bad[13] ^ 0x40);
            WriteAll(kPath, bad);
            auto r = Segment::Open(kPath);
            CHECK(!r.ok());
            CHECK(r.error().code == ErrorCode::kCorrupted);
        }
        // One byte inside every section payload.
        const uint64_t table_off = fmt::GetU64(
            reinterpret_cast<const uint8_t*>(good.data()) + 40);
        for (uint32_t sec_i = 0; sec_i < 5; ++sec_i) {
            const uint8_t* e = reinterpret_cast<const uint8_t*>(good.data()) +
                               table_off + sec_i * fmt::kSectionEntrySize;
            const uint64_t off = fmt::GetU64(e + 4);
            const uint64_t len = fmt::GetU64(e + 12);
            if (len == 0) continue;
            std::string bad = good;
            bad[off + len / 2] = static_cast<char>(bad[off + len / 2] ^ 0x01);
            WriteAll(kPath, bad);
            auto r = Segment::Open(kPath);
            CHECK(!r.ok());
            CHECK(r.error().code == ErrorCode::kCorrupted);

            // With CRC validation disabled, big-section corruption is not
            // detected at open (documented escape hatch) — but the header
            // and structure still are. Vectors section: opens fine.
            if (fmt::GetU32(e) == fmt::kVectors) {
                OpenOptions lax;
                lax.validate_section_crcs = false;
                auto r2 = Segment::Open(kPath, lax);
                CHECK(r2.ok());
            }
        }

        // Truncated file.
        {
            WriteAll(kPath, good.substr(0, good.size() - 100));
            auto r = Segment::Open(kPath);
            CHECK(!r.ok());
            CHECK(r.error().code == ErrorCode::kCorrupted);
        }
        // Footer stripped exactly.
        {
            WriteAll(kPath, good.substr(0, good.size() - fmt::kFooterSize));
            auto r = Segment::Open(kPath);
            CHECK(!r.ok());
        }
        // Unknown future version_major (patch header + fix its CRC so ONLY
        // the version check can reject it).
        {
            std::string bad = good;
            bad[8] = 99;  // version_major low byte
            uint8_t header[fmt::kHeaderSize];
            std::memcpy(header, bad.data(), fmt::kHeaderSize);
            std::memset(header + 52, 0, 4);
            const uint32_t crc = Crc32c(header, fmt::kHeaderSize);
            std::string crc_bytes;
            fmt::PutU32(crc_bytes, crc);
            bad.replace(52, 4, crc_bytes);
            WriteAll(kPath, bad);
            auto r = Segment::Open(kPath);
            CHECK(!r.ok());
            CHECK(r.error().code == ErrorCode::kUnsupported);
        }
        WriteAll(kPath, good);  // restore for any later block
    }

    // ---- empty segment (count = 0) -------------------------------------------
    {
        auto b = std::move(IndexBuilder::Create(24, Metric::kL2).value());
        auto frozen = std::move(std::move(b).Freeze().value());
        CHECK(WriteSegment("test_empty.vdb", frozen.View(), {}).ok());
        auto r = Segment::Open("test_empty.vdb");
        CHECK(r.ok());
        const auto& v = r.value().View();
        CHECK(v.count == 0);
        CHECK(v.entry == kNoEntry);
        auto rs = Search(v, std::vector<float>(24, 1.0f), 5);
        CHECK(rs.ok());
        CHECK(rs.value().empty());
        std::remove("test_empty.vdb");
    }

    // ---- writer validation -----------------------------------------------------
    {
        MetaColumn short_col{"bad", MetaType::kInt64, {1, 2, 3}, {}, {}};
        auto r = WriteSegment("test_bad.vdb", mem.View(), {short_col});
        CHECK(!r.ok());
        CHECK(r.error().code == ErrorCode::kInvalidArgument);
    }

    std::remove(kPath);
    return TestSummary("test_segment");
}
