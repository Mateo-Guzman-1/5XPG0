// Verifier track: stage-1 firmware with the phoneme verifier (USE_VERIFIER) on the PicoRV32 SoC RTL.
// usage: Vspike_soc firmware.bin streams.bin expected.bin results.csv hop
//   streams.bin:  records of uint32 n_frames + n_frames x 24 uint8 (state is reset per record)
//   expected.bin: per record int32 score_a, end_a, score_b, end_b, logit hash of the verifier on
//                 the last VERIFIER_FRAMES frames (zeros before the record start)
// Each record is streamed in hops (command 4), then command 6 verifies the history. The first
// record is also verified from the input buffer (command 6 with the frames). Cycles per call.
#include "Vspike_soc.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Sim {
    Vspike_soc d;
    uint64_t cycles = 0;
    void tick() {
        d.clk = 0; d.eval(); d.clk = 1; d.eval(); ++cycles;
        if (d.core_rst_n && d.core_trap) throw std::runtime_error("CPU trap");
    }
    void write(unsigned a, uint32_t v) {
        d.pa_addr = a / 4; d.pa_wdata = v; d.pa_be = 15; d.pa_we = 1; tick(); d.pa_we = 0;
    }
    uint32_t read(unsigned a) { d.pa_addr = a / 4; tick(); return d.pa_rdata; }
    void wait(unsigned a, uint32_t v, uint64_t budget = 4000000000ull) {
        uint64_t end = cycles + budget;
        while (read(a) != v) if (cycles >= end) throw std::runtime_error("Mailbox timeout");
    }
};

static std::vector<uint8_t> bytes(const char *p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot read ") + p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static const unsigned MB = 0x10400, INPUT = 0x10800, VFRAMES = 150, FB = 24;
static unsigned mb(unsigned w) { return MB + 4 * w; }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc != 6) throw std::runtime_error("usage: Vspike_soc firmware.bin streams.bin expected.bin results.csv hop");
        auto fw = bytes(argv[1]), streams = bytes(argv[2]), exp_raw = bytes(argv[3]);
        const unsigned hop = unsigned(std::stoul(argv[5]));
        std::vector<int32_t> expected(exp_raw.size() / 4);
        std::copy(exp_raw.begin(), exp_raw.end(), reinterpret_cast<uint8_t *>(expected.data()));
        if (fw.size() > 0x3c000) throw std::runtime_error("Invalid firmware size");
        Sim s; s.d.core_rst_n = 0; s.d.pa_we = 0;
        for (unsigned a = 0; a < 0x40000; a += 4) {
            uint32_t w = 0;
            for (unsigned b = 0; b < 4; ++b) if (a + b < fw.size()) w |= uint32_t(fw[a + b]) << (8 * b);
            s.write(a, w);
        }
        s.d.core_rst_n = 1;
        s.wait(mb(12), 0x4b575333u);
        uint32_t seq = 0;
        auto command = [&](unsigned op, unsigned len) {
            s.write(mb(2), op); s.write(mb(3), len); s.write(mb(0), ++seq); s.wait(mb(1), seq);
            return s.read(mb(6));
        };
        if (command(6, 7) != 1) throw std::runtime_error("Malformed verifier request accepted");
        std::ofstream out(argv[4]);
        out << "record,frames,mode,score_a,end_a,score_b,end_b,hash,cycles\n";
        size_t pos = 0, e = 0;
        unsigned record = 0;
        uint64_t worst = 0, total = 0, calls = 0;
        while (pos + 4 <= streams.size()) {
            uint32_t n = 0;
            for (unsigned b = 0; b < 4; ++b) n |= uint32_t(streams[pos + b]) << (8 * b);
            pos += 4;
            const size_t rec_start = pos;
            if (command(5, 0) != 0) throw std::runtime_error("Reset failed");
            for (uint32_t f0 = 0; f0 < n; f0 += hop) {
                const unsigned k = std::min<uint32_t>(hop, n - f0);
                for (unsigned a = 0; a < k * FB; a += 4) {
                    uint32_t w = 0;
                    for (unsigned b = 0; b < 4; ++b) w |= uint32_t(streams[pos + a + b]) << (8 * b);
                    s.write(INPUT + a, w);
                }
                pos += k * FB;
                if (command(4, k * FB) != 0) throw std::runtime_error("Stream request failed");
            }
            for (int mode = 0; mode < (record == 0 ? 2 : 1); ++mode) {
                if (mode == 1) {   // the same window from the input buffer (zero frames before the start)
                    for (unsigned f = 0; f < VFRAMES; ++f) {
                        const long src = long(n) - long(VFRAMES) + long(f);
                        for (unsigned a = 0; a < FB; a += 4) {
                            uint32_t w = 0;
                            for (unsigned b = 0; b < 4; ++b)
                                if (src >= 0) w |= uint32_t(streams[rec_start + size_t(src) * FB + a + b]) << (8 * b);
                            s.write(INPUT + f * FB + a, w);
                        }
                    }
                }
                if (command(6, mode ? VFRAMES * FB : 0) != 0) throw std::runtime_error("Verifier request failed");
                int32_t got[5];
                for (unsigned w = 0; w < 5; ++w) got[w] = int32_t(s.read(mb(26 + w)));
                const uint32_t cyc = s.read(mb(9));
                for (unsigned w = 0; w < 5; ++w)
                    if (got[w] != expected[e + w]) {
                        std::cerr << "record " << record << " mode " << mode << " word " << w << ": " << got[w]
                                  << " expected " << expected[e + w] << '\n';
                        throw std::runtime_error("Oracle mismatch");
                    }
                out << record << ',' << n << ',' << (mode ? "input" : "history");
                for (unsigned w = 0; w < 5; ++w) out << ',' << got[w];
                out << ',' << cyc << '\n';
                worst = std::max<uint64_t>(worst, cyc); total += cyc; ++calls;
            }
            e += 5;
            ++record;
        }
        if (e != expected.size()) throw std::runtime_error("Expected-value file length mismatch");
        std::cout << "PASS: " << record << " records, " << calls << " verifier calls bit-exact with the oracle"
                  << "; worst " << worst << " cycles per verification, mean " << (calls ? total / calls : 0)
                  << "; total cycles=" << s.cycles << '\n';
    } catch (const std::exception &ex) { std::cerr << ex.what() << '\n'; return 1; }
}
