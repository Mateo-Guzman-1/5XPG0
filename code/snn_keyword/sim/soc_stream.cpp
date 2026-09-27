// Streaming firmware (ABI v3) on the real PicoRV32 SoC RTL, checked against the integer oracle.
// usage: Vspike_soc firmware.bin streams.bin expected.bin results.csv hop_frames
//   streams.bin:  records of uint32 n_frames + n_frames x 24 uint8 (state is reset per record)
//   expected.bin: per frame int32 score, layer-1 spikes, layer-2 spikes (model.integer_forward_stream)
// Every hop must give the oracle's best score, last score and spike count, and the
// hold-off decision computed here from the oracle scores.
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
    uint64_t cycles = 0, led_rise = 0, last_pulse = 0;
    unsigned prev_led = 0;
    void tick() {
        d.clk = 0; d.eval(); d.clk = 1; d.eval(); ++cycles;
        if (d.led_o && !prev_led) led_rise = cycles;
        if (!d.led_o && prev_led) last_pulse = cycles - led_rise;
        prev_led = d.led_o;
        if (d.core_rst_n && d.core_trap) throw std::runtime_error("CPU trap");
    }
    void write(unsigned a, uint32_t v) {
        d.pa_addr = a / 4; d.pa_wdata = v; d.pa_be = 15; d.pa_we = 1; tick(); d.pa_we = 0;
    }
    uint32_t read(unsigned a) { d.pa_addr = a / 4; tick(); return d.pa_rdata; }
    void wait(unsigned a, uint32_t v, uint64_t budget = 2000000000ull) {
        uint64_t end = cycles + budget;
        while (read(a) != v) if (cycles >= end) throw std::runtime_error("Mailbox timeout");
    }
};

static std::vector<uint8_t> bytes(const char *p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot read ") + p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static const unsigned MB = 0x10400, INPUT = 0x10800, HOLDOFF = 100;
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
        const int32_t threshold = int32_t(s.read(mb(13)));
        const unsigned fb = s.read(mb(14)), maxf = s.read(mb(15));
        if (fb != 24 || hop > maxf) throw std::runtime_error("Unexpected frame size or hop");
        const unsigned window = s.read(mb(19));   // detection score: sum of the last `window` frame scores
        if (window < 1 || window > 64) throw std::runtime_error("Unexpected decision window");
        uint32_t seq = 0;
        auto command = [&](unsigned op, unsigned len) {
            s.write(mb(2), op); s.write(mb(3), len); s.write(mb(0), ++seq); s.wait(mb(1), seq);
            return s.read(mb(6));
        };
        // Malformed requests are rejected: partial frame, too many frames, unknown opcode.
        if (command(4, fb - 1) != 1 || command(4, (maxf + 1) * fb) != 1 || command(99, fb) != 1)
            throw std::runtime_error("Malformed request accepted");
        std::ofstream out(argv[4]);
        out << "record,hop,frames,best,last,spikes,events,cycles,detected,p_layer1,p_delayed,p_recurrent,p_layer2,p_readout\n";
        size_t pos = 0, e = 0;
        unsigned record = 0, hops = 0;
        uint64_t worst = 0, full_hops = 0, full_cycles = 0;
        bool pulse_checked = false;
        while (pos + 4 <= streams.size()) {
            uint32_t n = 0;
            for (unsigned b = 0; b < 4; ++b) n |= uint32_t(streams[pos + b]) << (8 * b);
            pos += 4;
            if (command(5, 0) != 0 || s.read(mb(18)) != 0) throw std::runtime_error("Reset failed");
            bool have = false; uint32_t last_event = 0;
            std::vector<int32_t> ring(window, 0); unsigned rpos = 0; int64_t dsum = 0;
            for (uint32_t f0 = 0; f0 < n; f0 += hop) {
                const unsigned k = std::min<uint32_t>(hop, n - f0);
                for (unsigned a = 0; a < k * fb; a += 4) {
                    uint32_t w = 0;
                    for (unsigned b = 0; b < 4; ++b) w |= uint32_t(streams[pos + a + b]) << (8 * b);
                    s.write(INPUT + a, w);
                }
                pos += k * fb;
                if (command(4, k * fb) != 0) throw std::runtime_error("Stream request failed");
                int32_t best = int32_t(s.read(mb(7))), last = int32_t(s.read(mb(8)));
                uint32_t cyc = s.read(mb(9)), spikes = s.read(mb(10)), det = s.read(mb(11));
                uint32_t events = s.read(mb(16)), at = s.read(mb(17)), frames = s.read(mb(18));
                // Oracle values for this hop.
                int32_t xb = INT32_MIN, xl = 0; uint32_t xs = 0, xd = 0, xat = ~0u;
                for (unsigned f = 0; f < k; ++f, e += 3) {
                    const int32_t sc = expected[e];
                    xb = std::max(xb, sc); xl = sc; xs += uint32_t(expected[e + 1] + expected[e + 2]);
                    const uint32_t idx = f0 + f;
                    dsum += int64_t(sc) - ring[rpos]; ring[rpos] = sc; rpos = (rpos + 1) % window;
                    if (dsum >= threshold) {
                        xd |= 2;
                        if (!have || idx - last_event >= HOLDOFF) {
                            have = true; last_event = idx;
                            if (!(xd & 1)) xat = f;
                            xd |= 1;
                        }
                    }
                }
                if (best != xb || last != xl || spikes != xs || det != xd || at != xat || frames != f0 + k) {
                    std::cerr << "record " << record << " frame " << f0 << ": best " << best << "/" << xb
                              << " last " << last << "/" << xl << " spikes " << spikes << "/" << xs
                              << " detected " << det << "/" << xd << '\n';
                    throw std::runtime_error("Oracle mismatch");
                }
                out << record << ',' << hops << ',' << k << ',' << best << ',' << last << ',' << spikes << ','
                    << events << ',' << cyc << ',' << det;
                for (unsigned w = 20; w < 25; ++w) out << ',' << s.read(mb(w));  // section cycles, PROFILE builds only
                out << '\n';
                if (k == hop) { worst = std::max<uint64_t>(worst, cyc); full_cycles += cyc; ++full_hops; }
                ++hops;
                if ((det & 1) && !pulse_checked) {
                    if (!s.d.led_o) throw std::runtime_error("LED did not turn on");
                    while (s.d.led_o) s.tick();
                    if (s.last_pulse != 100000000) throw std::runtime_error("LED pulse is not exactly one second");
                    pulse_checked = true;
                }
            }
            // Re-reading the acknowledgement does not execute the last command again.
            const uint32_t c = s.read(mb(9));
            for (unsigned t = 0; t < 100; ++t) s.tick();
            if (s.read(mb(1)) != seq || s.read(mb(9)) != c) throw std::runtime_error("Duplicate request executed");
            ++record;
        }
        if (e != expected.size()) throw std::runtime_error("Expected-value file length mismatch");
        std::cout << "PASS: " << record << " streams, " << hops << " hops bit-exact with the oracle; malformed "
                  << "requests, reset, duplicate sequence" << (pulse_checked ? ", 1 s LED pulse" : ", LED not exercised")
                  << "; worst " << worst << " cycles per " << hop << "-frame hop, mean "
                  << (full_hops ? full_cycles / full_hops : 0) << "; total cycles=" << s.cycles << '\n';
    } catch (const std::exception &ex) { std::cerr << ex.what() << '\n'; return 1; }
}
