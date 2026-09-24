// Execute real RV32IM firmware on the repository's PicoRV32 + BRAM + peripherals.
// Host drives the BRAM loading/mailbox port exactly as the PS does.
#include "Vspike_soc.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
    void wait(unsigned a, uint32_t v, uint64_t budget=200000000) {
        uint64_t end = cycles + budget;
        while (read(a) != v) if (cycles >= end) throw std::runtime_error("Mailbox timeout");
    }
};

static std::vector<uint8_t> bytes(const char *p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read input file");
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (argc != 5) throw std::runtime_error("usage: Vspike_soc firmware.bin vectors.bin results.csv threshold");
        const int32_t threshold = std::stoi(argv[4]);
        auto fw = bytes(argv[1]), input = bytes(argv[2]);
        if (fw.size() > 0x3c000 || input.size() % 768) throw std::runtime_error("Invalid file size");
        Sim s; s.d.core_rst_n = 0; s.d.pa_we = 0;
        // Entire memory initialized through the host port, never hierarchical preload.
        for (unsigned a=0; a<0x40000; a+=4) {
            uint32_t w=0;
            for (unsigned b=0;b<4;++b) if (a+b<fw.size()) w |= uint32_t(fw[a+b]) << (8*b);
            s.write(a,w);
        }
        s.d.core_rst_n = 1;
        s.wait(0x10430, 0x4b575331);
        std::ofstream out(argv[3]);
        out << "index,score0,score1,spikes,cycles,detected\n";
        uint32_t seq=0;
        // Reject malformed requests, then demonstrate recovery.
        for (unsigned op : {1u, 99u}) {
            s.write(0x10408, op); s.write(0x1040c, 767); s.write(0x10400, ++seq);
            s.wait(0x10404, seq);
            if (s.read(0x10418) != 1) throw std::runtime_error("Malformed request accepted");
        }
        bool pulse_checked=false;
        for (size_t n=0; n<input.size()/768; ++n) {
            for (unsigned a=0; a<768; a+=4) {
                uint32_t w=0;
                for (unsigned b=0;b<4;++b) w |= uint32_t(input[n*768+a+b]) << (8*b);
                s.write(0x10800+a,w);
            }
            s.write(0x10408,1); s.write(0x1040c,768); s.write(0x10400,++seq);
            s.wait(0x10404,seq);
            if (s.read(0x10418)) throw std::runtime_error("Inference failed");
            int32_t score0=s.read(0x1041c), score1=s.read(0x10420);
            uint32_t c=s.read(0x10424), spikes=s.read(0x10428), detected=s.read(0x1042c);
            if (detected != unsigned(score1-score0 >= threshold)) throw std::runtime_error("Decision mismatch");
            out << n << ',' << score0 << ',' << score1 << ',' << spikes << ',' << c << ',' << detected << '\n';
            out.flush();
            if (detected && !pulse_checked) {
                if (!s.d.led_o) throw std::runtime_error("LED did not turn on");
                while (s.d.led_o) s.tick();
                if (s.last_pulse != 100000000) throw std::runtime_error("LED pulse is not exactly one second");
                pulse_checked=true;
            }
            // Re-reading the acknowledgement does not execute the command again.
            for (unsigned k=0;k<100;++k) s.tick();
            if (s.read(0x10404)!=seq || s.read(0x10424)!=c) throw std::runtime_error("Duplicate request executed");
        }
        if (!pulse_checked) throw std::runtime_error("No positive test exercised LED");
        std::cout << "PASS: " << input.size()/768 << " RV32IM inferences, malformed requests, duplicate sequence, 100000000-cycle LED pulse; total cycles=" << s.cycles << '\n';
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
