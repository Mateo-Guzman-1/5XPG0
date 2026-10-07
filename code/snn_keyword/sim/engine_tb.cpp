// Unit test of rtl/neuron_engine.v against a C++ model of the streaming SNN
// (the arithmetic of firmware/stream_infer.c). Random tables within the export
// bounds. First half of the frames: random layer-1 spike patterns from silent to
// every neuron firing, sent as SPIKE writes (CTRL bit1). Second half: random
// input frames from silent to saturated, with layer 1 in the engine (FRAME,
// CTRL bit2). State resets in both. Reports mismatches and the engine cycles
// per frame.
// usage: Vneuron_engine [frames] [seed]
#include "Vneuron_engine.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static const int N1 = 128, N2 = 128, C = 3, YES = 2, PO = 3;

struct Model {
    // tables
    std::vector<uint16_t> off; std::vector<uint8_t> post; std::vector<int8_t> w;
    std::vector<int8_t> rect;        // rec[j][k] at k*N2 + j
    std::vector<int8_t> wot;         // j*C + c
    std::vector<int32_t> theta, p, km, ka, bq, b2, ko, bo;
    std::vector<int8_t> w1;          // h*24 + i
    std::vector<int32_t> theta1, r1, km1, ka1, bq1, b1;
    // state
    std::vector<std::vector<uint8_t>> ring{32}; int pos = 0;
    int32_t u[N2] = {}, a[N2] = {}; uint8_t s[N2] = {}; int32_t o[4] = {};
    int32_t u1[N1] = {}, a1[N1] = {}; uint8_t s1[N1] = {};
    static int32_t sat16(int64_t v) { return int32_t(std::min<int64_t>(32767, std::max<int64_t>(-32768, v))); }
    static int32_t sat32(int64_t v) { return int32_t(std::min<int64_t>(INT32_MAX, std::max<int64_t>(INT32_MIN, v))); }
    void reset() { for (auto &r : ring) r.clear(); pos = 0; std::fill(u, u + N2, 0); std::fill(a, a + N2, 0);
                   std::fill(s, s + N2, 0); std::fill(o, o + 4, 0);
                   std::fill(u1, u1 + N1, 0); std::fill(a1, a1 + N1, 0); std::fill(s1, s1 + N1, 0); }
    // Layer 1 of one frame: returns the indices of the neurons that spike.
    std::vector<uint8_t> layer1(const uint8_t *x) {
        std::vector<uint8_t> l1;
        for (int h = 0; h < N1; ++h) {
            int64_t acc = 0;
            for (int i = 0; i < 24; ++i) acc += int64_t(w1[h * 24 + i]) * x[i];
            int64_t cur = (acc >> r1[h]) + b1[h];
            a1[h] = a1[h] - (a1[h] >> ka1[h]) + (int32_t(s1[h]) << 8);
            int64_t thr = theta1[h] + ((int64_t(bq1[h]) * a1[h]) >> 8);
            u1[h] = sat16(int64_t(u1[h]) - (u1[h] >> km1[h]) + cur);
            s1[h] = u1[h] >= thr;
            if (s1[h]) { u1[h] = sat16(int64_t(u1[h]) - thr); l1.push_back(uint8_t(h)); }
        }
        return l1;
    }
    // returns score; spikes2, events out
    int32_t step(const std::vector<uint8_t> &l1, uint32_t &n2, uint32_t &ev) {
        ring[pos] = l1;
        int64_t acc[N2] = {}; ev = 0;
        for (int tau = 0; tau < 32; ++tau)
            for (uint8_t i : ring[(pos - tau) & 31]) {
                int k = i * 32 + tau;
                for (int e = off[k]; e < off[k + 1]; ++e) acc[post[e]] += w[e];
                ev += off[k + 1] - off[k];
            }
        for (int k = 0; k < N2; ++k) if (s[k]) { for (int jj = 0; jj < N2; ++jj) acc[jj] += rect[k * N2 + jj]; ev += N2; }
        int64_t acc_o[4] = {}; n2 = 0;
        for (int jj = 0; jj < N2; ++jj) {
            int64_t cur = acc[jj] * (int64_t(1) << p[jj]) + b2[jj];
            a[jj] = a[jj] - (a[jj] >> ka[jj]) + (int32_t(s[jj]) << 8);
            int64_t thr = theta[jj] + ((int64_t(bq[jj]) * a[jj]) >> 8);
            u[jj] = sat16(int64_t(u[jj]) - (u[jj] >> km[jj]) + cur);
            s[jj] = u[jj] >= thr;
            if (s[jj]) { u[jj] = sat16(int64_t(u[jj]) - thr); ++n2; ev += C;
                         for (int c = 0; c < C; ++c) acc_o[c] += wot[jj * C + c]; }
        }
        int64_t best = INT64_MIN;
        for (int c = 0; c < C; ++c) {
            o[c] = sat32(int64_t(o[c]) - (o[c] >> ko[c]) + acc_o[c] * (int64_t(1) << PO) + bo[c]);
            if (c != YES) best = std::max<int64_t>(best, o[c]);
        }
        pos = (pos + 1) & 31;
        return sat32(o[YES] - best);
    }
};

struct Tb {
    Vneuron_engine d; uint64_t cycles = 0;
    void tick() { d.clk = 0; d.eval(); d.clk = 1; d.eval(); ++cycles; }
    void write(uint8_t addr, uint32_t v) { d.wr = 1; d.addr = addr; d.wdata = v; tick(); d.wr = 0; }
    uint32_t read(uint8_t addr) { d.raddr = addr; d.eval(); return d.rdata; }
    void wait_idle() { for (int t = 0; read(0x04) & 1; ++t) { if (t > 10000000) throw std::runtime_error("engine hang"); tick(); } }
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    const int frames = argc > 1 ? atoi(argv[1]) : 3000;
    std::mt19937 rng(argc > 2 ? atoi(argv[2]) : 1);
    auto uni = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    try {
        Model m;
        // Random model within the export bounds (every synapse present, random delays).
        std::vector<std::vector<std::pair<uint8_t, int8_t>>> lists(N1 * 32);
        for (int jj = 0; jj < N2; ++jj)
            for (int i = 0; i < N1; ++i) {
                int8_t wv = int8_t(uni(-127, 127));
                if (wv) lists[i * 32 + uni(0, 31)].push_back({uint8_t(jj), wv});
            }
        m.off.push_back(0);
        for (auto &l : lists) { for (auto &pw : l) { m.post.push_back(pw.first); m.w.push_back(pw.second); } m.off.push_back(uint16_t(m.post.size())); }
        for (int t = 0; t < N2 * N2; ++t) m.rect.push_back(int8_t(uni(-127, 127)));
        for (int t = 0; t < N2 * C; ++t) m.wot.push_back(int8_t(uni(-127, 127)));
        for (int jj = 0; jj < N2; ++jj) {
            m.theta.push_back(uni(1024, 2047)); m.p.push_back(uni(0, 5)); m.km.push_back(uni(1, 7));
            m.ka.push_back(uni(2, 8)); m.bq.push_back(uni(0, 4000)); m.b2.push_back(uni(-60000, 20000));
        }
        for (int c = 0; c < C; ++c) { m.ko.push_back(uni(1, 6)); m.bo.push_back(uni(-3000, 3000)); }
        for (int t = 0; t < N1 * 24; ++t) m.w1.push_back(int8_t(uni(0, 9) ? uni(-128, 127) : 0));
        for (int h = 0; h < N1; ++h) {
            m.theta1.push_back(uni(1024, 2047)); m.r1.push_back(uni(0, 7)); m.km1.push_back(uni(1, 8));
            m.ka1.push_back(uni(2, 8)); m.bq1.push_back(uni(0, 4000)); m.b1.push_back(uni(-60000, 20000));
        }
        Tb tb; tb.d.wr = 0; tb.d.resetn = 0; tb.tick(); tb.d.resetn = 1; tb.wait_idle();
        if (tb.read(0x1c) != 0x4E454E47u) throw std::runtime_error("bad ID");
        auto table = [&](int t, const std::vector<uint32_t> &words) {
            tb.write(0x20, uint32_t(t) << 28); for (uint32_t v : words) tb.write(0x24, v); };
        std::vector<uint32_t> v;
        for (auto x : m.off) v.push_back(x); table(0, v); v.clear();
        for (size_t e = 0; e < m.post.size(); ++e) v.push_back(uint32_t(uint8_t(m.w[e])) << 8 | m.post[e]); table(1, v); v.clear();
        for (int t = 0; t < N2 * N2; t += 4) { uint32_t x = 0; for (int b = 0; b < 4; ++b) x |= uint32_t(uint8_t(m.rect[t + b])) << (8 * b); v.push_back(x); }
        table(2, v); v.clear();
        for (int jj = 0; jj < N2; ++jj) { uint32_t x = 0; for (int c = 0; c < C; ++c) x |= uint32_t(uint8_t(m.wot[jj * C + c])) << (8 * c); v.push_back(x); }
        table(3, v); v.clear();
        for (int jj = 0; jj < N2; ++jj) v.push_back(uint32_t(m.theta[jj]) | m.p[jj] << 16 | m.km[jj] << 20 | m.ka[jj] << 24); table(4, v); v.clear();
        for (int jj = 0; jj < N2; ++jj) v.push_back(uint32_t(m.bq[jj])); table(5, v); v.clear();
        for (int jj = 0; jj < N2; ++jj) v.push_back(uint32_t(m.b2[jj])); table(6, v); v.clear();
        for (int c = 0; c < C; ++c) v.push_back(uint32_t(m.ko[c])); table(7, v); v.clear();
        for (int c = 0; c < C; ++c) v.push_back(uint32_t(m.bo[c])); table(8, v); v.clear();
        for (int i = 0; i < 24; ++i)            // table 9: row = input * 8 + block, 4 words of 4 neurons
            for (int blk = 0; blk < 8; ++blk)
                for (int w = 0; w < 4; ++w) {
                    uint32_t x = 0;
                    for (int k = 0; k < 4; ++k) x |= uint32_t(uint8_t(m.w1[(blk * 16 + w * 4 + k) * 24 + i])) << (8 * k);
                    v.push_back(x);
                }
        table(9, v); v.clear();
        for (int h = 0; h < N1; ++h) v.push_back(uint32_t(m.theta1[h]) | m.r1[h] << 16 | m.km1[h] << 20 | m.ka1[h] << 24);
        table(10, v); v.clear();
        for (int h = 0; h < N1; ++h) v.push_back(uint32_t(m.bq1[h])); table(11, v); v.clear();
        for (int h = 0; h < N1; ++h) v.push_back(uint32_t(m.b1[h])); table(12, v); v.clear();
        tb.write(0x28, YES | C << 4 | PO << 8);
        tb.write(0x00, 1); tb.wait_idle(); m.reset();
        uint64_t worst = 0, worst_events = 0, worst_l1 = 0; int mism = 0, fires = 0, sat_frames = 0, l1_fires = 0;
        for (int f = 0; f < frames; ++f) {
            if (f % 500 == 499 || f == frames / 2) { tb.write(0x00, 1); tb.wait_idle(); m.reset(); }
            if (f >= frames / 2) {
                // Layer 1 in the engine: one random input frame, silent to saturated.
                int mode = uni(0, 9);
                uint8_t x[24];
                for (int i = 0; i < 24; ++i)
                    x[i] = uint8_t(mode == 0 ? 0 : mode == 1 ? 255 : mode == 2 ? (uni(0, 3) ? 0 : uni(0, 255)) : uni(0, 255));
                for (int k = 0; k < 24; k += 4)
                    tb.write(0x2c, uint32_t(x[k]) | uint32_t(x[k + 1]) << 8 | uint32_t(x[k + 2]) << 16 | uint32_t(x[k + 3]) << 24);
                tb.write(0x00, 4); tb.wait_idle();
                std::vector<uint8_t> l1 = m.layer1(x);
                uint32_t n2, ev; int32_t score = m.step(l1, n2, ev);
                uint32_t hw_score = tb.read(0x0c), hw_n2 = tb.read(0x10), hw_ev = tb.read(0x14), hw_n1 = tb.read(0x2c), cyc = tb.read(0x18);
                if (int32_t(hw_score) != score || hw_n2 != n2 || hw_ev != ev || hw_n1 != l1.size()) {
                    if (mism < 5) printf("L1 frame %d: score %d/%d spikes1 %u/%zu spikes2 %u/%u events %u/%u\n", f, int32_t(hw_score), score,
                                         hw_n1, l1.size(), hw_n2, n2, hw_ev, ev);
                    ++mism;
                }
                l1_fires += !l1.empty(); fires += n2 > 0; sat_frames += score == INT32_MAX || score == INT32_MIN;
                if (cyc > worst_l1) worst_l1 = cyc;
                continue;
            }
            // Activity from silent to saturated: density per frame drawn from a wide range.
            int mode = uni(0, 9);
            double density = mode == 0 ? 0.0 : mode == 9 ? 1.0 : mode == 8 ? uni(50, 100) / 100.0 : uni(0, 15) / 100.0;
            if (f % 1000 >= 900 && f % 1000 < 940) density = 1.0;   // 40-frame burst: every delay slot full
            std::vector<uint8_t> l1;
            for (int i = 0; i < N1; ++i) if (std::uniform_real_distribution<double>(0, 1)(rng) < density) l1.push_back(uint8_t(i));
            for (uint8_t i : l1) tb.write(0x08, i);
            tb.write(0x00, 2); tb.wait_idle();
            uint32_t n2, ev; int32_t score = m.step(l1, n2, ev);
            uint32_t hw_score = tb.read(0x0c), hw_n2 = tb.read(0x10), hw_ev = tb.read(0x14), cyc = tb.read(0x18);
            if (int32_t(hw_score) != score || hw_n2 != n2 || hw_ev != ev) {
                if (mism < 5) printf("frame %d: score %d/%d spikes %u/%u events %u/%u\n", f, int32_t(hw_score), score, hw_n2, n2, hw_ev, ev);
                ++mism;
            }
            fires += n2 > 0; sat_frames += score == INT32_MAX || score == INT32_MIN;
            if (cyc > worst) { worst = cyc; worst_events = ev; }
        }
        printf("%s: %d frames (%d with layer 1 in the engine), %d mismatches, %d frames with layer-1 spikes (engine layer 1), "
               "%d frames with layer-2 spikes, %d saturated scores; worst %llu engine cycles per frame with SPIKE input "
               "(%llu synaptic events), %llu with layer 1 in the engine\n", mism ? "FAIL" : "PASS", frames, frames - frames / 2, mism,
               l1_fires, fires, sat_frames, (unsigned long long)worst, (unsigned long long)worst_events, (unsigned long long)worst_l1);
        return mism ? 1 : 0;
    } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 2; }
}
