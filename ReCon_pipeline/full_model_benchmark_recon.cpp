// full_model_benchmark_qad.cpp
// ─────────────────────────────────────────────────────────────
// Dedicated on-device benchmark for the TinyBERT E8-lattice QAD
// student (stage_e8_qad.py v2 + export_qad_recon_models.py v2).
// Kept as its own file (not merged into the QAT harness) since this
// is a single, standalone model -- matches the project's convention
// of separate, clearly-named files rather than one file trying to
// cover every model.

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <arm_neon.h>
#include <unistd.h>


//  STRUCTS

#pragma pack(push, 1)
struct block_e8_8 { float scale; uint8_t idx[4]; };
#pragma pack(pop)

static_assert(sizeof(block_e8_8) == 8, "");


//  CODEBOOK
//  NOTE: this is the QAD-specific codebook (tinybert_e8_qad_codebook_8bit.bin),
//  built by export_qad_recon_models.py with the SAME construction as
//  stage_e8_qad.py's training-time codebook. It is NOT the same file
//  as the QAT pipeline's E8 codebook -- the two are built with
//  different code and are not guaranteed to match point-for-point.
//  Loading the wrong one here would silently corrupt every decode.

static float cb8[256][8];


//  METRICS

long get_rss_kb() {
    std::ifstream f("/proc/self/status"); if (!f) return 0;
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            long v = 0;
            std::istringstream(line.substr(6)) >> v;
            return v;
        }
    return 0;
}

struct CpuSnap {
    long jiffies = 0;
    std::chrono::steady_clock::time_point wall;
};
CpuSnap cpu_snap() {
    CpuSnap s; s.wall = std::chrono::steady_clock::now();
    std::ifstream f("/proc/self/stat"); if (!f) return s;
    std::string t;
    for (int i=1;i<=15;++i){
        f>>t;
        if(i==14) s.jiffies =std::stol(t);
        if(i==15) s.jiffies+=std::stol(t);
    }
    return s;
}
double cpu_pct(const CpuSnap& a, const CpuSnap& b) {
    double cs=(b.jiffies-a.jiffies)/(double)sysconf(_SC_CLK_TCK);
    double ws=std::chrono::duration<double>(b.wall-a.wall).count();
    double p = ws>0 ? cs/ws*100.0 : 0.0;
    return p > 100.0 ? 100.0 : p;
}


//  STREAMING LOADER

template<typename B>
bool stream_load(const std::string& path, std::vector<B>& out,
                 const std::string& label) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "  Cannot open: " << path << "\n"; return false; }
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), (long long)n * sizeof(B));
    double mb = (double)n * sizeof(B) / (1024.0*1024.0);
    std::cout << "  Streamed " << label << ": "
              << n << " blocks  (" << mb << " MB)\n";
    return true;
}


//  NEON KERNEL

inline float dot_e8_8(const block_e8_8& b, const float* x) {
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
    float32x4_t vs=vdupq_n_f32(b.scale);
    for (int i=0;i<4;++i){
        uint8_t id=b.idx[i];
        a0=vmlaq_f32(a0,vmulq_f32(vld1q_f32(&cb8[id][0]),vs),
                        vld1q_f32(&x[i*8  ]));
        a1=vmlaq_f32(a1,vmulq_f32(vld1q_f32(&cb8[id][4]),vs),
                        vld1q_f32(&x[i*8+4]));
    }
    float32x4_t t=vaddq_f32(a0,a1);
    return vgetq_lane_f32(t,0)+vgetq_lane_f32(t,1)
          +vgetq_lane_f32(t,2)+vgetq_lane_f32(t,3);
}


//  FULL-PASS BENCHMARK
//  Streams through all model blocks ITERS times.

struct FPResult { double ms; double cpu; long rss; double checksum; };

template<typename B, typename DotFn>
FPResult bench_fullpass(const std::vector<B>& model,
                        const std::vector<float>& x,
                        DotFn dot, int iters)
{
    uint32_t n    = (uint32_t)model.size();
    int      xmod = (int)x.size() - 32;

    double sink = 0;
    for (int w=0;w<3;++w)
        for (uint32_t b=0;b<n;++b)
            sink += dot(model[b], &x[(b*32)%xmod]);

    CpuSnap c0 = cpu_snap();
    auto    t0 = std::chrono::high_resolution_clock::now();
    double  cs = sink * 0;
    for (int it=0;it<iters;++it)
        for (uint32_t b=0;b<n;++b)
            cs += dot(model[b], &x[(b*32)%xmod]);
    auto    t1 = std::chrono::high_resolution_clock::now();
    CpuSnap c1 = cpu_snap();

    double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    return {ms, cpu_pct(c0,c1), get_rss_kb(), cs};
}


//  SINGLE-LAYER LATENCY BENCHMARK
//  total_layer_blocks = rows*ceil(cols/32), NOT floor((rows*cols)/32) --
//  matches how export_qad_recon_models.py pads each row individually
//  to a multiple of 32 (needed since TinyBERT's 312 isn't 32-aligned).

template<typename B, typename DotFn>
double bench_layer(const std::vector<B>& model,
                   const std::vector<float>& x,
                   DotFn dot,
                   int rows, int cols,
                   double& layer_checksum_out)
{
    long long total_layer_blocks = (long long)rows * (((long long)cols + 31) / 32);

    if ((long long)model.size() < total_layer_blocks) {
        std::cout << "    WARNING: model has " << model.size()
                  << " blocks, need " << total_layer_blocks
                  << " for " << rows << "x" << cols << " layer\n";
        layer_checksum_out = 0;
        return -1.0;
    }

    int xmod = (int)x.size() - 32;
    const int WARMUP = 500;
    const int ITERS  = 10000;

    double sink = 0;
    for (int w=0;w<WARMUP;++w)
        for (long long b=0;b<total_layer_blocks;++b)
            sink += dot(model[b], &x[(b*32)%xmod]);

    auto t0 = std::chrono::high_resolution_clock::now();
    double cs = sink * 0;
    for (int it=0;it<ITERS;++it)
        for (long long b=0;b<total_layer_blocks;++b)
            cs += dot(model[b], &x[(b*32)%xmod]);
    auto t1 = std::chrono::high_resolution_clock::now();

    layer_checksum_out = std::fabs(cs);
    double total_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    return (total_ms * 1000.0) / ITERS;   // µs per layer
}


//  PRINT RESULT

void print_result(const std::string& label,
                  long rss_load, long rss_bench,
                  double full_ms, double layer_us,
                  double cpu, double fp_checksum,
                  double layer_checksum,
                  int rows, int cols,
                  double bits, double snr)
{
    std::cout << "\n  ┌──────────────────────────────────────────────────┐\n";
    std::cout << "  │  " << label << "\n";
    std::cout << "  ├──────────────────────────────────────────────────┤\n";
    std::cout << "  │  Layer shape          : "<<rows<<" x "<<cols<<"\n";
    std::cout << "  │  Bits / weight        : "<<bits<<"\n";
    std::cout << "  │  SNR                  : "
              <<(snr<0?"inf (lossless)":std::to_string(snr)+" dB")<<"\n";
    std::cout << "  ├──────────────────────────────────────────────────┤\n";
    std::cout << "  │  RSS RAM (post-load)  : "<<rss_load
              <<" kB  (~"<<rss_load/1024<<" MB)\n";
    std::cout << "  │  RSS RAM (post-bench) : "<<rss_bench
              <<" kB  (~"<<rss_bench/1024<<" MB)\n";
    std::cout << "  ├──────────────────────────────────────────────────┤\n";
    std::cout << "  │  Full-pass time       : "<<full_ms<<" ms\n";
    if (layer_us > 0)
        std::cout << "  │  Single-layer latency : "<<layer_us<<" µs\n";
    else
        std::cout << "  │  Single-layer latency : N/A (model too small)\n";
    std::cout << "  │  CPU Utilisation      : "<<cpu<<" %\n";
    std::cout << "  │  Full-pass checksum   : "<<fp_checksum<<"\n";
    std::cout << "  │  Layer checksum       : "<<layer_checksum<<"\n";
    std::cout << "  └──────────────────────────────────────────────────┘\n\n";
}


//  MAIN
//  IMPORTANT: the SNR value passed to print_result below is a
//  PLACEHOLDER (0.0) -- replace it with whatever exact-decomposition
//  SNR export_qad_recon_models.py printed for YOUR run before
//  reporting this number anywhere. Baking in a stale/guessed SNR here
//  was exactly the bug in the original harness (it displayed the old
//  file's 3.87dB even after re-exporting).

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  TinyBERT E8-Lattice QAD Benchmark (ReCon)           ║\n";
    std::cout << "║  RSS + Full-pass time + Single-layer latency         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    {
        std::ifstream f("/data/local/tmp/tinybert_e8_qad_codebook_8bit.bin",
                        std::ios::binary);
        if(!f){std::cerr<<"Cannot open QAD codebook\n";return 1;}
        f.read(reinterpret_cast<char*>(cb8),sizeof(cb8));
        std::cout<<"QAD E8 8-bit codebook loaded (8 KB)\n\n";
    }

    const int MAXDIM = 1024+32;
    std::vector<float> x(MAXDIM);
    for(int i=0;i<MAXDIM;++i) x[i]=std::sin(i*0.1f);

    long   rss_load=0;
    double lcs=0;

    std::cout<<"══════ TinyBERT + E8 (QAD) ══════\n";
    { std::vector<block_e8_8> m;
      if(!stream_load("/data/local/tmp/tinybert_e8_qad_quant.bin", m, "TinyBERT E8-QAD")) return 1;
      rss_load=get_rss_kb();
      auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},5);
      double lu=bench_layer(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},312,312,lcs);
      // TODO: replace 0.0 with the actual SNR printed by
      // export_qad_recon_models.py for this run.
      print_result("TinyBERT + E8 (QAD, ReCon)",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,2.0,0.0);
    }

    std::cout<<"Final RSS: "<<get_rss_kb()/1024<<" MB\n";
    return 0;
}
