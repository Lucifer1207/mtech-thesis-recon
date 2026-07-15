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
struct block_fp32 { float    weights[32]; };
struct block_e8_16{ float    scale; uint16_t idx[4]; };
struct block_e8_8 { float    scale; uint8_t  idx[4]; };
#pragma pack(pop)

static_assert(sizeof(block_fp32)  == 128, "");
static_assert(sizeof(block_e8_16) ==  12, "");
static_assert(sizeof(block_e8_8)  ==   8, "");


//  CODEBOOKS

static float cb16[65536][8];
static float cb8 [256  ][8];   // E8
static float cb8z[256  ][8];   // Z8

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

//  NEON KERNELS

inline float dot_fp32(const block_fp32& b, const float* x) {
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),
                a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
    for (int i=0;i<32;i+=16){
        a0=vmlaq_f32(a0,vld1q_f32(&b.weights[i   ]),vld1q_f32(&x[i   ]));
        a1=vmlaq_f32(a1,vld1q_f32(&b.weights[i+ 4]),vld1q_f32(&x[i+ 4]));
        a2=vmlaq_f32(a2,vld1q_f32(&b.weights[i+ 8]),vld1q_f32(&x[i+ 8]));
        a3=vmlaq_f32(a3,vld1q_f32(&b.weights[i+12]),vld1q_f32(&x[i+12]));
    }
    float32x4_t t=vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3));
    return vgetq_lane_f32(t,0)+vgetq_lane_f32(t,1)
          +vgetq_lane_f32(t,2)+vgetq_lane_f32(t,3);
}

inline float dot_e8_16(const block_e8_16& b, const float* x) {
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
    float32x4_t vs=vdupq_n_f32(b.scale);
    for (int i=0;i<4;++i){
        uint16_t id=b.idx[i];
        a0=vmlaq_f32(a0,vmulq_f32(vld1q_f32(&cb16[id][0]),vs),
                        vld1q_f32(&x[i*8  ]));
        a1=vmlaq_f32(a1,vmulq_f32(vld1q_f32(&cb16[id][4]),vs),
                        vld1q_f32(&x[i*8+4]));
    }
    float32x4_t t=vaddq_f32(a0,a1);
    return vgetq_lane_f32(t,0)+vgetq_lane_f32(t,1)
          +vgetq_lane_f32(t,2)+vgetq_lane_f32(t,3);
}

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

// Z8 kernel -- byte-identical to dot_e8_8 except it indexes the Z8
// codebook (cb8z) instead of E8's (cb8). Block struct is the same
// block_e8_8 (float scale + 4x uint8 idx, 8 bytes) since both E8 and
// Z8 use the same 1-scale-per-32/256-codeword-8bit-index format.
inline float dot_z8_8(const block_e8_8& b, const float* x) {
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
    float32x4_t vs=vdupq_n_f32(b.scale);
    for (int i=0;i<4;++i){
        uint8_t id=b.idx[i];
        a0=vmlaq_f32(a0,vmulq_f32(vld1q_f32(&cb8z[id][0]),vs),
                        vld1q_f32(&x[i*8  ]));
        a1=vmlaq_f32(a1,vmulq_f32(vld1q_f32(&cb8z[id][4]),vs),
                        vld1q_f32(&x[i*8+4]));
    }
    float32x4_t t=vaddq_f32(a0,a1);
    return vgetq_lane_f32(t,0)+vgetq_lane_f32(t,1)
          +vgetq_lane_f32(t,2)+vgetq_lane_f32(t,3);
}


//  FULL-PASS BENCHMARK
//  Streams through all model blocks ITERS times.

template<typename B, typename DotFn>
struct FPResult { double ms; double cpu; long rss; double checksum; };

template<typename B, typename DotFn>
FPResult<B,DotFn> bench_fullpass(const std::vector<B>& model,
                                  const std::vector<float>& x,
                                  DotFn dot, int iters)
{
    uint32_t n    = (uint32_t)model.size();
    int      xmod = (int)x.size() - 32;

    // Warmup
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
//
//  IMPORTANT: total blocks for one layer are computed at the WHOLE-
//  MATRIX level -- (rows*cols)/32 -- not as rows*(cols/32). The two
//  are only equal when cols itself happens to be a multiple of 32
//  (true for BERT-Base's 768, false for TinyBERT's 312: 312/32=9.75).
//  Training/export flatten the entire weight tensor before chunking
//  into 32-element blocks, so blocks legitimately cross row
//  boundaries whenever cols isn't 32-aligned -- per-row block
//  indexing would silently read misaligned blocks for TinyBERT.
//  This benchmark instead treats the layer as one contiguous stream
//  of blocks (matching the true on-disk layout), which is sufficient
//  for a throughput/latency proxy -- per-row output separation isn't
//  needed here, only realistic total per-layer compute time.

template<typename B, typename DotFn>
double bench_layer(const std::vector<B>& model,
                   const std::vector<float>& x,
                   DotFn dot,
                   int rows, int cols,
                   double& layer_checksum_out)
{
    // rows * ceil(cols/32) -- NOT (rows*cols)/32. Export now pads each
    // ROW individually to a multiple of 32 (to match v5 training's
    // per-row scale), so a real exported layer has rows*ceil(cols/32)
    // blocks, not floor((rows*cols)/32) of them. Identical for shapes
    // already 32-aligned (e.g. BERT-Base's 768x768: both give 18432);
    // differs for TinyBERT's 312x312 (3120 vs 3042).
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

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  ReCon QAT Benchmark -- BERT-Base/TinyBERT x E8/Z8   ║\n";
    std::cout << "║  RSS + Full-pass time + Single-layer latency         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // Load codebooks
    // {
    //     std::ifstream f("/data/local/tmp/bert_codebook_e8_16bit.bin",
    //                     std::ios::binary);
    //     if(!f){std::cerr<<"Cannot open 16-bit codebook\n";return 1;}
    //     f.read(reinterpret_cast<char*>(cb16),sizeof(cb16));
    //     std::cout<<"16-bit codebook loaded (2 MB)\n";
    // }
    {
        std::ifstream f("/data/local/tmp/bert_codebook_e8_8bit.bin",
                        std::ios::binary);
        if(!f){std::cerr<<"Cannot open E8 8-bit codebook\n";return 1;}
        f.read(reinterpret_cast<char*>(cb8),sizeof(cb8));
        std::cout<<"E8 8-bit codebook loaded (8 KB)\n";
    }
    {
        std::ifstream f("/data/local/tmp/bert_codebook_z8_8bit.bin",
                        std::ios::binary);
        if(!f){std::cerr<<"Cannot open Z8 8-bit codebook\n";return 1;}
        f.read(reinterpret_cast<char*>(cb8z),sizeof(cb8z));
        std::cout<<"Z8 8-bit codebook loaded (8 KB)\n\n";
    }

    // Large activation vector for full-pass cycling
    const int MAXDIM = 1024+32;
    std::vector<float> x(MAXDIM);
    for(int i=0;i<MAXDIM;++i) x[i]=std::sin(i*0.1f);

    long   rss_load=0;
    double lcs=0;    // layer checksum

    // ── 1. BERT-Base FP32 ────────────────────────────────────
    // std::cout<<"══════ 1. BERT-Base FP32 ══════\n";
    // { std::vector<block_fp32> m;
    //   if(!stream_load("/data/local/tmp/bert_full_fp32.bin",m,"BERT-Base FP32"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},5);
    //   double lu=bench_layer(m,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},768,768,lcs);
    //   print_result("BERT-Base FP32",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,768,768,32,-1);
    // }

    // // ── 2. BERT-Base E8 16-bit ───────────────────────────────
    // std::cout<<"══════ 2. BERT-Base E8 16-bit ══════\n";
    // { std::vector<block_e8_16> m;
    //   if(!stream_load("/data/local/tmp/bert_full_e8_16bit.bin",m,"E8-16bit"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_e8_16&b,const float*x){return dot_e8_16(b,x);},5);
    //   double lu=bench_layer(m,[](const block_e8_16&b,const float*x){return dot_e8_16(b,x);},768,768,lcs);
    //   print_result("BERT-Base E8 16-bit",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,768,768,3.0,6.79);
    // }

    // ── 1. BERT-Base + E8 ────────────────────────────────────
    std::cout<<"══════ 1. BERT-Base + E8 ══════\n";
    { std::vector<block_e8_8> m;
      if(!stream_load("/data/local/tmp/bert_base_e8_quant.bin", m, "BERT-Base E8")) return 1;
      rss_load=get_rss_kb();
      auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},5);
      double lu=bench_layer(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},768,768,lcs);
      print_result("BERT-Base + E8 (ReCon QAT)",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,768,768,2.0,148.27);
    }

    // ── 2. BERT-Base + Z8 ────────────────────────────────────
    std::cout<<"══════ 2. BERT-Base + Z8 ══════\n";
    { std::vector<block_e8_8> m;
      if(!stream_load("/data/local/tmp/bert_base_z8_quant.bin", m, "BERT-Base Z8")) return 1;
      rss_load=get_rss_kb();
      auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_z8_8(b,x);},5);
      double lu=bench_layer(m,x,[](const block_e8_8&b,const float*x){return dot_z8_8(b,x);},768,768,lcs);
      print_result("BERT-Base + Z8 (ReCon QAT)",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,768,768,2.0,150.66);
    }

    // ── 3. TinyBERT + E8 ─────────────────────────────────────
    std::cout<<"══════ 3. TinyBERT + E8 ══════\n";
    { std::vector<block_e8_8> m;
      if(!stream_load("/data/local/tmp/tinybert_e8_quant.bin", m, "TinyBERT E8")) return 1;
      rss_load=get_rss_kb();
      auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},5);
      double lu=bench_layer(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},312,312,lcs);
      print_result("TinyBERT + E8 (ReCon QAT)",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,2.0,149.70);
    }

    // ── 4. TinyBERT + Z8 ─────────────────────────────────────
    std::cout<<"══════ 4. TinyBERT + Z8 ══════\n";
    { std::vector<block_e8_8> m;
      if(!stream_load("/data/local/tmp/tinybert_z8_quant.bin", m, "TinyBERT Z8")) return 1;
      rss_load=get_rss_kb();
      auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_z8_8(b,x);},5);
      double lu=bench_layer(m,x,[](const block_e8_8&b,const float*x){return dot_z8_8(b,x);},312,312,lcs);
      print_result("TinyBERT + Z8 (ReCon QAT)",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,2.0,152.07);
    }

    // // ── 4. DistilBERT FP32 ───────────────────────────────────
    // std::cout<<"══════ 4. DistilBERT FP32 ══════\n";
    // { std::vector<block_fp32> m;
    //   if(!stream_load("/data/local/tmp/distilbert_full_fp32.bin",m,"DistilBERT FP32"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},5);
    //   double lu=bench_layer(m,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},768,768,lcs);
    //   print_result("DistilBERT FP32",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,768,768,32,-1);
    // }

    // // ── 5. MobileBERT FP32 ───────────────────────────────────
    // std::cout<<"══════ 5. MobileBERT FP32 ══════\n";
    // { std::vector<block_fp32> m;
    //   if(!stream_load("/data/local/tmp/mobilebert_full_fp32.bin",m,"MobileBERT FP32"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},20);
    //   double lu=bench_layer(m,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},512,128,lcs);
    //   print_result("MobileBERT FP32",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,512,128,32,-1);
    // }

    // // ── 6. TinyBERT FP32 ─────────────────────────────────────
    // std::cout<<"══════ 6. TinyBERT (4L) FP32 ══════\n";
    // { std::vector<block_fp32> m;
    //   if(!stream_load("/data/local/tmp/tinybert_full_fp32.bin",m,"TinyBERT FP32"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},20);
    //   double lu=bench_layer(m,[](const block_fp32&b,const float*x){return dot_fp32(b,x);},312,312,lcs);
    //   print_result("TinyBERT (4L) FP32",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,32,-1);
    // }

    // ── 7. TinyBERT E8 16-bit ────────────────────────────────
    // NEW in v2 — mirrors BERT-Base's E8 16-bit section exactly.
    // SNR (6.85 dB) taken from export_all_models_v2.py Python run.
    // Layer shape kept at 312x312 to match TinyBERT's FP32 section above.
    // std::cout<<"══════ 7. TinyBERT (4L) E8 16-bit ══════\n";
    // { std::vector<block_e8_16> m;
    //   if(!stream_load("/data/local/tmp/tinybert_full_e8_16bit.bin",m,"TinyBERT E8-16bit"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_e8_16&b,const float*x){return dot_e8_16(b,x);},20);
    //   double lu=bench_layer(m,[](const block_e8_16&b,const float*x){return dot_e8_16(b,x);},312,312,lcs);
    //   print_result("TinyBERT (4L) E8 16-bit",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,3.0,6.85);
    // }

    // ── 8. TinyBERT E8 8-bit ─────────────────────────────────
    // NEW in v2 — mirrors BERT-Base's E8 8-bit section exactly.
    // SNR (3.87 dB) taken from export_all_models_v2.py Python run.
    // std::cout<<"══════ 8. TinyBERT (4L) E8 8-bit ══════\n";
    // { std::vector<block_e8_8> m;
    //   if(!stream_load("/data/local/tmp/tinybert_full_e8_8bit.bin",m,"TinyBERT E8-8bit"))return 1;
    //   rss_load=get_rss_kb();
    //   auto fp=bench_fullpass(m,x,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},20);
    //   double lu=bench_layer(m,[](const block_e8_8&b,const float*x){return dot_e8_8(b,x);},312,312,lcs);
    //   print_result("TinyBERT (4L) E8 8-bit",rss_load,fp.rss,fp.ms,lu,fp.cpu,fp.checksum,lcs,312,312,2.0,3.87);
    // }

    std::cout<<"Final RSS: "<<get_rss_kb()/1024<<" MB\n";
    return 0;
}