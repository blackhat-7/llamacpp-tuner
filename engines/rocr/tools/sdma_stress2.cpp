// Like sdma_stress, but copies of mixed sizes and directions with cross-stream event waits, so the SDMA
// command sizes vary and ring wraps leave end-of-ring padding of every length (as in a real inference engine).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 4000000;
    const int streams = 4;
    const size_t maxb = 1 << 20;
    char *h = nullptr, *d = nullptr;
    if (hipHostMalloc((void**) &h, maxb * streams) != hipSuccess || hipMalloc((void**) &d, maxb * streams) != hipSuccess) return 2;
    hipStream_t s[streams];
    hipEvent_t ev[streams];
    for (int i = 0; i < streams; ++i) {
        hipStreamCreateWithFlags(&s[i], hipStreamNonBlocking);
        hipEventCreateWithFlags(&ev[i], hipEventDisableTiming);
    }
    unsigned x = 12345;
    for (long i = 0; i < iters; ++i) {
        x = x * 1103515245u + 12345u;
        const int k = (int) ((x >> 8) % streams);
        const size_t bytes = 4096u << ((x >> 12) % 9);           // 4 KB .. 1 MB
        if ((x >> 20) % 3 == 0) hipStreamWaitEvent(s[k], ev[(k + 1) % streams], 0);
        const bool h2d = (x >> 24) % 4 != 0;
        hipError_t e = h2d ? hipMemcpyAsync(d + k * maxb, h + k * maxb, bytes, hipMemcpyHostToDevice, s[k])
                           : hipMemcpyAsync(h + k * maxb, d + k * maxb, bytes, hipMemcpyDeviceToHost, s[k]);
        if (e != hipSuccess) { std::printf("copy failed at %ld\n", i); return 1; }
        hipEventRecord(ev[k], s[k]);
        if (i % 2048 == 2047) hipDeviceSynchronize();
        if (i % 500000 == 0) { std::printf("%ld copies\n", i); std::fflush(stdout); }
    }
    hipDeviceSynchronize();
    std::printf("done: %ld copies\n", iters);
    return 0;
}
