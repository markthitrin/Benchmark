#include <iostream>
#include <stdlib.h>

int main() {
    int* p = (int*) malloc(sizeof(int) * 100);
    for(int q = 0;q < 100;q++) {
        std::cout << p[q] << " ";
    }
    return 0;
}