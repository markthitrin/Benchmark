#include <iostream>
#include <fstream>
#include <string>

long long format_number(const std::string& input) {
    double return_value;
    int divisor = 1;
    int q;
    for(q = 0;q < input.size();q++) {
        if(input[q] >= '0' && input[q] <= '9') {
            return_value = return_value * 10 + (input[q] - '0');
        }
        else {
            break;
        }
    }
    if(q < input.size() && input[q] == '.') {
        ++q;
        for(;q < input.size();q++) {
            divisor *= 10;
            if(input[q] >= '0' && input[q] <= '9') {
                return_value = return_value + double(input[q] - '0') / divisor;
            }
            else break;
        }
    }
    if(q < input.size()) {
        switch(input[q]) {
            case 'T': return_value *= 1000000000000; break;
            case 'G': return_value *= 1000000000; break;
            case 'M': return_value *= 1000000; break;
            case 'k': return_value *= 1000; break;
        }
    }
    return return_value;
}

int main(int argc, char* argv[]) {
    bool number_format = false;
    if(argc < 2) {
        std::cerr << "Please provide file name";
        return -1;
    }
    if(argc < 3) {
        std::cerr << "Please procide prefix";
        return -1;
    }
    if(argc >= 4) {
        if(std::string(argv[3]) == "-n") {
            number_format = true;
        }
    }
    std::string file_name(argv[1]);
    std::string prefix(argv[2]);
    std::ifstream file(file_name);
    std::string input;
    while(!file.eof()) {
        file >> input;
        bool match = true;
        for(int q = 0;q < prefix.size();q++) {
            if(input[q] != prefix[q]) {
                match = false;
                break;
            }
        }
        if(match) {
            if(number_format)
                std::cout << format_number(input.substr(prefix.size(),input.size())) << std::endl;
            else 
                std::cout << input.substr(prefix.size(),input.size()) << std::endl;
        }
    }
    return 0;
}