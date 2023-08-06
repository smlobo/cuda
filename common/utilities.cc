#include <cstdlib>
#include <ctime>
#include <cassert>
#include <string>

#include "utilities.h"

class PreMain {
public:
    PreMain() {
        // Random seed based on time
        srand(time(nullptr));
    }
};

PreMain premain;

// Random double between -1 and 1
double randomDouble() {
    double d = rand()/(double)(RAND_MAX/2);
    return d - 1.0;
}

// Random float between -1 and 1
float randomFloat() {
    float f = rand()/(float)(RAND_MAX/2);
    return f - 1.0;
}

// Random int between 0 (inclusive) and to (exclusive)
int randomInt(int to) {
    return randomInt(0, to);
}

// Random int between from (inclusive) and to (exclusive)
int randomInt(int from, int to) {
    assert(from < to);
    int diff = to - from;
    int r = rand() % diff;
    return r + from;
}

// Take in number of nanoseconds, and return a string with ms, us, ns
char timeStr[100];
std::string timeSuffixTable[] = {"ns", "us", "ms", "sec"};
char* nanoToString(long n) {
    int count = 0;
    int whole = n;
    int decimal = 0;
    while (whole > 1000 && count <= 3) {
        count++;
        decimal = whole%1000;
        whole /= 1000;
    }
    sprintf(timeStr, "%d.%d%s", whole, decimal, timeSuffixTable[count].c_str());
    return timeStr;
}
