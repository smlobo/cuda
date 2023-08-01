#include <cstdlib>
#include <ctime>
#include <cassert>

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
