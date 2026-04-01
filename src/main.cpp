#include "fledge.hpp"

int main(int argc, const char** argv) {
    FledgeSim sim;
    return nest::io::run(argc, argv, sim);
}
