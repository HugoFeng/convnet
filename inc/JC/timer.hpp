#ifndef JC_TIMER_H__
#define JC_TIMER_H__

#include <ostream>

#ifdef __linux__
#include <sys/time.h>
#else
#include <windows.h>
#endif

namespace jc {

class Timer 
{
#ifdef __linux__
    timeval start_;
    timeval stop_;

 public:

    void start() {
        gettimeofday(&start_, NULL);
    }

    void stop() {
        gettimeofday(&stop_, NULL);
    }

    unsigned long getTime() const {
        unsigned long elapsed_time;
        elapsed_time  = 1000 * 1000 * (stop_.tv_sec - start_.tv_sec);
        elapsed_time += (stop_.tv_usec - start_.tv_usec);
        return elapsed_time;
    }
#else
    LARGE_INTEGER   start_;
    LARGE_INTEGER   stop_;
    LARGE_INTEGER   frequency_;

public:
    Timer() {
        start_.QuadPart = 0;
        stop_.QuadPart  = 0;
        QueryPerformanceFrequency(&frequency_);
    }

    void start() {
        QueryPerformanceCounter(&start_);
    }

    void stop() {
        QueryPerformanceCounter(&stop_);
    }

    unsigned long getTime() const {
        float difference;
        float ticksPerSecond;

        difference = (float) stop_.QuadPart - start_.QuadPart;
        ticksPerSecond = (float) frequency_.QuadPart;
        return static_cast<unsigned long>(1000000 * difference / ticksPerSecond);
    }
#endif
};

std::ostream& operator<<(std::ostream& oss, const Timer& t) {
    unsigned long factors[] = { 60000000, 1000000, 1000 };
    char     *time_scales[] = {      "m",     "s", "ms" };

    unsigned long rr = t.get_time();
    for (int i = 0; i < 3; ++i) {
        oss << rr / factors[i] << time_scales[i] << " ";
        rr %= factors[i];
    }
    oss << rr << "us";

    return oss;
}


}
#endif
