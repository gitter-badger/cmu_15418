#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void sqrtAvx(int N,
             float initialGuess,
             float values[],
             float output[])    
{
    float constants[] = {0.00001f, 1.0f, 0.f, 3.f, 0.5f};
    unsigned int o = 0Xffffffff;

    __m256 kThreshold = _mm256_broadcast_ss(constants);
    __m256 ones = _mm256_broadcast_ss(constants+1);
    __m256 zeros = _mm256_broadcast_ss(constants+2);
    __m256 threes = _mm256_broadcast_ss(constants+3);
    __m256 halfs = _mm256_broadcast_ss(constants+4);
    __m256 allones = _mm256_castsi256_ps(_mm256_set1_epi32(o));

    for (int i = 0; i < N; i += 8) 
    {       
        __m256 xs = _mm256_load_ps(values + i);

        __m256 guess = _mm256_set1_ps(initialGuess);

        __m256 error = _mm256_mul_ps(guess, guess);
        error = _mm256_mul_ps(error, xs);
        error = _mm256_sub_ps(error ,ones);

        __m256 minusError = _mm256_sub_ps(zeros, error);
        error = _mm256_max_ps(error, minusError);

        __m256 cmpRes = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ);
        int res = _mm256_testz_ps(cmpRes, allones);

        while (res == 0)
        {
            __m256 tmp1 = _mm256_mul_ps(xs, guess);
            tmp1 = _mm256_mul_ps(tmp1, guess);
            tmp1 = _mm256_mul_ps(tmp1, guess);
            guess = _mm256_mul_ps(guess, threes);
            guess = _mm256_sub_ps(guess, tmp1);
            guess = _mm256_mul_ps(guess, halfs);

            error = _mm256_mul_ps(guess, guess);
            error = _mm256_mul_ps(error, xs);
            error = _mm256_sub_ps(error ,ones);
            minusError = _mm256_sub_ps(zeros, error);
            error = _mm256_max_ps(error, minusError);

            cmpRes = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ);
            res = _mm256_testz_ps(cmpRes, allones);
        }

        guess = _mm256_mul_ps(xs, guess);
        _mm256_store_ps(output+i, guess);

    }
}

