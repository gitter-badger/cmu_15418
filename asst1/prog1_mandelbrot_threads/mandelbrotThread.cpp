#include <stdio.h>
#include <pthread.h>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
void* workerThreadStart(void* threadArgs) {

    double start_time = CycleTimer::currentSeconds();
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);

    printf("Hello world from thread %d\n", args->threadId);

    // int part_size = (args->height + args->numThreads - 1) / args->numThreads;
    int i = args->threadId;
    // int end_row = 0;

    // if (i == (args->numThreads - 1))
    // {
    //     end_row = args->height - part_size * i;
    // }
    // else
    // {
    //     end_row = part_size;
    // }

    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height,
                     i, args->numThreads, args->maxIterations, args->output);
    double end_time = CycleTimer::currentSeconds();
    printf("Time in thread %d is %f\n", args->threadId, end_time - start_time);
    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        args[i].threadId = i;
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].numThreads = numThreads;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}
