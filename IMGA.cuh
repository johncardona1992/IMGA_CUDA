/*
  Island model genetic algorithm apply to optimize near real time agents schedules for Contact centers
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>

// includes, project

#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace std;
namespace cg = cooperative_groups;

// constant memory
__device__ __constant__ int const_numAgents;
__device__ __constant__ int const_numSchedules;
__device__ __constant__ int const_numPeriods;
__device__ __constant__ int *const_arrASchCount;
__device__ __constant__ int *const_arrAScanSchCount;
__device__ __constant__ int const_lenArrL;
__device__ __constant__ int *const_arrL;
__device__ __constant__ int *const_arrN;
// ----------------- Genetic variables ------------------
// The size of sub_population (number of chromosomes), should be power of 2
#define SUB_POPULATION_SIZE 32 
// number of threads to cooperate on each individual, should be power of 2
#define THREADS_PER_INDIVIDUAL 16 
// number of islands per SM
#define NUM_ISLANDS_PER_SM 3 
// number of threads per island
#define THREADS_PER_BLOCK  SUB_POPULATION_SIZE * THREADS_PER_INDIVIDUAL
// number of islands per grid
#define BLOCKS_PER_GRID  NUM_ISLANDS_PER_SM * 28
// The number of survivors in each epoch.
#define NUM_SURVIVORS 40
// The number of elites in each epoch. They are copied directly into a new generation.
#define NUM_ELITES 10
// The probability of mutation
#define MUTATION_RATE 0.05f  
// The maximal numbers of epoches.
#define MAX_EPOCHES 10000
// The number of individual to swap
#define SWAP_SIZE 10




__host__ void initGblVars(int &numAgents, int &numSchedules, int &numPeriods);
__host__ int countRows(string filePath);
__host__ void readCSV_A(int &numAgents, int *arrASchCount, int *arrAScanSchCount, int &lenArrL);
__host__ void readCSV_L(int *arrL, int &lenArrL);
__host__ void readCSV_E(int *read_arrE, int &lenArrE);
__host__ void readCSV_P(int *arrN, int &numPeriods);

//------------- Device-------------------
__global__ void kernel_IMGA(int *arrE, curandState *state);
__global__ void setup_curand(curandState *state);