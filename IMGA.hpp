/*
  Island model genetic algorithm apply to optimize near real time agents schedules for Contact centers

  COMMAND LINE ARGUMENTS

  "--n=<N>"         :Specify the number of elements to reduce (default 33554432)
  "--threads=<N>"   :Specify the number of threads per block (default 128)
  "--maxblocks=<N>" :Specify the maximum number of thread blocks to launch
 (kernel 6 only, default 64)
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

#include <cuda_runtime.h>

const char *sSDKsample = "reductionMultiBlockCG";

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

__host__ void initGblVars(int &numAgents, int &numSchedules, int &numPeriods);
__host__ int countRows(string filePath);
__host__ void readCSV_A(int &numAgents, int *arrASchCount, int *arrAScanSchCount, int &lenArrL);
__host__ void readCSV_L(int *arrL, int &lenArrL);
__host__ void readCSV_E(int *read_arrE, int &lenArrE);
__host__ void readCSV_P(int *arrN, int &numPeriods);