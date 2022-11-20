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
#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace std;
namespace cg = cooperative_groups;

// ----------------- Genetic variables ------------------
// The size of sub_population (number of chromosomes), should be power of 2
#define SUB_POPULATION_SIZE 32
// number of threads to cooperate on each individual, should be power of 2
#define THREADS_PER_INDIVIDUAL 16
// number of tournaments per individual
#define NUM_TOURNAMENTS 8
// number of islands per SM
#define NUM_ISLANDS_PER_SM 2
// number of threads per island
#define THREADS_PER_BLOCK SUB_POPULATION_SIZE *THREADS_PER_INDIVIDUAL
// number of islands per grid
#define NUM_SM 28
// number of islands per grid
#define BLOCKS_PER_GRID NUM_ISLANDS_PER_SM *NUM_SM
#define MUTATION_RATE 0.02f
// The maximal numbers of epoches.
#define MAX_EPOCHES 30
// The number generations per Epoch
#define MAX_GENERATIONS 100
// The number of individual to be migrate from one island to another
#define MIGRATION_SIZE 1
// ----------------- Problem variables ------------------
// number of agents
#define AGENTS_SIZE 30
// number of schedules
#define SCHEDULES_SIZE 1775
// number of schedules
#define PERIODS_SIZE 96
// length of L dataset (feasible schedules)
#define L_SIZE 5114
// crossover point
#define CROSSPOINT 15
// Subpopulation shared memory
#define SUBPOPULATION_BYTES AGENTS_SIZE *SUB_POPULATION_SIZE

// constant memory
__device__ __constant__ int const_numSchedules;
__device__ __constant__ int const_numPeriods;
__device__ __constant__ int const_lenArrL;
__device__ __constant__ int const_arrASchCount[AGENTS_SIZE];
__device__ __constant__ int const_arrAScanSchCount[AGENTS_SIZE];
__device__ __constant__ int const_arrL[L_SIZE];
__device__ __constant__ int const_arrN[PERIODS_SIZE];

__host__ void initGblVars(int &numAgents, int &numSchedules, int &numPeriods);
__host__ void printSolution(int &numAgents, vector<string> &agentsIDS, vector<string> &schedulesIDS, int *global_solution);
__host__ int countRows(string filePath);
__host__ void readCSV_A(int &numAgents, int *arrASchCount, int *arrAScanSchCount, int &lenArrL, vector<string> &agentsIDS);
__host__ void readCSV_S(int &numSchedules, vector<string> &schedulesIDS);
__host__ void readCSV_L(int *arrL, int &lenArrL);
__host__ void readCSV_E(int *read_arrE, int &lenArrE);
__host__ void readCSV_P(int *arrN, int &numPeriods);

//------------- Device-------------------
__global__ void kernel_IMGA(int *arrE, curandState *state, int *emigrants, int *fitness_emigrants, int *global_solutions, int *islands_fitness, int *best_fitness);

template <int T>
__device__ void
initialize_population(curandState &localState, int *subPopulation, cg::thread_block_tile<T> tile_individual);

template <int T>
__device__ void
calculate_fitness(int *arrE, int *arrFitness, int *subPopulation, cg::thread_block_tile<T> tile_individual);

template <int T>
__device__ void
elitism(int *highlander, int *highlander_fitness, int *highlander_chromosome, int *subPopulation, int *arrFitness, cg::thread_block_tile<T> tile_individual);

template <int T>
__device__ void
migration_shared_to_global(curandState &localState, int *subPopulation, int *arrFitness, int *arrEmigrantID, int *fitness_emigrants, int *emigrants, int *arrWeakID, cg::thread_block_tile<T> tile_individual, cg::thread_block block);

template <int T>
__device__ void
migration_global_to_shared(int *subPopulation, int *arrFitness, int *neighbor, int *fitness_emigrants, int *emigrants, int *arrWeakID, cg::thread_block_tile<T> tile_individual, cg::thread_block block);

template <int T>
__device__ void
parent_selection(curandState &localState, int *arrParents, int *arrFitness, cg::thread_block block);

template <int T>
__device__ void
crossover( int *arrParents, int *subOffsprings, int *subPopulation, cg::thread_block_tile<T> tile_individual);

template <int T>
__device__ void
mutation(curandState &localState, int *subOffsprings, cg::thread_block_tile<T> tile_individual);

template <int T>
__device__ void
highlander_the_immortal(curandState &localState, int *subOffsprings, int *highlander_chromosome, cg::thread_block_tile<T> tile_individual);

