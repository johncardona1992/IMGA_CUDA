#include "IMGA.cuh"

int main()
{
	cudaError_t err = cudaSuccess;
	//-------------- Problem variables ----------------
	// number of agents
	int numAgents = 0;
	// number of schedules
	int numSchedules = 0;
	// number of periods
	int numPeriods = 0;
	// agents IDs
	vector<string> agentsIDS;
	// Schedules IDs
	vector<string> schedulesIDS;
	// length of read_arrE
	int lenArrE = 0;
	// array of count of schedules per Agent
	int *arrASchCount = NULL;
	// array of cummulative count of schedules per Agent
	int *arrAScanSchCount = NULL;
	// length of arrAScanSchCount
	int lenArrL = 0;
	// array of schedules index per Agent => try it on constant memory
	int *arrL = NULL;
	// array of schedules index per Agent host version
	int *read_arrE = NULL;
	// array of schedules index per Agent device version
	int *arrE;
	// array of number of Agents required per period
	int *arrN = NULL;
	// device id
	int deviceId = cudaGetDevice(&deviceId);
	// curand state
	curandState *d_state;
	// vector of emigrants chromosomes
	int *emigrants;
	// vector of fitness from emigrants
	int *fitness_emigrants;
	// best island solution
	int *global_solution;
	// best island solution
	int *islands_fitness;
	// best fitness solution
	int *best_fitness;

	// ----------------- Genetic variables ------------------

	// initilize global variables
	initGblVars(numAgents, numSchedules, numPeriods);
	// initilize arrays
	arrASchCount = (int *)malloc(sizeof(int) * numAgents);
	arrAScanSchCount = (int *)malloc(sizeof(int) * numAgents);

	// read csv data A.csv
	readCSV_A(numAgents, arrASchCount, arrAScanSchCount, lenArrL, agentsIDS);
	// read csv data S.csv
	readCSV_S(numSchedules, schedulesIDS);
	// read csv data L.csv+
	arrL = (int *)malloc(sizeof(int) * lenArrL);
	readCSV_L(arrL, lenArrL);
	printf("len L: %i", lenArrL);
	// read csv data E.csv
	lenArrE = numSchedules * numPeriods;
	read_arrE = (int *)malloc(sizeof(int) * lenArrE);
	readCSV_E(read_arrE, lenArrE);
	// check schedule
	// for (int p = 0; p < numPeriods; p++)
	//	printf("period %i: %i\n", p+1, read_arrE[1774*numPeriods + p]);

	// read csv data P.csv
	arrN = (int *)malloc(sizeof(int) * numPeriods);
	readCSV_P(arrN, numPeriods);

	// unified memory for arrE
	err = cudaMallocManaged(&arrE, lenArrE * sizeof(int));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector arrE (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// hint to prioritize host transfer
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	// read data
	for (int e = 0; e < lenArrE; e++)
	{
		arrE[e] = read_arrE[e];
	}

	// curand memory allocation
	err = cudaMallocManaged(&d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_state (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// emigrants memory allocation
	err = cudaMallocManaged(&emigrants, BLOCKS_PER_GRID * MIGRATION_SIZE * AGENTS_SIZE * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector emigrants (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// fitness emigrants memory allocation
	err = cudaMallocManaged(&fitness_emigrants, BLOCKS_PER_GRID * MIGRATION_SIZE * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector fitness_emigrants (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// global solutions memory allocation
	err = cudaMallocManaged(&global_solution, AGENTS_SIZE * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector global_solution (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// global solutions memory allocation
	err = cudaMallocManaged(&islands_fitness, BLOCKS_PER_GRID * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector islands_fitness (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// best fitness solution memory allocation
	err = cudaMallocManaged(&best_fitness, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector best_fitness (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// launch init kernel
	cudaMemAdvise(d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState), cudaMemAdviseSetPreferredLocation, deviceId);
	setup_curand<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_state);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch setup_curand kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//------- allocate Device constant memory-----
	err = cudaMemcpyToSymbol(const_numSchedules, &numSchedules, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_numSchedules (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_numPeriods, &numPeriods, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_numPeriods (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_lenArrL, &lenArrL, sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_lenArrL (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//------- allocate Device constant arrays memory-----
	err = cudaMemcpyToSymbol(const_arrASchCount, arrASchCount, numAgents * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrASchCount (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrAScanSchCount, arrAScanSchCount, numAgents * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrAScanSchCount (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrL, arrL, lenArrL * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrL (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToSymbol(const_arrN, arrN, numPeriods * sizeof(int));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device constant const_arrN (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//-----hints------ unified memory
	// hint for read mostly global data
	cudaMemAdvise(arrE, lenArrE * sizeof(int), cudaMemAdviseSetReadMostly, deviceId);
	// prefetching from host to device
	cudaMemPrefetchAsync(arrE, lenArrE * sizeof(int), deviceId);
	cudaMemAdvise(d_state, BLOCKS_PER_GRID * THREADS_PER_BLOCK * sizeof(curandState), cudaMemAdviseSetReadMostly, deviceId);
	cudaMemAdvise(emigrants, BLOCKS_PER_GRID * MIGRATION_SIZE * AGENTS_SIZE * sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);
	cudaMemAdvise(fitness_emigrants, BLOCKS_PER_GRID * MIGRATION_SIZE * sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);
	cudaMemAdvise(global_solution, AGENTS_SIZE * sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);
	cudaMemAdvise(islands_fitness, BLOCKS_PER_GRID * sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);
	cudaMemAdvise(best_fitness, sizeof(int), cudaMemAdviseSetPreferredLocation, deviceId);

	//------------------- calculate theorical occupancy -------------------
	int dev = 0;
	int supportsCoopLaunch = 0;
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
	if (!supportsCoopLaunch)
	{
		printf(
			"\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
			"Waiving the run\n",
			dev);
		exit(0);
	}

	/// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
	int numBlocksPerSm = 0;
	// Number of threads my_kernel will be launched with
	int numThreads = THREADS_PER_BLOCK;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel_IMGA, numThreads, 0);
	// launch
	void *kernelArgs[] = {
		(void *)&arrE,
		(void *)&d_state,
		(void *)&emigrants,
		(void *)&fitness_emigrants,
		(void *)&global_solution,
		(void *)&islands_fitness,
		(void *)&best_fitness,
	};
	dim3 dimBlock(numThreads, 1, 1);
	// dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
	dim3 dimGrid(BLOCKS_PER_GRID, 1, 1);
	// execute kernel
	printf("\nblocks: %i", dimGrid.x);
	printf("\nthreads: %i", numThreads);
	cudaLaunchCooperativeKernel((void *)kernel_IMGA, dimGrid, dimBlock, kernelArgs);
	// kernel_IMGA<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(arrE, d_state, emigrants, fitness_emigrants);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel_IMGA kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// prefetch results from device to host
	cudaMemPrefetchAsync(global_solution, AGENTS_SIZE * sizeof(int), cudaCpuDeviceId);
	cudaMemPrefetchAsync(best_fitness, sizeof(int), cudaCpuDeviceId);
	printf("\n---------------\n");
	printf("\nbest fitness: %i\n", best_fitness[0]);
	for (int i = 0; i < AGENTS_SIZE; i++)
	{
		// printf("\nagent %i: sch %i", i, global_solution[i]);
		std::cout << "agent: " << agentsIDS[i] << " sch: "<< schedulesIDS[global_solution[i]]<< std::endl;
	}
	// export solution to csv
	printSolution(numAgents, agentsIDS, schedulesIDS, global_solution);
	// deallocate dynamic memory
	free(arrASchCount);
	free(arrAScanSchCount);
	free(arrL);
	free(read_arrE);
	free(arrN);
	cudaFree(arrE);
	cudaFree(d_state);
	cudaFree(emigrants);
	cudaFree(fitness_emigrants);
	cudaFree(global_solution);
	cudaFree(islands_fitness);
	cudaFree(best_fitness);

	// reset device
	cudaDeviceReset();
}

__host__ void printSolution(int &numAgents, vector<string> &agentsIDS, vector<string> &schedulesIDS, int *global_solution)
{
	char filename[] = "../MILP/result.csv";
	fstream appendFileToWorkWith;

	appendFileToWorkWith.open(filename, std::fstream::in | std::fstream::out | std::ofstream::trunc | std::fstream::app);

	// If file does not exist, Create new file
	if (!appendFileToWorkWith)
	{
		// cout << "Cannot open file, file does not exist. Creating new file..";

		appendFileToWorkWith.open(filename, fstream::in | fstream::out | fstream::trunc);
		appendFileToWorkWith << "A_ID,S_ID,x_value\n";
		for (int i = 0; i < numAgents; i++)
		{
			appendFileToWorkWith << agentsIDS[i] << "," << schedulesIDS[global_solution[i]] << ","
								 << "1"
								 << "\n";
		}

		appendFileToWorkWith.close();
	}
	else
	{ // use existing file

		appendFileToWorkWith << "A_ID,S_ID,x_value\n";
		for (int i = 0; i < numAgents; i++)
		{
			appendFileToWorkWith << agentsIDS[i] << "," << schedulesIDS[global_solution[i]] << ","
								 << "1"
								 << "\n";
		}
		appendFileToWorkWith.close();
	}
}

__host__ void initGblVars(int &numAgents, int &numSchedules, int &numPeriods)
{
	// initilize gbl variables
	numAgents = countRows("../MILP/A.csv") - 1;
	printf("\nnum agents: %i", numAgents);
	numSchedules = countRows("../MILP/S.csv") - 1;
	printf("\nnum schedules: %i", numSchedules);
	numPeriods = countRows("../MILP/P.csv") - 1;
	printf("\nnum periods: %i\n", numPeriods);
}

__host__ int countRows(string filePath)
{
	// count rows in a csv file
	int rows = 0;
	ifstream file;
	file.open(filePath);
	string line;
	while (getline(file, line))
		rows++;
	file.close();
	return rows;
}

__host__ void readCSV_A(int &numAgents, int *arrASchCount, int *arrAScanSchCount, int &lenArrL, vector<string> &agentsIDS)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/A.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < numAgents) // while the end of file is NOT reached
		{
			arrAScanSchCount[i] = lenArrL;
			getline(coeff, col, ',');
			agentsIDS.push_back(col);
			getline(coeff, col, ',');
			arrASchCount[i] = stoi(col);
			lenArrL += arrASchCount[i];
			getline(coeff, col, '\n');
			// printf("\nagent %i: %i\n", i, arrASchCount[i]);
			// printf("agent scan %i: %i\n", i, arrAScanSchCount[i]);
			// printf("agent total scan: %i\n", lenArrL);
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_S(int &numSchedules, vector<string> &schedulesIDS)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/S.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < numSchedules) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			schedulesIDS.push_back(col);
			getline(coeff, col, '\n');
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_L(int *arrL, int &lenArrL)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/L.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < lenArrL) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			arrL[i] = stoi(col);
			getline(coeff, col, '\n');
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_E(int *read_arrE, int &lenArrE)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/E.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < lenArrE) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			getline(coeff, col, ',');
			read_arrE[i] = stoi(col);
			// printf("\nE %i: %i", i, read_arrE[i]);
			getline(coeff, col, '\n');
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__host__ void readCSV_P(int *arrN, int &numPeriods)
{
	// read data A.csv
	string col; // variables from file are here
	// input filename
	string file = "../MILP/P.csv";

	// number of lines
	int i = 0;

	ifstream coeff(file); // opening the file.
	if (coeff.is_open())  // if the file is open
	{
		// ignore first line
		string line;
		getline(coeff, line);

		while (i < numPeriods) // while the end of file is NOT reached
		{
			getline(coeff, col, ',');
			getline(coeff, col, '\n');
			arrN[i] = stoi(col);
			// printf("\nperiod %i: %i", i, arrN[i]);
			i += 1; // increment number of lines
		}
		coeff.close(); // closing the file
	}
	else
		cout << "Unable to open file"; // if the file is not open output
}

__global__ void setup_curand(curandState *state)
{
	// each island has a different seed, and each individual has a different sequence
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(blockIdx.x, threadIdx.x, 0, &state[tid]);
}

__global__ void kernel_IMGA(int *arrE, curandState *state, int *emigrants, int *fitness_emigrants, int *global_solution, int *islands_fitness, int *best_fitness)
{
	// initlize cooperative groups
	// the grid represents the global population
	cg::grid_group grid = cooperative_groups::this_grid();
	// each block represents an island population
	cg::thread_block block = cg::this_thread_block();
	// each tile represents an individual
	cg::thread_block_tile<THREADS_PER_INDIVIDUAL> tile_individual = cg::tiled_partition<THREADS_PER_INDIVIDUAL>(block);
	// each tile represents a tournament
	cg::thread_block_tile<NUM_TOURNAMENTS> tile_tournament = cg::tiled_partition<NUM_TOURNAMENTS>(tile_individual);

	cg::sync(tile_individual);
	// shared memory
	// island population of parents
	int __shared__ subPopulation_source[SUBPOPULATION_BYTES];
	// island population of children
	int __shared__ subOffsprings_source[SUBPOPULATION_BYTES];
	// fitnes vector for each island
	int __shared__ arrFitness
		[SUB_POPULATION_SIZE];
	// Parent ID vector for each island
	int __shared__ arrParents[SUB_POPULATION_SIZE];
	// Highlander ID for each island
	int __shared__ highlander[1];
	// Highlander fitness for each island
	int __shared__ highlander_fitness[1];
	// Highlander Chromosome
	int __shared__ highlander_chromosome[AGENTS_SIZE];
	// Emigrant ID vector for each island
	int __shared__ arrEmigrantID[MIGRATION_SIZE];
	// weak ID vector for each island
	int __shared__ arrWeakID[MIGRATION_SIZE];

	int __shared__ *subPopulation;
	int __shared__ *subOffsprings;

	int __shared__ neighbor[1];

	if (block.thread_index().x == 0)
	{
		subPopulation = &subPopulation_source[0];
		subOffsprings = &subOffsprings_source[0];
		neighbor[0] = (block.group_index().x + 1);
		if (neighbor[0] >= BLOCKS_PER_GRID)
		{
			neighbor[0] = 0;
		}
	}
	cg::sync(block);
	// ------------------- Initilize sub-populations ------------------------------
	// Copy random number state to local memory (registers) for efficiency
	curandState localState = state[grid.thread_rank()];

	for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
	{
		float random_value = curand_uniform(&localState) * const_arrASchCount[a];
		int random_pos = (int)truncf(random_value);

		subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = const_arrL[const_arrAScanSchCount[a] + random_pos];
		//--------Validate initial Population
		// int idb = blockIdx.x;
		// if (idb == 0)
		// {
		// 	printf("\nblock: %i, individual: %i, agent: %i, feasible: %i, startID: %i, random: %i, scheduleID: %i", block.group_index().x, tile_individual.meta_group_rank(), a, const_arrASchCount[a], const_arrAScanSchCount[a], random_pos, const_arrL[const_arrAScanSchCount[a] + random_pos]);
		// }
	}

	//---------------------start epoch-----------------------------
	for (int epoch = 0; epoch < MAX_EPOCHES; epoch++)
	{
		//---------------------start generation------------------------
		for (int generation = 0; generation < MAX_GENERATIONS; generation++)
		{
			// initilize fitness and parent vectors for each generation
			if (tile_individual.thread_rank() == 0)
			{
				arrFitness[tile_individual.meta_group_rank()] = 0;
				arrParents[tile_individual.meta_group_rank()] = 0;
			}
			// syncronize all threads from the same island
			cg::sync(block);
			//------------------ calculate fitness--------------------------
			// local memory
			int objective = 0;
			int active_agents = 0;
			for (int p = 0; p < const_numPeriods; p++)
			{
				active_agents = 0;
				// grid stride loops along agents dimension
				for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
				{
					int idSchedule = subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a];
					active_agents += arrE[idSchedule * const_numPeriods + p];
					//  print schedules and set covering for period p
					// if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 0)
					// {
					// 	printf("\nagent %i, schedule %i: %i", a, idSchedule, arrE[idSchedule * const_numPeriods + p]);
					// }
				}

				cg::sync(tile_individual);
				// reduce cooperative function
				active_agents = cg::reduce(tile_individual, active_agents, cg::plus<int>());

				// calculate objective funtion
				if (tile_individual.thread_rank() == 0)
				{
					// objective could be moved to shared memory
					objective = objective + max(const_arrN[p] - active_agents, 0);
					// print fo along the periods
					//  if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 2)
					//  {
					//  	printf("\nPeriodo %i, Activos: %i, requeridos: %i, fo: %i", p, active_agents, const_arrN[p], objective);
					//  }
					// roulette selection
					// atomicAdd(&totalFitness[block.group_index().x],objective);
				}
			}
			if (tile_individual.thread_rank() == 0)
			{
				arrFitness[tile_individual.meta_group_rank()] = objective;
			}
			// print fitness vector for island 0
			// if (block.group_index().x == 27 && tile_individual.thread_rank() == 0)
			// {
			// 	printf("\nindividual %i: %i faltantes, %i", tile_individual.meta_group_rank(), objective,arrFitness[tile_individual.meta_group_rank()]);
			// }

			cg::sync(block);
			//-----------------------Elitism ------------------------------
			if (tile_individual.meta_group_rank() == 0)
			{
				int fitness = 1000000;
				for (int c = tile_individual.thread_rank(); c < SUB_POPULATION_SIZE; c += tile_individual.size())
				{
					fitness = min(arrFitness[c], fitness);
				}
				fitness = cg::reduce(tile_individual, fitness, cg::less<int>());
				for (int c = tile_individual.thread_rank(); c < SUB_POPULATION_SIZE; c += tile_individual.size())
				{
					if (tile_individual.shfl(fitness, 0) == arrFitness[c])
					{
						atomicExch(&highlander[0], c);
						atomicExch(&highlander_fitness[0], arrFitness[c]);
					}
				}
				cg::sync(tile_individual);
				// make a copy of highlander chromosome
				for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
				{
					highlander_chromosome[a] = subPopulation[highlander[0] * AGENTS_SIZE + a];
				}
			}
			cg::sync(block);
			// validate highlander
			// if (block.thread_index().x == 0 && block.group_index().x == 0)
			// {
			// 	for (int c = 0; c < SUB_POPULATION_SIZE; c++)
			// 	{
			// 		printf("\nindividual %i: %i", c, arrFitness[c]);
			// 	}
			// 	printf("\nbest %i",highlander[0]);
			// }
			//---------------- Migration -------------------------------
			if (generation == (MAX_GENERATIONS - 1))
			{
				if (tile_individual.meta_group_rank() < MIGRATION_SIZE)
				{
					// select emigrants
					int emigrantID = 0;
					float random_value = curand_uniform(&localState) * SUB_POPULATION_SIZE;
					emigrantID = (int)truncf(random_value);
					objective = arrFitness[emigrantID];
					cg::sync(tile_individual);
					objective = cg::reduce(tile_individual, objective, cg::less<int>());
					// fill up emigrants ID and respective fitness
					if (tile_individual.shfl(objective, 0) == arrFitness[emigrantID])
					{
						// shared memory - winners
						atomicExch(&arrEmigrantID[tile_individual.meta_group_rank()], emigrantID);
						// global memory - fitness of the winners
						atomicExch(&fitness_emigrants[block.group_index().x * MIGRATION_SIZE + tile_individual.meta_group_rank()], objective);
					}
					cg::sync(tile_individual);
					// select weak emigrants
					int weakID = 0;
					random_value = curand_uniform(&localState) * SUB_POPULATION_SIZE;
					weakID = (int)truncf(random_value);
					objective = arrFitness[weakID];
					cg::sync(tile_individual);
					objective = cg::reduce(tile_individual, objective, cg::greater<int>());
					// fill up weaks ID
					if (tile_individual.shfl(objective, 0) == arrFitness[weakID])
					{
						atomicExch(&arrWeakID[tile_individual.meta_group_rank()], weakID);
					}
					// copy emigrant chromosome from shared to global memory
					for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
					{
						emigrants[block.group_index().x * MIGRATION_SIZE * AGENTS_SIZE + tile_individual.meta_group_rank() * AGENTS_SIZE + a] = subPopulation[arrEmigrantID[tile_individual.meta_group_rank()] * AGENTS_SIZE + a];
					}
				}
				// a grid sync is necessary before starting migration from global to shared memory
				cg::sync(grid);
				if (tile_individual.meta_group_rank() < MIGRATION_SIZE)
				{
					for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
					{
						subPopulation[arrWeakID[tile_individual.meta_group_rank()] * AGENTS_SIZE + a] = emigrants[neighbor[0] * MIGRATION_SIZE * AGENTS_SIZE + tile_individual.meta_group_rank() * AGENTS_SIZE + a];
					}
					if (tile_individual.thread_rank() == 0)
					{
						arrFitness[arrWeakID[tile_individual.meta_group_rank()]] = fitness_emigrants[neighbor[0] * MIGRATION_SIZE + tile_individual.meta_group_rank()];
					}
				}
				// validate copy
				// if (block.group_index().x == 27 && block.thread_index().x == 0)
				// {
				// 	int k = 1;
				// 	for (int a = 0; a < AGENTS_SIZE; a++)
				// 	{
				// 		printf("\nagente %i: %i, %i", a, subPopulation[arrWeakID[k] * AGENTS_SIZE + a], emigrants[(neighbor[0] * MIGRATION_SIZE * AGENTS_SIZE) + (k * AGENTS_SIZE) + a]);
				// 	}
				// 	printf("\nneighbor %i", neighbor[0]);
				// 	printf("\nweakID %i", arrWeakID[k]);
				// 	printf("\nfitness %i, %i", arrFitness[arrWeakID[k]], fitness_emigrants[neighbor[0] * MIGRATION_SIZE + k]);
				// }
			}
			cg::sync(block);
			//---------end migration----------//
			//---------------- tournament selection --------------------
			if (tile_tournament.meta_group_rank() == 0)
			{

				int parentID = 0;
				float random_value = curand_uniform(&localState) * SUB_POPULATION_SIZE;
				parentID = (int)truncf(random_value);
				objective = arrFitness[parentID];
				cg::sync(tile_tournament);
				// get winner fitness by reduce cooperative function
				//  if (block.group_index().x == 27 && tile_tournament.meta_group_rank() == 0)
				//  {
				//   	printf("\nindividual %i: %i faltantes", tile_tournament.thread_rank(), objective);
				//  }
				objective = cg::reduce(tile_tournament, objective, cg::less<int>());
				// if (block.group_index().x == 0 && tile_tournament.meta_group_rank() == 0)
				// {
				//   	printf("\nindividual %i, parentID %i, fitness %i, minimo %i", tile_tournament.thread_rank(), parentID, arrFitness[parentID], tile_tournament.shfl(objective,0));
				// }
				// deterministic using atomic operators
				if (tile_tournament.shfl(objective, 0) == arrFitness[parentID])
				{
					atomicExch(&arrParents[tile_individual.meta_group_rank()], parentID);
				}
			}
			cg::sync(block);
			// if (block.group_index().x == 0 && tile_individual.meta_group_rank() == 0)
			// {
			//   	printf("\nparentID selected %i", arrParents[tile_individual.meta_group_rank()]);
			// }
			//-----------------------Crossover-------------------------
			// generate crossover point
			// first half from parent
			for (int a = tile_individual.thread_rank(); a < CROSSPOINT; a += tile_individual.size())
			{
				subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = subPopulation[arrParents[tile_individual.meta_group_rank()] * AGENTS_SIZE + a];
			}
			// second half from individual
			for (int a = CROSSPOINT + tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
			{
				subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = subPopulation[tile_individual.meta_group_rank() * AGENTS_SIZE + a];
			}
			cg::sync(block);
			// print crossover of one individual
			//  if (block.group_index().x == 0 && block.thread_rank() == 0)
			//  {
			//  	for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a++)
			//  	{
			//  		printf("\ngene %i, parent1: %i, parent2: %i, offspring: %i", a, subPopulation[1 * AGENTS_SIZE + a], subPopulation[arrParents[1] * AGENTS_SIZE + a], subOffsprings[1 * AGENTS_SIZE + a]);
			//  	}
			//  }
			//-----------------------Mutation-------------------------------
			for (int a = tile_individual.thread_rank(); a < AGENTS_SIZE; a += tile_individual.size())
			{
				float random_v = curand_uniform(&localState);
				if (random_v < MUTATION_RATE)
				{
					random_v = curand_uniform(&localState) * const_arrASchCount[a];
					int random_pos = (int)truncf(random_v);
					subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a] = const_arrL[const_arrAScanSchCount[a] + random_pos];
					// if (block.group_index().x == 0)
					// 	printf("\nagent %i: idschedule %i", a, subOffsprings[tile_individual.meta_group_rank() * AGENTS_SIZE + a]);
				}
			}
			cg::sync(block);
			//-------------------------- replace old population -------------------
			// replace random child by highlander
			if (tile_individual.meta_group_rank() == 0)
			{
				int childID = 0;
				int random_value = curand_uniform(&localState) * SUB_POPULATION_SIZE;
				childID = (int)truncf(random_value);
				// not necesary to use sync(tile_individual)
				for (int c = tile_individual.thread_rank(); c < SUB_POPULATION_SIZE; c += tile_individual.size())
				{
					atomicExch(&subOffsprings[tile_individual.shfl(childID, 0) * AGENTS_SIZE + c], highlander_chromosome[c]);
				}
			}
			cg::sync(block);
			// validate replacement highlander
			// if (block.thread_index().x == 0 && block.group_index().x == 0)
			// {
			// 	for (int c = 0; c < AGENTS_SIZE; c++)
			// 	{
			// 		printf("\nposition %i: %i, %i", c, subPopulation[highlander[0] * AGENTS_SIZE + c], subOffsprings[0 * AGENTS_SIZE + c]);
			// 	}
			// }
			// children replace parents
			if (block.thread_index().x == 0)
			{
				int *p = &subPopulation[0];
				subPopulation = subOffsprings;
				subOffsprings = p;
			}

			// if (block.thread_index().x == 0)
			// {
			// 	printf("\nepoch %i, block %i, fitness %i", epoch, block.group_index().x, highlander_fitness[0]);
			// }
			// cg::sync(block);
		}
		//---------------------- end epoch ------------------------
	}
	cg::sync(block);
	// extract best solution
	if (block.thread_index().x == 0)
	{
		islands_fitness[block.group_index().x] = highlander_fitness[0];
	}
	cg::sync(grid);

	if (grid.thread_rank() == 0)
	{
		best_fitness[0] = 100000;
		for (int i = 0; i < BLOCKS_PER_GRID; i++)
		{
			best_fitness[0] = min(best_fitness[0], islands_fitness[i]);
			if (best_fitness[0] == islands_fitness[i])
			{
				emigrants[0] = i;
			}
		}
		// printf("\nbest %i, pos %i", best_fitness[0], emigrants[0]);
	}
	cg::sync(grid);
	if (block.group_index().x == emigrants[0] && block.thread_index().x == 0)
	{
		for (int a = 0; a < AGENTS_SIZE; a++)
		{
			global_solution[a] = highlander_chromosome[a];
			// printf("\nAgent %i: sch %i", a, highlander_chromosome[a]);
		}
	}
}