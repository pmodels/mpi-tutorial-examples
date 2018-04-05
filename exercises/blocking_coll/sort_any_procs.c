#include <mpi.h>
#include <stdlib.h>
#define N 60

/*
 * This program sorts integer arrays in ascending order. Array sizes must be divisible by the number of processes
 */

int main(int argc, char *argv[]) {
	int rank, num_procs;
	int a[N];
	int *b, i, j;
	int sort_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (i = 0; i < N; i++) {
		a[i] = N - i;
	}

	sort_size = N / num_procs;
	b = (int *) malloc(sort_size * sizeof(int));
	MPI_Scatter(a, sort_size, MPI_INT, b, sort_size, MPI_INT, 0,
			MPI_COMM_WORLD);
	/*
	 * Sort sub array
	 */
	for (i = 0; i < sort_size - 1; i++) {
		int min = i;
		int temp = b[i];
		for (j = i + 1; j < sort_size; j++) {
			if (temp > b[j]) {
				temp = b[j];
				min = j;
			}
		}
		temp = b[i];
		b[i] = b[min];
		b[min] = temp;
	}

	MPI_Gather(b, sort_size, MPI_INT, a, sort_size, MPI_INT, 0, MPI_COMM_WORLD);

	/*
	 * Merge in root
	 */
	if (rank == 0) {
		i = 0;
		while (i < num_procs - 1) {
			int first = 0, second = 0;
			int *temp1 = (int*) malloc((i+1)*sort_size*sizeof(int));
			int *temp2 = (int*) malloc(sort_size*sizeof(int));
			for (j = 0; j < (i + 1) * sort_size; j++) {
				temp1[j] = a[j];
			}

			for (j = 0; j < sort_size; j++) {
				temp2[j] = a[j + (i + 1) * sort_size];
			}

			j = 0;
			while (first < (i + 1 * sort_size) && second < sort_size) {
				if (temp1[first] < temp2[second]) {
					a[j++] = temp1[first];
					first++;
				} else {
					a[j++] = temp2[second];
					second++;
				}
			}

			while (first < ((i + 1) * sort_size)) {
				a[j++] = temp1[first++];
			}

			while (second < sort_size) {
				a[j++] = temp2[second++];
			}

			i++;
			free(temp1);
			free(temp2);
		}

		for (i = 0; i < N; i++) {
			printf("%d\n", a[i]);
		}
	}

	MPI_Finalize();
	return 0;
}
