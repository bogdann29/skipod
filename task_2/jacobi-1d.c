#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>

#define TSTEPS 20
#define N 30
#define KILLED_PROC_NUM 3

MPI_Comm comm;
int dead_proc = 0;

void init_array(int start, int end, float A[N], float B[N])
{
    int i;
    for (i = start; i < end; i++)
    {
        A[i-start] = ((float)i+2) / N;
        B[i-start] = ((float)i+3) / N;
    }
}

void print_array(float A[N])
{
    int i;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s\n", "A");
    for (i = 0; i < N; i++)
    {
        fprintf(stderr, "%0.2f ", A[i]);
    }
    fprintf(stderr, "\nend   dump: %s\n", "A");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

void save_into_file(char *filename, int start_index, int end_index, float* A, float* B)
{
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    int idxs[2] = {start_index, end_index};
    MPI_File_write_at(fh, 0, idxs, 2, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, sizeof(int) * 2, A, end_index-start_index, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, sizeof(int) * 2 + (end_index-start_index)*sizeof(float), B, end_index-start_index, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    // int cnt = 0;
    // FILE *fd = fopen(filename, "wb");
    // fwrite(&start_index, sizeof(int), 1, fds);
    // fwrite(&end_index, sizeof(int), 1, fd);
    // fwrite(A, sizeof(float), end_index-start_index, fd);
    // fwrite(B, sizeof(float), end_index-start_index, fd);
    // fclose(fd);
}

void read_from_file(char *filename, int *start_index, int *end_index, float** A, float** B)
{
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, 0, start_index, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, sizeof(int), end_index, 1, MPI_INT, MPI_STATUS_IGNORE);

    int part_size = *end_index - *start_index;
    // printf("start = %d, end = %d, part = %d\n", *start_index, *end_index, part_size);

    *A = (float*)malloc(part_size * sizeof(float));
    *B = (float*)malloc(part_size * sizeof(float));
    MPI_File_read_at(fh, sizeof(int) * 2, *A, part_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, sizeof(int) * 2 + (part_size)*sizeof(float), *B, part_size, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
    // FILE *fd = fopen(filename, "rb");
    // fread(start_index,  sizeof(int), 1, fd);
    // fread(end_index,  sizeof(int), 1, fd);

    // int part_size = *end_index - *start_index;
    // printf("start = %d, end = %d, part = %d\n", *start_index, *end_index, part_size);

    // *A = (float*)malloc(part_size * sizeof(float));
    // *B = (float*)malloc(part_size * sizeof(float));

    // fread(*A, sizeof(float), part_size, fd);
    // fread(*B, sizeof(float), part_size, fd);
    // // for(int i = 0; i < part_size; ++i)
    // //     printf("%f ",A[i]);
    // fflush(stdout);
    // fclose(fd);
}

void my_errhandler(MPI_Comm *com, int *err, ...) {
    int size, rank;
    MPI_Comm_rank(comm, &rank);
    printf("rank = %d\n", rank);
    
    char dead_filename[20];
    MPI_Group group_failed;
      
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int num_failed;

    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_failed);
    MPI_Group_size(group_failed, &num_failed);
    if (num_failed > 1) {
        printf("More than 1 proc failed.\n");
    }

    int new_rank, new_size;
    MPIX_Comm_shrink(comm, &comm);
    MPI_Comm_rank(comm, &new_rank);
    MPI_Comm_size(comm, &new_size);

    int *procs = (int*)malloc(sizeof(int) * new_size);
    MPI_Barrier(comm);
    MPI_Gather(&rank, 1, MPI_INT, procs, 1, MPI_INT, 0, comm);

    int dead_start_idx, dead_end_idx;

    if (new_rank == 0) {
        for (int i = 0; i < new_size - 1; ++i) {
            if (procs[i + 1] - procs[i] > 1) {
                dead_proc = procs[i] + 1;
            }
        }
        if (dead_proc == 0) {
            dead_proc = procs[new_size-1] + 1;
        }
        printf("Dead proc num: %d\n", dead_proc);
        sprintf(dead_filename, "%d", dead_proc);
        float *A, *B;
        read_from_file(dead_filename, &dead_start_idx, &dead_end_idx, &A, &B);
        save_into_file("tmp", dead_start_idx, dead_end_idx, A, B);
        free(A);
        free(B);
    }

    free(procs);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int tsteps = TSTEPS, n = N, i = 0;
    int rank = 0, size = 0;
    char filename[20];
    MPI_Errhandler my_errh;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sprintf(filename, "%d", rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size--;

    int start_index = (int)(n * rank / size);
    int end_index = (int)(n * (rank + 1)/ size); //не включается
    int part_size = end_index - start_index;

    float *A, *B; 

    MPI_Comm_create_errhandler(my_errhandler, &my_errh);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, my_errh);
    comm = MPI_COMM_WORLD;
    MPI_Barrier(comm);

    if(rank != size + 1){
        A = (float*)malloc(part_size * sizeof(float));
        B = (float*)malloc(part_size * sizeof(float));

        init_array(start_index, end_index, A, B);

        // printf("process_rank = %d, num_processes = %d\n", rank, size);

        printf("start = %d, end = %d\n", start_index, end_index);

        for (i = 1; i < part_size - 1; i++) {
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]);
        }

        save_into_file(filename, start_index, end_index, A, B);

        if (rank == KILLED_PROC_NUM) {
            raise(SIGKILL);
        }
    }
    MPI_Barrier(comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    float left, right;

    sprintf(filename, "%d", rank);
    free(A);
    free(B);
    read_from_file(filename, &start_index, &end_index, &A, &B);
    part_size = end_index - start_index;
    MPI_Barrier(comm);
    if(rank == 0){
        MPI_Send(&(A[part_size - 1]), 1, MPI_FLOAT, rank + 1, 0, comm);
        MPI_Recv(&right, 1, MPI_FLOAT, rank + 1, 0, comm, &status);
        
        B[part_size-1] = 0.33333 * (A[part_size-2] + A[part_size-1] + right);
        // printf("rank = %d, B[last] = %f\n", rank, B[part_size-1]);
    }

    else if (rank == size - 1){
        MPI_Recv(&left, 1, MPI_FLOAT, rank - 1, 0, comm, &status);
        MPI_Send(A, 1, MPI_FLOAT, rank - 1, 0, comm);
        B[0] = 0.33333 * (left + A[0] + A[1]); 
        // printf("rank = %d, B[0] = %f\n", rank, B[0]);
    }
    else{
        // printf("0 rank = %d, B[0] = %f\n", rank, B[0]);
        MPI_Recv(&left, 1, MPI_FLOAT, rank - 1, 0, comm, &status);
        MPI_Send(&(A[part_size - 1]), 1, MPI_FLOAT, rank + 1, 0, comm);
        MPI_Recv(&right, 1, MPI_FLOAT, rank + 1, 0, comm, &status);
        MPI_Send(A, 1, MPI_FLOAT, rank - 1, 0, comm);
        B[0] = 0.33333 * (left + A[0] + A[1]); 
        B[part_size-1] = 0.33333 * (A[part_size-2] + A[part_size-1] + right);
        // printf("rank = %d, B[0] = %f\n", rank, B[0]);
    }
    printf("end rank = %d\n", rank);
    MPI_Barrier(comm);
    free(A);
    free(B);
    
    MPI_Finalize();

    return 0;
}