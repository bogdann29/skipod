#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>
#include <memory.h>

#define MATRIX_SIZE 4
#define MIN(x, y) x > y ? y : x

MPI_Request send_number(int first_coord, int second_coord, MPI_Comm comm, int *data){
    int target_rank;
    int coords[] = {first_coord, second_coord};
    MPI_Cart_rank(comm, coords, &target_rank);

    MPI_Request request;
    MPI_Isend(data, 3, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &request);
    return request;
}

int* receive_number(int first_coord, int second_coord, MPI_Comm comm) {
    int target_rank;
    int coords[] = {first_coord, second_coord};
    MPI_Cart_rank(comm, coords, &target_rank);

    int* data = (int*)malloc(sizeof(int) * 3);
    MPI_Recv(data, 3, MPI_INT, target_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return data;
}


int main(int argc, char **argv) {
    int process_rank, num_processes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // printf("process_rank = %d, num_processes = %d\n", process_rank, num_processes);

    int n_dims = 2;
    int coords[2];
    int dims[2] = {MATRIX_SIZE, MATRIX_SIZE};
    int periods[2] = {0, 0};
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, n_dims, dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, process_rank, n_dims, coords);

    srand(time(NULL) + process_rank);

    int num = rand() % 256;
    printf("Number in [%d, %d] = %d\n", coords[0], coords[1], num);
    sleep(1);

    int* recv_data;
    int data_to_send[3] = {num, coords[0], coords[1]};
    if (coords[0] == 0) {
        recv_data = receive_number(coords[0] + 1, coords[1], cart_comm);
        printf("Receive number %d, my coords are [%d, %d]\n", recv_data[0], coords[0], coords[1]);
        fflush(stdout);
        if(recv_data[0] < num){
            num = recv_data[0];
            memcpy(data_to_send, recv_data, 3 * sizeof(int));
        }
        free(recv_data);

        if (coords[1] != MATRIX_SIZE - 1) {
            recv_data = receive_number(coords[0], coords[1] + 1, cart_comm);
            printf("Receive number %d, my coords are [%d, %d]\n", recv_data[0], coords[0], coords[1]);
            if(recv_data[0] < num){
                num = recv_data[0];
                memcpy(data_to_send, recv_data, 3 * sizeof(int));
            }
            free(recv_data);
        }

        if (coords[1] != 0) {
            MPI_Request request = send_number(coords[0], coords[1] - 1, cart_comm, data_to_send);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }
    else {
        if (coords[0] != MATRIX_SIZE - 1) {
            recv_data = receive_number(coords[0] + 1, coords[1], cart_comm);
            printf("Receive number %d, my coords are [%d, %d]\n", recv_data[0], coords[0], coords[1]);
            if(recv_data[0] < num){
                num = recv_data[0];
                memcpy(data_to_send, recv_data, 3 * sizeof(int));
            }
            free(recv_data);
        }
        MPI_Request request = send_number(coords[0] - 1, coords[1], cart_comm, data_to_send);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    sleep(1);

    if (process_rank == 0)
        printf("\nMin number is %d\n\n", num);

    MPI_Barrier(cart_comm);
    sleep(1);

    if (coords[0] == 0) {
        if (coords[1] != 0) {
            recv_data = receive_number(coords[0], coords[1] - 1, cart_comm);
            printf("Receive number, my coords are [%d, %d]\n", coords[0], coords[1]);
            printf("Minimum is %d in coords - [%d, %d]\n", recv_data[0], recv_data[1], recv_data[2]);
            memcpy(data_to_send, recv_data, 3 * sizeof(int));
            free(recv_data);
        }

        MPI_Request request1, request2;
        request1 = send_number(coords[0] + 1, coords[1], cart_comm, data_to_send);

        if (coords[1] != MATRIX_SIZE - 1) {
            request2 = send_number(coords[0], coords[1] + 1, cart_comm, data_to_send);
            MPI_Wait(&request2, MPI_STATUS_IGNORE);
        }
        MPI_Wait(&request1, MPI_STATUS_IGNORE);

    } else {
        recv_data = receive_number(coords[0] - 1, coords[1], cart_comm);
        printf("Receive number, my coords are [%d, %d]\n", coords[0], coords[1]);
        printf("Minimum is %d in coords - [%d, %d]\n", recv_data[0], recv_data[1], recv_data[2]);
        memcpy(data_to_send, recv_data, 3 * sizeof(int));
        free(recv_data);

        MPI_Request request;
        if (coords[0] != MATRIX_SIZE - 1) {
            request = send_number(coords[0] + 1, coords[1], cart_comm, data_to_send);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }

    MPI_Finalize();
    return 0;
}