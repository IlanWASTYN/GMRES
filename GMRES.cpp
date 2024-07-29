/* -*-coding: utf - 8 - *-

Created on Mon Jul 22 2024

@author : WASTYN Ilan
*/

#include <mpi.h>
#include <iostream>
#include <new>
#include <cmath>

class vector
{
public:
    int size;
    double* body;
    void allocate();
    void print();
};

void vector::allocate()
{
    body = new double[size];
    for (int i = 0; i < size; i++)
    {
        body[i] = 0;
    }
}

void vector::print()
{
    for (int i = 0; i < size; i++) {
        printf("%lf\n", body[i]);
    }
    printf("\n");
}

class matrix {
public:
    int size_row;
    int size_col;
    double* body;
    void allocate();
    void print();
};

void matrix::allocate()
{
    body = new double[size_row * size_col];
    for (int i = 0; i < size_row * size_col; i++)
    {
        body[i] = 0;
    }
}

void matrix::print()
{
    for (int j = 0; j < size_row; j++) {
        for (int i = 0; i < size_col; i++)
        {
            printf("%lf ", body[i + j * size_col]);
        }
        printf("\n");
    }
    printf("\n");
}

class vector_par
{
public:

    // Variables
    int size; // Size vector on proc 0
    int size_par; // Size vector parallelized (proc != 0)
    double* body; // Body of the vector, we put values in here
    double norm_var; // Value of the norm
    int proc_nb; // Value of the current proc
    int proc_size; // Value of the total proc

    // Functions
    void initialize(int, int, int, int); // Getting everyneed values (size,size_par,proc_nb,proc_size)
    void allocate(); // Allocating needed memory space on each proc
    void print(); // Printing vector, column format
    void master_to_procs(); // Spliting and sending vector to each proc
    void procs_to_master(); // Combining and sending vector to master proc (0)
    void prepare_prod(); // Prepare vector for matrix product (tridiagonal matrix) by adding needed values 
    void unprepare_prod(); // Unpreparing prepared vector after usage
    double norm(); // Norm of the vector
};

void vector_par::prepare_prod() {
    // Allocating with new dimensions
    if (proc_nb != 0)
    {
        free(body);
        if (proc_nb == 1 || proc_nb == proc_size - 1)
        {
            body = new double[size_par + 1];
            for (int i = 0; i < size_par + 1; i++)
            {
                body[i] = 0;
            }
        }
        else
        {
            body = new double[size_par + 2];
            for (int i = 0; i < size_par + 2; i++)
            {
                body[i] = 0;
            }
        }
    }
    // Master_to_procs
    if (proc_nb == 0)
    {
        for (int i_proc = 1; i_proc < proc_size; i_proc++)
        {
            for (int i = 0; i < size_par; i++)
            {
                MPI_Send(&body[i + (i_proc - 1) * size_par], 1, MPI_DOUBLE, i_proc, 1000, MPI_COMM_WORLD);
            }
        }
    }
    else if(proc_nb != 1)
    {
        for (int i = 0; i < size_par; i++)
        {
            MPI_Recv(&body[i+1], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int i = 0; i < size_par; i++)
        {
            MPI_Recv(&body[i], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    
    if (proc_nb == 1) {
        // Send to proc 2 and receive from proc 2 
        MPI_Send(&body[size_par - 1], 1, MPI_DOUBLE, 2, 1000, MPI_COMM_WORLD);
        MPI_Recv(&body[size_par], 1, MPI_DOUBLE, 2, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (proc_nb == proc_size - 1)
    {
        // Send to last proc - 1 and receive from last proc - 1
        MPI_Recv(&body[0], 1, MPI_DOUBLE, proc_size - 2, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&body[1], 1, MPI_DOUBLE, proc_size - 2, 1000, MPI_COMM_WORLD);
    }
    else if (proc_nb != 0 && proc_nb != 1 && proc_nb != proc_size - 1)
    {
        // Send to proc - 1 and receive from proc - 1 
        MPI_Recv(&body[0], 1, MPI_DOUBLE, proc_nb - 1, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&body[1], 1, MPI_DOUBLE, proc_nb - 1, 1000, MPI_COMM_WORLD);

        // Send to proc + 1 and receive from proc + 1
        MPI_Send(&body[size_par], 1, MPI_DOUBLE, proc_nb + 1, 1000, MPI_COMM_WORLD);
        MPI_Recv(&body[size_par + 1], 1, MPI_DOUBLE, proc_nb + 1, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void vector_par::initialize(int n, int nl, int world_rank, int world_size) {
    if (n/(world_size - 1) % int(nl) != 0)
    {
        printf("cant allocate");
    }
    size = n; // Getting size value
    size_par = nl; // Getting size after par value
    proc_nb = world_rank; // Getting proc rank value 
    proc_size = world_size; // Getting number of procs used value
    allocate(); // Allocating
}

void vector_par::allocate()
{
    if (proc_nb == 0)
    {
        body = new double[size];
        for (int i = 0; i < size; i++)
        {
            body[i] = 0;
        }
    }
    else
    {
        body = new double[size_par];
        for (int i = 0; i < size_par; i++)
        {
            body[i] = 0;
        }
    }

}

void vector_par::print()
{
    printf("Hello world from rank %d out of %d processors\n",
        proc_nb, proc_size);
    if (proc_nb == 0)
    {
        for (int i = 0; i < size; i++) {
            printf("%lf\n", body[i]);
        }
        printf("\n");
    }
    else
    {
        for (int i = 0; i < size_par; i++) {
            printf("%lf\n", body[i]);
        }
        printf("\n");
    }
}

void vector_par::master_to_procs() {
    if (proc_nb == 0)
    {
        for (int i_proc = 1; i_proc < proc_size; i_proc++)
        {
            for (int i = 0; i < size_par; i++)
            {
                MPI_Send(&body[i + (i_proc - 1) * size_par], 1, MPI_DOUBLE, i_proc, 1000, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (int i = 0; i < size_par; i++)
        {
            MPI_Recv(&body[i], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void vector_par::procs_to_master() {
    if (proc_nb == 0)
    {
        for (int i_proc = 1; i_proc < proc_size; i_proc++)
        {
            for (int i = 0; i < size_par; i++)
            {
                MPI_Recv(&body[i + (i_proc - 1) * size_par], 1, MPI_DOUBLE, i_proc, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }


    }
    else
    {
        for (int i = 0; i < size_par; i++)
        {
            MPI_Ssend(&body[i], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD);
        }

    }
}

double vector_par::norm() {
    double norm_loc = 0;
    master_to_procs(); 
    if (proc_nb != 0)
    {
        for (int i = 0; i < size; i++) {
            norm_loc += body[i] * body[i];
        }
    }
    MPI_Reduce(&norm_loc, &norm_var, 1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD);
    norm_var = sqrt(norm_var);
    MPI_Bcast(&norm_var, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return norm_var;
}



class matrix_par
{
public:

    //Variables
    int size_row;
    int size_col;
    int size_row_par;
    
    double* body;
    int proc_nb;
    int proc_size;
    MPI_Status status;

    // Functions
    void initialize(int, int, int, int, int);
    void allocate();
    void print();
    void master_to_procs();
    void procs_to_master();
};


void matrix_par::initialize(int n, int m,int nl, int world_rank, int world_size) {
    size_row = n;
    size_col = m;
    size_row_par = nl;
    proc_nb = world_rank;
    proc_size = world_size;
    allocate();
}

void matrix_par::allocate() {
    if (proc_nb == 0)
    {
        body = new double[size_row * size_col];
        for (int j = 0; j < size_row; j++)
        {
            for (int i = 0; i < size_col; i++)
            {
                body[i + j * size_col] = 0;
            }
        }
    }
    else
    {
        body = new double[size_row_par * size_col];
        for (int j = 0; j < size_row_par; j++)
        {
            for (int i = 0; i < size_col; i++)
            {
                body[i + j * size_col] = 0;
            }
        }
    }
}

void matrix_par::print()
{
    if (proc_nb == 0)
    {
        printf("Hello world from rank %d out of %d processors\n",
            proc_nb, proc_size);
        for (int j = 0; j < size_row; j++) {
            for (int i = 0; i < size_col; i++)
            {
                printf("%lf ", body[i + j * size_col]);
            }
            printf("\n");
        }
        printf("\n");
    }
    else
    {
        printf("Hello world from rank %d out of %d processors\n",
            proc_nb, proc_size);
        for (int j = 0; j < size_row_par; j++) {
            for (int i = 0; i < size_col; i++)
            {
                printf("%lf ", body[i + j * size_col]);
            }
            printf("\n");
        }
        printf("\n");
    }

}

void matrix_par::master_to_procs() {
    if (proc_nb == 0)
    {
        for (int i_proc = 1; i_proc < proc_size; i_proc++)
        {
            for (int i = 0; i < size_row_par * size_col; i++)
            {
                MPI_Send(&body[i + (i_proc - 1) * size_row_par * size_col], 1, MPI_DOUBLE, i_proc, 1000, MPI_COMM_WORLD);
            }
        }


    }
    else
    {
        for (int i = 0; i < size_row_par * size_col; i++)
        {
            MPI_Recv(&body[i], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void matrix_par::procs_to_master() {
    if (proc_nb == 0)
    {
        for (int i_proc = 1; i_proc < proc_size; i_proc++)
        {
            for (int i = 0; i < size_row_par * size_col; i++)
            {
                MPI_Recv(&body[i + (i_proc - 1) * size_row_par * size_col], 1, MPI_DOUBLE, i_proc, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }


    }
    else
    {
        for (int i = 0; i < size_row_par * size_col; i++)
        {
            MPI_Send(&body[i], 1, MPI_DOUBLE, 0, 1000, MPI_COMM_WORLD);
        }
    }
}

vector_par sum(vector_par U, vector_par V) {
    vector_par W;
    W.initialize(U.size, U.size_par, U.proc_nb, U.proc_size);
    U.master_to_procs(); V.master_to_procs();
    if (U.proc_nb != 0)
    {
        for (int i = 0; i < U.size_par; i++) {
            W.body[i] = U.body[i] + V.body[i];
        }
    }
    W.procs_to_master();
    return W;
}
vector_par diff(vector_par U, vector_par V) {
    vector_par W;
    W.initialize(U.size, U.size_par, U.proc_nb, U.proc_size);
    U.master_to_procs(); V.master_to_procs();
    if (U.proc_nb != 0)
    {
        for (int i = 0; i < U.size_par; i++) {
            W.body[i] = U.body[i] - V.body[i];
        }
    }
    W.procs_to_master();
    return W;
}

vector_par prod_matr_vect_par(matrix_par A, vector_par U) {
    vector_par Y;
    Y.initialize(U.size, U.size_par, U.proc_nb, U.proc_size);

    U.prepare_prod();
    A.master_to_procs();
    printf("Hello world from rank %d out of %d processors\n",
        A.proc_nb, A.proc_size);
    vector row_index;
    if (A.proc_nb == 0)
    {
        matrix buff;
        buff.size_row = A.proc_size - 1; buff.size_col = 3;
        buff.print();
        row_index.size = A.size_row + 1;
        //row_index.allocate();
        //row_index.print();
    }
    else if (A.proc_nb != 0) {
        int nnz = 0;
        for (int j = 0; j < A.size_row_par; j++)
        {
            for (int i = 0; i < A.size_col; i++)
            {
                if (A.body[i + j * A.size_col] != 0)
                {
                    nnz++;
                }
            }
        }
        
        vector val, col_index;
        val.size = col_index.size = nnz;
        row_index.size = 3;
        val.allocate(); col_index.allocate(); row_index.allocate();
        row_index.body[0] = 0;
        nnz = 0;
        /*for (int j = 0; j < A.size_row_par; j++)
        {
            for (int i = 0; i < A.size_col; i++)
            {
                if (A.body[i + j * A.size_col] != 0)
                {
                    val.body[nnz] = A.body[i + j * A.size_col];
                    col_index.body[nnz] = i;
                    nnz++;
                }
            }
            //row_index.body[j + 1] = nnz;
        }
        row_index.print();*/
    }
    
        /*for (int i = 0; i < row_index.size - 1; i++)
        {
            for (int j = row_index.body[i]; j < row_index.body[i + 1]; j++) {
                Y.body[i] += val.body[j] * U.body[int(col_index.body[j])];
            }
        }
        printf("nnz elem = %d\n", nnz);*/
        
    
    //Y.procs_to_master();
    
    return Y;
}

double scalar_product(vector_par U, vector_par V) {
    double ps_loc = 0;
    double ps;
    U.master_to_procs(); V.master_to_procs();
    if (U.proc_nb != 0)
    {
        for (int i = 0; i < U.size_par; i++) {
            ps_loc += U.body[i] * V.body[i];
        }
    }
    MPI_Reduce(&ps_loc, &ps, 1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD);
    MPI_Bcast(&ps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return ps;
}

vector_par extract_vect_par(matrix_par A, int col){
    vector_par W;
    W.initialize(A.size_row,A.size_row_par,A.proc_nb,A.proc_size);
    if (W.proc_nb != 0)
    {
        for (int i = 0; i < W.size_par; i++)
        {
            W.body[(W.proc_nb - 1) * W.size_par + i] = A.body[(W.proc_nb - 1) * W.size_par * (W.proc_nb + 1) + (W.proc_nb + 1) * i + col];
        }
    }
    return W;
}

vector_par include_vect_par() {
    vector_par W;
    return W;
}

vector_par GMRES(vector_par x, matrix_par A, vector_par b, int nkry) {
    // Initializing result vector
    vector_par res;
    res.initialize(x.size, x.size_par, x.proc_nb, x.proc_size);
    
    // Initializing matrixes from Arnoldi's process
    matrix_par H, Q;
    Q.initialize(x.size, nkry + 1 ,x.size_par, x.proc_nb, x.proc_size);
    H.initialize(nkry + 1, nkry ,x.size_par, x.proc_nb, x.proc_size);
    
    // Calculating r_0
    res = diff(b,prod_matr_vect_par(A,x));
    res.norm();
    printf("Norm = %lf\n", res.norm_var);
    
    // Q[:][0] = r_0/||r_0||_2
    if (x.proc_nb != 0) {
        for (int i = 0; i < x.size_par; i++)
        {
            Q.body[(x.proc_nb - 1) * x.size_par * (nkry + 1) + (nkry+1) * i] = res.body[(x.proc_nb - 1) * x.size_par + i]/res.norm_var;
        }
    }
    // Calculating Arnoldi's
    for (int j = 0; j < nkry; j++)
    {
    }
    Q.print();
    return res;
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Variables
    int row = 8;
    int col = 8;
    int nkry = 5;
    int row_par = row / (world_size-1);

    // Giving names
    vector_par x,b;
    matrix_par A;
    x.initialize(row, row_par, world_rank, world_size);
    b.initialize(row, row_par, world_rank, world_size);
    A.initialize(row, col, row_par, world_rank, world_size);

    // Giving values to proc 0
    if (world_rank == 0) {
        for (int j = 0; j < row; j++) {
            x.body[j] = j + 1;
            b.body[j] = 1;
            A.body[j + j * col] = 2;
        }
        for (int j = 0; j < row - 1; j++) 
        {
            A.body[j + j * col + 1] = 1;
            A.body[j + (j + 1)* col] = 1;
        }
    } 
    
    prod_matr_vect_par(A,x);
    //printf("%lf",scalar_product(x, b));
    //extract_vect_par(A, 1).print();
    //GMRES(x, A, b, nkry);
    //prod_matr_vect_par(A, b).print();
    // Finalize the MPI environment. 
    MPI_Finalize();
}