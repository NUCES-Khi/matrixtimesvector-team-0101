# Assignment 1
## Team Members
|std_id|Name|
|--------|-|
|k21-4714|Muhammad Hasnain|
|k21-4713|Khush Bakht Aleeza|
|k21-3056|Ahmed Raza|
## Output Screenshots

![All_outputs](https://github.com/NUCES-Khi/matrixtimesvector-team-0101/assets/125374283/9ce36b56-895e-4657-9a89-ea16a961546d)

![seq](https://github.com/NUCES-Khi/matrixtimesvector-team-0101/assets/125374283/b2c2f0b0-22e9-413c-a851-4d04fb5399b7)



## Results and Analysis
//-- Show graphs results and charts where necessary and discuss the results and what they signify. --// 
## Major Problems Encountered
1. Issue 1:
In the initial implementation of our distributed matrix multiplication algorithm using MPI, we encountered a significant challenge with the `MPI_Gather` function, which was intended to collect the partial results computed by each processor. The problem arose because `MPI_Gather` assembles data from all processes in a collective manner, expecting each process to contribute a segment of equal size. However, due to the nature of our matrix partitioning strategy, which was designed to optimize load balancing and computational efficiency, the data chunks processed by each processor varied in size, especially towards the end of the matrix where the remaining rows didn't divide evenly among processors.
This discrepancy led to a mismatch in the expected data sizes during the gathering phase, causing incorrect assembly of the final result matrix. Essentially, processors with fewer rows to process ended up contributing less data than anticipated, leading to gaps in the assembled matrix.
To rectify this issue, we shifted from using `MPI_Gather` to `MPI_Gatherv`. The `MPI_Gatherv` function is a more flexible variant that allows each process to send varying amounts of data to the root process. By specifying the exact count of elements each process was sending, along with the appropriate displacements where each segment should be placed in the final array, we were able to correctly assemble the final result matrix regardless of the uneven distribution of workload among the processors. This adjustment ensured that our distributed matrix multiplication algorithm could efficiently and accurately compute the results across different processor configurations, maintaining both scalability and correctness.
    - **Resolved**
3. Issue 2: Blah blah blah ....
    - Solution1: tried to blah blahb
    - **Resolved**
