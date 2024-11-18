## General Flow to Execute Code on GPU
1. Set Up Data on Host (CPU):  \
   Prepare your data on the CPU. This includes any input arrays or variables that will be processed by the GPU.

2. Allocate Memory on the Device (GPU):  \
   CUDA uses device memory, and you need to allocate space for the input and output data on the GPU.
3. Copy Data from Host to Device:  \
4. After allocating memory on the GPU, you need to copy the data from the CPU (host) to the GPU (device) using cudaMemcpy.

5. Define CUDA Kernel: \
A kernel is a function that runs on the GPU. It is marked with ```__global__```. Each thread in the GPU will run this kernel, and the kernel can be designed to perform operations in parallel on large datasets.

6. Launch the Kernel: \
The kernel is executed by specifying the number of threads and blocks. Each thread will work on a part of the dataset. The kernel is launched asynchronously to the CPU, meaning the CPU continues executing while the GPU works in parallel.

7. Synchronize the Device: \
After launching the kernel, ensure that the GPU finishes its work before the CPU proceeds. This is done using cudaDeviceSynchronize() or using events to measure the time taken by the GPU.

8. Copy the Results from Device to Host: \
After the GPU has completed the operation, you need to copy the results back to the host (CPU) memory for further processing or output.

9. Free Device Memory: \
Once you're done with the GPU memory, free it using cudaFree