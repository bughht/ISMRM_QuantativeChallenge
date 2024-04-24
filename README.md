# T1 and T2 star mapping for ISMRM Quantative Challenge 2024

## Preprocessing

Based on visual inspection, gibbs ringing artifacts were observed in the magnitude dicom images, this might caused by the partial echo acquisition. To reduce the influence of the ringing artifacts, a 2D Gaussian filter was applied to the images to smooth the ringings. We've also compared the results with and without the filtering procedure.

## Mapping: SVD compressed Dictionary Based Mapping

The idea of SVD compressed dictionary based mapping was first proposed by McGivney, Debra F., et al in 2014 [1]. Here we implemented the algorithm for T1 and T2 star mapping. The only difference between the original algorithm and our implementation is that we calculated the pearson correlation coefficient instead of vector production to find the best match between the dictionary and the measured signal.

Here the number of singular values used for compression is decided by the threshold, which is the percentage of the total energy of the singular values. 

## Dictionary Generation

The dictionary for T1 and T2 star mapping were generated based on the Bloch equations of T1 and T2 star relaxation. The size of and the step size of the dictionary were given as follows:

+ T1: 50 ms to 2300 ms, step size 0.1 ms, dictionary size 22500
+ T2 star: 0.05 ms to 500 ms, step size 0.1 ms, dictionary size 10000

[1] McGivney, D.F., Pierre, E., Ma, D., Jiang, Y., Saybasili, H., Gulani, V. and Griswold, M.A., 2014. SVD compression for magnetic resonance fingerprinting in the time domain. IEEE transactions on medical imaging, 33(12), pp.2311-2322.