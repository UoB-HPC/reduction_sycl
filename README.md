This repo contains a reproducible bug on our hand-rolled tree reduction (not the SYCL2020 one). The
bug only occurs in newer versions (> 0.8.2) of hipSYCL, presumably introduced by the new scheduler.

This program implements the classic tree reduction with `fmin` as operator. The original
implementation is part of the SYCL port of [CloverLeaf](https://github.com/UoB-HPC/cloverleaf_sycl).
As CloverLeaf requires multiple reductions with different operators, we've implemented a fully
templated one there. For reproduction, we've lifted that out into `reduction.hpp`. There's also a
non-generic version directly in `main.cpp`, as the generic version is quite convoluted to work with.
The end result is the same, the reduction gives the wrong answer.

We've tested this repo against DPC++ and ComputeCpp and both produce the correct answer.

Note that the generic version of the reduction creates multiple `accessors` on the same `buffer`
object. I believe this is allowed in the spec:

> When multiple accessors in the same command group define requirements to the
> same memory object, the access mode  is resolved as the union of all the
> different access modes, ...
> (SYCL™ Specification 1.2.1R6 p.g 25)

## Building

To reproduce with hipSYCL:

First, verify that the results are passing for hipSYCL <= 0.8.2

```shell
> cmake -Bbuild -H. \ 
  -DSYCL_RUNTIME=HIPSYCL \
  -DHIPSYCL_PLATFORM=cpu \ 
  -DHIPSYCL_INSTALL_DIR="/opt/hipsycl/46bc9bd/" \ 
  -DCMAKE_BUILD_TYPE=Release
> cmake3 --build build --target reduction --config Release -j $(nproc)
```

```
[SYCL] Device        : hipCPU OpenMP host device
[SYCL]  - Vendor     : hipCPU
[SYCL]  - Extensions : 
[SYCL]  - Platform   : hipSYCL [SYCL over CUDA/HIP] on hipCPU host device
[SYCL]     - Vendor  : The hipSYCL project
[SYCL]     - Version : hipSYCL 0.8.2-release on HIP/CUDA 99999
[SYCL]     - Profile : FULL_PROFILE
Run #0 Expected=0 Actual=0 OK!
Run #1 Expected=0 Actual=0 OK!
Run #2 Expected=0 Actual=0 OK!
Run #3 Expected=0 Actual=0 OK!
Run #4 Expected=0 Actual=0 OK!
Run #5 Expected=0 Actual=0 OK!
...
```

Then, run with hipSYCL > 0.8.2:

```shell
> cmake -Bbuild -H. \ 
  -DSYCL_RUNTIME=HIPSYCL-NEXT \ # XXX use HIPSYCL-NEXT because of dir structure changes 
  -DHIPSYCL_PLATFORM=cpu \ 
  -DHIPSYCL_INSTALL_DIR="/opt/hipsycl/cff515c/" \ 
  -DCMAKE_BUILD_TYPE=Release
> cmake3 --build build --target reduction --config Release -j $(nproc)
```

```
[SYCL] Device        : hipSYCL OpenMP host device
[SYCL]  - Vendor     : the hipSYCL project
[SYCL]  - Extensions : 
[SYCL]  - Platform   : OpenMP
[SYCL]     - Vendor  : The hipSYCL project
[SYCL]     - Version : hipSYCL 0.9.0-git
[SYCL]     - Profile : FULL_PROFILE
Run #0 Expected=0 Actual=0 OK!
Run #1 Expected=0 Actual=0 OK!
Run #2 Expected=0 Actual=1 FAIL!
Run #3 Expected=0 Actual=1 FAIL!
Run #4 Expected=0 Actual=1 FAIL!
Run #5 Expected=0 Actual=0 OK!
Run #6 Expected=0 Actual=0 OK!
Run #7 Expected=0 Actual=0 OK!
Run #8 Expected=0 Actual=1 FAIL!
Run #9 Expected=0 Actual=1 FAIL!
...
```

The use of generic reduction can be selected with `useGeneric` in `main.cpp`.  
There's also this `const double size = 128;` in `main.cpp`. Setting that to larger sizes (> 8092)
would reduce verification errors but it does still happens occasionally.

To test with any other SYCL implementations, replace the hipSYCL CMake flags with:

* DPC++ - `-DSYCL_RUNTIME DPCPP`
* ComputeCpp - `-DSYCL_RUNTIME COMPUTECPP -DComputeCpp_DIR=<path_to_computecpp>`