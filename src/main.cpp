
#include <CL/sycl.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include "reduction.hpp"

using namespace cl;

double reduceSTL(const size_t N) {
  const size_t size = N * N;
  std::vector<double> xs(size, 0);
  // fill with dummy reduction data
  for (size_t i = 0; i < size; ++i) {
    xs[i] = i + 1;
  }
  // XXX set the second element to 0, the expected min value
  xs[1] = 0;
  return *std::min_element(xs.begin(), xs.end());
}

constexpr auto DOUBLE_MAX = std::numeric_limits<double>::max();

auto prepareSYCLBuffer(sycl::queue& queue, const size_t size) {
  sycl::buffer<double, 1> result((sycl::range<1>(size)));
  // fill with dummy reduction data
  queue.submit([&](sycl::handler& h) {
    auto xs = result.get_access<sycl::access::mode::write>(h);
    h.parallel_for<class fill>(sycl::range<1>(size), [=](sycl::id<1> idx) {
      // XXX set the second element to 0, the expected min value
      xs[idx] = idx[0] == 1 ? 0 : idx[0] + 1;
    });
  });
  return result;
}

double reduceSYCLGeneric(sycl::queue queue, const size_t N) {
  const size_t size = N * N;
  sycl::buffer<double, 1> result = prepareSYCLBuffer(queue, size);

  typedef LocalReducer<double, double,
                       sycl::accessor<double, 1, sycl::access::mode::read>>
      reducer;

  parallel_reduce_1d<class reduce>(
      queue, Range1D(0u, N * N),
      [=](sycl::handler& h, size_t& size) mutable {
        return reducer(
            h, size, {result.get_access<sycl::access::mode::read>(h)}, result);
      },
      [](const reducer& r, sycl::id<1> lidx) { r.local[lidx] = DOUBLE_MAX; },
      [](const reducer& r, sycl::id<1> lidx, sycl::id<1> idx) {
        r.local[lidx] = sycl::fmin(r.local[lidx], r.actual[idx]);
      },
      [](const reducer& r, sycl::id<1> idx, sycl::id<1> idy) {
        r.local[idx] = sycl::fmin(r.local[idx], r.local[idy]);
      },
      [](const reducer& r, size_t group, sycl::id<1> idx) {
        r.result[group] = r.local[idx];
      });

  queue.wait_and_throw();
  auto actual = result.get_access<sycl::access::mode::read>()[0];

  return actual;
}

double reduceSYCLNonGeneric(cl::sycl::queue queue, const size_t N) {
  const size_t size = N * N;
  sycl::buffer<double, 1> result = prepareSYCLBuffer(queue, size);
  // setup reduction parameters
  auto dev = queue.get_device();
  size_t dotNumGroups;
  size_t dotWgsize;
  if (dev.is_cpu()) {
    dotNumGroups = dev.get_info<sycl::info::device::max_compute_units>();
    dotWgsize =
        dev.get_info<sycl::info::device::native_vector_width_double>() * 2;

  } else {
    dotNumGroups = dev.get_info<sycl::info::device::max_compute_units>() * 4;
    dotWgsize = dev.get_info<sycl::info::device::max_work_group_size>();
  }

  dotNumGroups = std::min(size, dotNumGroups);

  // typical partial sum reduction, we reduce to groups first
  // we are reusing the result buffer here
  queue.submit([=](sycl::handler& h) mutable {
    cl::sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
        local(cl::sycl::range<1>(dotWgsize), h);
    auto drain(result.get_access<sycl::access::mode::read_write>(h));
    h.parallel_for<class reduce>(
        sycl::nd_range<1>(dotNumGroups * dotWgsize, dotWgsize),
        [=](sycl::nd_item<1> item) {
          size_t globalId = item.get_global_id(0);
          size_t localId = item.get_local_id(0);
          size_t globalSize = item.get_global_range()[0];

          local[localId] = DOUBLE_MAX;
          for (; globalId < size; globalId += globalSize) {
            local[localId] = sycl::fmin(local[localId], drain[globalId]);
          }
          size_t localSize = item.get_local_range()[0];
          for (size_t offset = localSize / 2; offset > 0; offset /= 2) {
            item.barrier(sycl::access::fence_space::local_space);
            if (localId < offset) {
              local[localId] =
                  sycl::fmin(local[localId], local[localId + offset]);
            }
          }
          if (localId == 0) {
            drain[item.get_group(0)] = local[0];
          }
        });
  });

  // then reduce groups to the first element
  queue.submit([=](cl::sycl::handler& h) mutable {
    cl::sycl::accessor<double, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
        local(cl::sycl::range<1>(dotNumGroups), h);

    auto drain(result.get_access<sycl::access::mode::read_write>(h));
    h.parallel_for<class final_reduction>(
        cl::sycl::nd_range<1>(1, 1), [=](auto) {
          local[0] = DOUBLE_MAX;
          for (size_t i = 0; i < dotNumGroups; ++i) {
            local[i] = drain[i];
          }
          for (size_t i = 1; i < dotNumGroups; ++i) {
            local[0] = sycl::fmin(local[0], local[i]);
          }
          drain[0] = local[0];
        });
  });

  queue.wait_and_throw();
  auto actual = result.get_access<sycl::access::mode::read>()[0];

  return actual;
}

int main() {
  cl::sycl::queue queue;

  const sycl::device& device = queue.get_device();
  auto exts = device.get_info<sycl::info::device::extensions>();
  std::ostringstream extensions;
  std::copy(exts.begin(), exts.end(),
            std::ostream_iterator<std::string>(extensions, ","));
  sycl::platform platform = device.get_platform();
  std::cout << "[SYCL] Device        : "
            << device.get_info<sycl::info::device::name>()
            << "\n[SYCL]  - Vendor     : "
            << device.get_info<sycl::info::device::vendor>()
            << "\n[SYCL]  - Extensions : " << extensions.str()
            << "\n[SYCL]  - Platform   : "
            << platform.get_info<sycl::info::platform::name>()
            << "\n[SYCL]     - Vendor  : "
            << platform.get_info<sycl::info::platform::vendor>()
            << "\n[SYCL]     - Version : "
            << platform.get_info<sycl::info::platform::version>()
            << "\n[SYCL]     - Profile : "
            << platform.get_info<sycl::info::platform::profile>() << "\n";

  // for small sizes, try 128 or 256
  // for large sizes, try >= 8192
  const double size = 128;

  constexpr bool useGeneric = true;

  for (int i = 0; i < 100; ++i) {
    auto expected = reduceSTL(size);

    double actual;
    if constexpr (useGeneric) {
      actual = reduceSYCLGeneric(queue, size);
    } else {
      actual = reduceSYCLNonGeneric(queue, size);
    }
    std::cout << "Run #" << i << " Expected=" << expected << " ";
    std::cout << "Actual=" << actual << " ";
    if (expected != actual) {
      std::cout << "FAIL!" << std::endl;
    } else {
      std::cout << "OK!" << std::endl;
    }
  }

  std::cout << "Done" << std::endl;
  return EXIT_SUCCESS;
}
