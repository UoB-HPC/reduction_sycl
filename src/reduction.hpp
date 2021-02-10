#pragma once

#include <CL/sycl.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace cl;

template <typename T, typename U, typename C>
struct LocalReducer {
  cl::sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
      local;

  C actual;
  sycl::accessor<U, 1, sycl::access::mode::read_write,
                 sycl::access::target::global_buffer>
      result;

  LocalReducer(sycl::handler& h, size_t size, C actual, sycl::buffer<U, 1>& b)
      : local(sycl::range<1>(size), h),
        actual(actual),
        result(b.template get_access<sycl::access::mode::read_write>(h)) {}

  inline void drain(sycl::id<1> lid, sycl::id<1> gid) const {
    local[lid] = result[gid];
  }
};

struct Range1D {
  const size_t from, to;
  const size_t size;
  template <typename A, typename B>
  Range1D(A from, B to) : from(from), to(to), size(to - from) {
    assert(from < to);
    assert(size != 0);
  }
  friend std::ostream& operator<<(std::ostream& os, const Range1D& d) {
    os << "Range1d{"
       << " X[" << d.from << "->" << d.to << " (" << d.size << ")]"
       << "}";
    return os;
  }
};

template <typename nameT,
          class LocalAllocator = std::nullptr_t,  //
          class Empty = std::nullptr_t,           //
          class Functor = std::nullptr_t,         //
          class BinaryOp = std::nullptr_t,        //
          class Finaliser = std::nullptr_t>
void parallel_reduce_1d(cl::sycl::queue& q,
                        const Range1D& range,      //
                        LocalAllocator allocator,  //
                        Empty empty,               //
                        Functor functor,           //
                        BinaryOp combiner,         //
                        Finaliser finaliser) {
  auto dev = q.get_device();
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

  const size_t n = range.size;
  dotNumGroups = std::min(n, dotNumGroups);

  q.submit([=](sycl::handler& h) mutable {
    auto ctx = allocator(h, dotWgsize);
    h.parallel_for<nameT>(
        sycl::nd_range<1>(dotNumGroups * dotWgsize, dotWgsize),
        [=](sycl::nd_item<1> item) {
          size_t globalId = item.get_global_id(0);
          size_t localId = item.get_local_id(0);
          size_t globalSize = item.get_global_range()[0];
          empty(ctx, sycl::id<1>(localId));
          for (; globalId < n; globalId += globalSize) {
            functor(ctx, sycl::id<1>(localId), range.from + globalId);
          }

          size_t localSize = item.get_local_range()[0];  // 8
          for (size_t offset = localSize / 2; offset > 0; offset /= 2) {
            item.barrier(sycl::access::fence_space::local_space);
            if (localId < offset) {
              combiner(ctx, sycl::id<1>(localId),
                       sycl::id<1>(localId + offset));
            }
          }
          if (localId == 0) {
            finaliser(ctx, item.get_group(0) * dotWgsize, sycl::id<1>(0));
          }
        });
  });

  q.submit([=](sycl::handler& h) mutable {
    auto ctx = allocator(h, dotNumGroups);
    h.parallel_for<class final_reduction>(sycl::nd_range<1>(1, 1), [=](auto) {
      auto zero = sycl::id<1>(0);
      empty(ctx, zero);  // local[0] = empty
      for (size_t i = 0; i < dotNumGroups; ++i) {
        ctx.drain(sycl::id<1>(i), sycl::id<1>(i * dotWgsize));
      }
      for (size_t i = 1; i < dotNumGroups; ++i) {
        combiner(ctx, zero, sycl::id<1>(i));
      }
      finaliser(ctx, 0, zero);  // xs[0] = local[0]
    });
  });
}
