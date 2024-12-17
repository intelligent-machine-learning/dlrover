// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/macro.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/nvidia/nvidia_dtype_util.h"
#include "xpu_timer/nvidia/nvidia_timer.h"

namespace xpu_timer {
namespace nvidia {

FaParser::FaParser(const std::string& library_path)
    : LibraryLoader(library_path) {
  const std::string err = "libparse_params.so, skip get fa shape";
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "parse_fwd_shape", get_fwd_shape_,
                                getShapeFn, err);
  SETUP_SYMBOL_FOR_LOAD_LIBRARY(handle_, "parse_bwd_shape", get_bwd_shape_,
                                getShapeFn, err);
  XLOG(INFO) << "Load fn `parse_bwd_shape` OK";
  can_use_ = true;
}
std::vector<uint64_t> FaParser::getFaFwdShape(void** args) {
  if (can_use_) return get_fwd_shape_(args);
  return {0, 0, 0, 0, 0};
}

std::vector<uint64_t> FaParser::getFaBwdShape(void** args) {
  if (can_use_) return get_bwd_shape_(args);
  return {0, 0, 0, 0, 0};
}

ptrdiff_t InterceptManager::getOffset(const void* symbol) {
  Dl_info info;
  if (dladdr(symbol, &info) != 0) {
    return (char*)symbol - (char*)info.dli_fbase;
  }
  return 0;
}

bool InterceptManager::isIntercepted(const void* func,
                                     const InterceptSymbol** sym) {
  if (fns_to_skip_.find(func) != fns_to_skip_.end()) {
    return false;
  }
  if (fns_to_name_.find(func) == fns_to_name_.end()) {
    ptrdiff_t offset = getOffset(func);
    auto it = addr_to_name_.find(offset);
    if (it != addr_to_name_.end()) {
      fns_to_name_[func] = &(it->second);
    } else {
      fns_to_skip_.insert(func);
      return false;
    }
  }
  *sym = fns_to_name_[func];
  return true;
}

std::vector<uint64_t> InterceptManager::getFaFwdShape(void** args) {
  return fa_parser_.getFaFwdShape(args);
}
std::vector<uint64_t> InterceptManager::getFaBwdShape(void** args) {
  return fa_parser_.getFaBwdShape(args);
}

std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
InterceptManager::handleNccl(
#if defined(CUDA_LAUNCH_EXC)
    const cudaLaunchConfig_t* config,
#else
    const void* config,  // not used
#endif
    const void* func, void** args, const InterceptSymbol* sym, bool* skip) {
  void* dev_comm = args[0];
  auto it = nccl_info_map_.find(dev_comm);
  // TODO add prometheus metric when not found.
  if (it == nccl_info_map_.end()) {
    // may be only tp
    *skip = skip_tp_;
    return cuda_launch_kernel_exc_default_;
  }

  if (it->second->empty()) {
    // if empty, the front do not push NcclInfo here
    *skip = skip_tp_;
    return cuda_launch_kernel_exc_default_;
  }
  auto nccl_info = it->second->front();
  it->second->pop();

#if defined(CUDA_LAUNCH_EXC)
  dim3 grid_dim = config->gridDim;
  dim3 block_dim = config->blockDim;
  auto fn = [sym, nccl_info, grid_dim,
             block_dim](NvidiaGpuTimer* timer) -> NvidiaGpuTimer::FnReturn {
#else
  auto fn = [sym,
             nccl_info](NvidiaGpuTimer* timer) -> NvidiaGpuTimer::FnReturn {
#endif
    timer->trace->Clear();
    timer->trace->set_kernel_type(constant::Metrics::CollMetrics::KERNEL_TYPE);
    hook::NcclDebugData* nccl_debug = timer->trace->mutable_nccl_debug();

#if defined(CUDA_LAUNCH_EXC)
    nccl_debug->add_grids(grid_dim.x);
    nccl_debug->add_grids(grid_dim.y);
    nccl_debug->add_grids(grid_dim.z);
    nccl_debug->add_blocks(block_dim.x);
    nccl_debug->add_blocks(block_dim.y);
    nccl_debug->add_blocks(block_dim.z);
#endif
    size_t count = nccl_info.count;
    auto comm = nccl_info.comm;
    ncclDataType_t datatype = nccl_info.datatype;
    std::string dtype = CudaDataTypeUtils::getNcclDataType(datatype);
    uint64_t comm_size = count * CudaDataTypeUtils::getDtypeSizeInBytes(dtype);

    std::ostringstream oss;
    oss << "xpu_timer_" << sym->func_name << "_size_" << comm_size;
    std::string coll_type = sym->coll_type;
    int nranks = comm->nRanks;
    double factor = 1.0;
    if (coll_type == "AllReduce") {
      factor = 2.0 * (nranks - 1) / nranks;
    } else if (coll_type == "AllGather" || coll_type == "ReduceScatter") {
      // input of reduce_scatter/allgather is sharded, so we do not device
      // world_size
      factor = static_cast<double>(nranks - 1);
    }
    uint64_t problem_size = static_cast<uint64_t>(factor * comm_size);

    nccl_debug->set_comm_hash(nccl_info.comm->commHash);
    nccl_debug->set_send_recv_type(static_cast<int>(nccl_info.send_recv_type));
    nccl_debug->set_input_size_in_bytes(comm_size);
    nccl_debug->set_dtype(dtype);
    nccl_debug->set_ranks(nranks);
    nccl_debug->set_nodes(comm->nNodes);
    nccl_debug->set_seq(++(timer->nccl_seq_num[nccl_info.comm->commHash]));
    nccl_debug->set_problem_size(problem_size);

    uint64_t problem_size_bits = problem_size * 8;

    return std::make_tuple(
        oss.str(), problem_size_bits,
        xpu_timer::Labels{
            {"dtype", dtype},
            {"operation", coll_type},
            {"algorithm", sym->algo},
            {"transport", comm->nNodes > 1 ? "InterNode" : "IntraNode"}});
  };
  return fn;
}

std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
InterceptManager::handleFa(void** args, const InterceptSymbol* sym) {
  std::vector<uint64_t> fa_params;
  if (sym->operation == "FaBwd")
    fa_params = getFaBwdShape(args);
  else if (sym->operation == "FaFwd")
    fa_params = getFaFwdShape(args);
  else
    fa_params = {0, 0, 0, 0, 0};

  auto fn = [sym,
             fa_params](NvidiaGpuTimer* timer) -> NvidiaGpuTimer::FnReturn {
    timer->trace->Clear();
    timer->trace->set_kernel_type(
        constant::Metrics::MatmulMetrics::KERNEL_TYPE);

    std::ostringstream oss;
    oss << "xpu_timer_" << sym->func_name << "_bssh_";

    hook::FaDebugData* fa_debug = timer->trace->mutable_fa_debug();
    for (int i = 0; i < 4; i++) {
      oss << fa_params[i] << "_";
      fa_debug->add_shapes(fa_params[i]);
    }
    uint64_t flops;

    // there are three backward kernels in flash attn
    // flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel
    // flash_bwd_dot_do_o_kernel
    // flash_bwd_convert_dq_kernel
    // only flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel has matmul
    if (sym->only_trace)
      flops = 0;
    else {
      flops = sym->operation == "FaFwd" ? 1 : 2;
      // 4bssh
      flops =
          flops * 4 * fa_params[0] * fa_params[1] * fa_params[2] * fa_params[3];
    }
    return std::make_tuple(
        oss.str(), flops,
        xpu_timer::Labels{{"dtype", sym->dtype}, {"operation", "FA"}});
  };
  return fn;
}

#if defined(CUDA_LAUNCH_EXC)
std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
InterceptManager::handleCudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                            const void* func, void** args,
                                            const InterceptSymbol* sym,
                                            bool* skip) {
  if (sym->func_type == "NCCL")
    return handleNccl(config, func, args, sym, skip);
  return cuda_launch_kernel_exc_default_;
}
#endif

std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
InterceptManager::handleCudaLaunchKernel(const void* func, dim3 gridDim,
                                         dim3 blockDim, void** args,
                                         size_t sharedMem, cudaStream_t stream,
                                         const InterceptSymbol* sym,
                                         bool* skip) {
  if (sym->func_type == "FA")
    return handleFa(args, sym);
  else if (sym->func_type == "NCCL")
    return handleNccl(nullptr, func, args, sym, skip);
  return cuda_launch_kernel_default_;
}
std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
InterceptManager::deviceMemory(const std::string& name, const size_t size,
                               const std::string& kind, bool is_host) {
  return [name, size, kind, is_host](NvidiaGpuTimer* timer) {
    timer->trace->Clear();
    timer->trace->set_kernel_type(
        xpu_timer::constant::Metrics::MemMetrics::KERNEL_TYPE);
    hook::MemoryDebugData* memory_debug = timer->trace->mutable_memory_debug();
    memory_debug->set_size(size);
    if (!kind.empty()) {
      memory_debug->set_direction(kind);
      return std::make_tuple(name, size,
                             xpu_timer::Labels{
                                 {"operation", name},
                                 {"kind", kind},
                             });
    }
    return std::make_tuple(name, size, xpu_timer::Labels{{"operation", name}});
  };
}

InterceptSymbol::InterceptSymbol(const InterceptSymbolPb& pb) {
  func_name = pb.func_name();
  coll_type = pb.coll_type();
  algo = pb.algo();
  operation = pb.operation();
  dtype = pb.dtype();
  func_type = pb.func_type();
  only_trace = pb.only_trace();
}

void InterceptManager::setUp() {
  resetSymsMap();
  skip_tp_ = util::EnvVarRegistry::GetEnvVar<bool>("XPU_TIMER_SKIP_TP");
  tp_size_ = util::EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_TP_SIZE");
}

void InterceptManager::resetSymsMap() {
  // reparse offset and kernel name of nccl lib
  std::filesystem::path file_path;
  auto path =
      util::EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_SYMS_FILE");
  if (path == util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
    Dl_info dl_info;
    void* symbol = (void*)::xpu_timer::util::REGISTER_ENV;
    if (dladdr(symbol, &dl_info) == 0) {
      XLOG(WARNING) << "Symsbol of xpuTimerNvidia is not found";
      return;
    }
    std::filesystem::path lib_path(dl_info.dli_fname);
    file_path = lib_path.parent_path() / "intercepted.sym.default";
  } else {
    file_path = path;
  }
  if (std::filesystem::exists(file_path) &&
      !std::filesystem::is_directory(file_path)) {
    std::ifstream file(file_path, std::ios::binary);
    XLOG(INFO) << "Symsbol file found " << file_path;
    if (!file.is_open()) {
      XLOG(WARNING)
          << "Symsbol file could not be opened, NCCL syms can not parse";
      return;
    }

    addr_to_name_.clear();
    ::xpu_timer::hook::InterceptSymbolByOffset intercepted_symbols;
    // TODO(lizhi) push error metric to prometheus
    intercepted_symbols.ParseFromIstream(&file);
    for (const auto& entry : intercepted_symbols.symbols()) {
      int64_t offset = entry.first;
      std::stringstream ss;
      ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << offset;
      const InterceptSymbolPb& symbol = entry.second;
      addr_to_name_.emplace(offset, symbol);
      XLOG(INFO) << "read symbols " << symbol.func_name() << " with type "
                 << symbol.dtype() << " with offset " << ss.str();
    }
  }
}

/*
 * ===================================
 * InterceptManager Static variables
 * ===================================
 */
std::unordered_map<void*,
                   std::shared_ptr<std::queue<InterceptManager::NcclInfo>>>
    InterceptManager::nccl_info_map_;

std::unordered_set<const void*> InterceptManager::fns_to_skip_;
std::unordered_map<const void*, const InterceptSymbol*>
    InterceptManager::fns_to_name_;
std::unordered_map<ptrdiff_t, const InterceptSymbol>
    InterceptManager::addr_to_name_;
std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
    InterceptManager::cuda_launch_kernel_exc_default_ = [](NvidiaGpuTimer*) {
      return std::make_tuple("CudaLaunchKernelExC_UNKNOWN", 0,
                             xpu_timer::Labels{});
    };
std::function<NvidiaGpuTimer::FnReturn(NvidiaGpuTimer*)>
    InterceptManager::cuda_launch_kernel_default_ = [](NvidiaGpuTimer*) {
      return std::make_tuple("CudaLaunchKernel_UNKNOWN", 0,
                             xpu_timer::Labels{});
    };
bool InterceptManager::skip_tp_{false};
int InterceptManager::tp_size_{0};
}  // namespace nvidia
}  // namespace xpu_timer
