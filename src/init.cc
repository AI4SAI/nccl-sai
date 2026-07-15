/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include "tuner.h"
#include "ras.h"
#include "profiler.h"
#include "mnnvl.h"
#include "sai_profile.h"
#include <algorithm>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include "param.h"
#include "nvtx_payload_schemas.h"
#include "utils.h"
#include <mutex>
#include "ce_coll.h"
#include "nvtx.h"
#include "env.h"

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CtaPolicy, "CTA_POLICY", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);

extern int64_t ncclParamSingleProcMemRegEnable();

static ncclResult_t commReclaim(ncclComm_t comm);

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192*1024)

static ncclResult_t setCpuStackSize() {
  if (ncclParamSetCpuStackSize() != 0) {
    // Query the stack size used for newly launched threads.
    pthread_attr_t attr;
    size_t stackSize;
    PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
    PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

    if (stackSize < SAFE_STACK_SIZE) {
      // GNU libc normally uses RLIMIT_STACK as the default pthread stack size, unless it's set to "unlimited" --
      // in that case a fallback value of 2MB (!) is used.

      // Query the actual resource limit so that we can distinguish between the settings of 2MB and unlimited.
      struct rlimit stackLimit;
      char buf[30];
      SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
      if (stackLimit.rlim_cur == RLIM_INFINITY)
        strcpy(buf, "unlimited");
      else
        snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur/1024);
      INFO(NCCL_INIT|NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
           buf, SAFE_STACK_SIZE/1024);

      // Change the default pthread stack size (via a nonportable API, which will become necessary if we switch
      // to C++ threads).
      PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
      PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
    }

    PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  }

  return ncclSuccess;
}

static ncclResult_t initResult = ncclSuccess;
static std::once_flag initOnceFlag;

static void initOnceFunc() {
  setCpuStackSize();
  initGdrCopy();
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);

  initNvtxRegisteredEnums();
exit:;
}

static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

static ncclResult_t envInitResult = ncclSuccess;
static std::once_flag envInitOnceFlag;

static void envInitOnceFunc() {
  NCCLCHECKGOTO(ncclEnvPluginInit(), envInitResult, exit);
exit:;
}

ncclResult_t ncclInitEnv() {
  std::call_once(envInitOnceFlag, envInitOnceFunc);
  return envInitResult;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInitEnv());
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  struct ncclBootstrapHandle handle;
  NCCLCHECK(bootstrapGetUniqueId(&handle));
  // ncclUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset(out, 0, sizeof(*out));
  // copy to avoid alignment mismatch
  memcpy(out, &handle, sizeof(handle));
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
  comm->startMagic = comm->endMagic = 0;
}

#undef NCCL_NO_OPTIMIZE


static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;
  /* commFree() should not involve any sync among ranks. */
  if (comm == NULL)
    return ncclSuccess;

  NCCLCHECK(ncclCeFinalize(comm));

  if (comm->symmetricSupport) {
    NCCLCHECK(ncclSymkFinalize(comm));
    NCCLCHECK(ncclDevrFinalize(comm));
  }
  NCCLCHECK(ncclRasCommFini(comm));

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
    PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
    if (comm->proxyState->threadUDS) {
      // UDS support
      PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
    }
  }

  if (comm->saiA2a.islandScratch != nullptr) {
    CUDACHECK(cudaFree(comm->saiA2a.islandScratch));
    comm->saiA2a.islandScratch = nullptr;
    comm->saiA2a.islandScratchBytes = 0;
  }
  if (comm->memPool) CUDACHECK(cudaMemPoolDestroy(comm->memPool));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

  free(comm->peerInfo);
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->saiA2a.nodeToFabricGroup);
  free(comm->saiA2a.fabricGroupCounts);
  free(comm->saiA2a.raggedRoundOrder);
  free(comm->saiA2a.raggedPhaseEnds);
  free(comm->collNetHeads);
  free(comm->clique.ranks);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));

  // GIN may use proxy. We need to finalize it before destroying the proxy.
  NCCLCHECK(ncclGinFinalize(comm));

  int sharedResRefCount = 0;
  if (comm->sharedRes) {
    sharedResRefCount = ncclAtomicRefCountDecrement(&comm->sharedRes->refCount);
    if (sharedResRefCount == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->launchEvent));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->scratchEvent));
      NCCLCHECK(ncclProxyDestroy(comm));
      free(comm->sharedRes);
    }
  }

  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  abort = *comm->abortFlag;
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    free(comm->abortFlag);
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  free(comm->gproxyConn);

  NCCLCHECK(ncclRegCleanup(comm));

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");

  commPoison(comm); // poison comm before free to avoid comm reuse.
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  if (sharedResRefCount == 0) NCCLCHECK(ncclNetFinalize(comm));
  ncclCudaContextDrop(comm->context);
  free(comm);

  return ncclSuccess;
}

NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<20)
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
enum ncclLaunchMode ncclParamLaunchMode;

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);

// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL || ncclCudaLibraryInit() != ncclSuccess) return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // Query device to see if DMA-BUF support is available
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret == ncclInProgress) {
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      ret = ncclInvalidArgument;
      goto exit;
    }
    /* if ret is not ncclInProgress, we just keep it. */
  }

exit:
  return ret;
}

static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  if (parent == NULL || !parent->shareResources) {
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));
    /* most of attributes are assigned later in initTransportsRank(). */
    sharedRes->owner = comm;
    sharedRes->tpNRanks = comm->nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));
    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;
    NCCLCHECK(ncclNetInit(comm));
  } else {
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
    NCCLCHECK(ncclNetInitFromParent(comm, parent));
  }

  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  if (parent && parent->shareResources) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }
  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  CUDACHECK(cudaGetDevice(&comm->cudaDev));

  NCCLCHECK(ncclCudaContextTrack(&comm->context));

  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  nvmlDevice_t nvmlDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));

  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);

  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

  do {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclKernelCommAndChannels tmpCommAndChans;
  struct ncclKernelCommAndChannels *devCommAndChans = NULL;
  struct ncclNvmlCCStatus ccStatus;
  bool ccEnable;
  cudaStream_t deviceStream;

  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  memset(&ccStatus, 0, sizeof(ccStatus));
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) && (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);
  if (ccEnable) {
    comm->workFifoBytes = 0;
  } else {
    comm->workFifoBytes = ncclParamWorkFifoBytes();
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
    }
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
  }

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);
  } else {
    // The workFifoBuf lives in cudaHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  comm->workFifoProduced = 0;
  comm->workFifoProducedLastRecorded = 0;
  comm->workFifoConsumed = 0;

  // Alloc profiler counters for the kernel
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
  }

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
    }
  }

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);
exit:
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    VERSION("%s", VERSION_STRING);
  } else {
    INFO(NCCL_ALL,"%s", VERSION_STRING);
  }
}

NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  cudaDeviceProp prop;
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->nvmlDev = comm->nvmlDev;
  NCCLCHECK(ncclGetVersion(&info->version));
  // Reject mixed upstream builds and mismatched NCCL-SAI init schemas before
  // exchanging the extended all-gather record below.
  info->version = (int)ncclSaiPeerVersion((uint32_t)info->version);
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;
  info->cuMemSupport = ncclCuMemEnable();
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  info->comm = comm;
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  // MNNVL support
  {
    // MNNVL: Request the fabric UUID and partition info
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    nvmlDevice_t nvmlDev;
    NCCLCHECK(int64ToBusId(info->busId, busId));
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);
    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      unsigned long uuid0 = 0;
      unsigned long uuid1 = 0;
      if (ncclParamMNNVLUUID() != -1) {
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }
      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));
      if (ncclParamMNNVLCliqueId() == -2) {
        nvmlPlatformInfo_t platformInfo = { 0 };
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex,
             platformInfo.hostId, platformInfo.peerType, platformInfo.moduleId);
        // Use a hash of the Rack serial number to partition the NVLD clique
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId,
           uuid0, uuid1,
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
  }

  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  if (comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();
  else if (comm->isAllNvlink) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  else comm->p2pChunkSize = ncclParamP2pPciChunkSize();

  // Make sure P2P chunksize is not larger than coll chunksize.
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE]) comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  if (comm->sharedRes->owner != comm) {
    /* make sure split comm p2pChunkSize won't exceed shared p2pChunkSize. */
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

extern int64_t ncclParamWinStride();

static ncclResult_t initNvlDomainInfo(struct ncclComm* comm) {
  // Initialize NVLink domain info
  comm->nvlDomainInfo.nNvlDomains = comm->nNodes;
  comm->nvlDomainInfo.minRanksPerNvlDomain = comm->minLocalRanks;
  comm->nvlDomainInfo.maxRanksPerNvlDomain = comm->maxLocalRanks;

  TRACE(NCCL_INIT, "NVLink domains: %d domains, min ranks per domain: %d, max ranks per domain: %d",
        comm->nNodes, comm->nvlDomainInfo.minRanksPerNvlDomain, comm->nvlDomainInfo.maxRanksPerNvlDomain);

  return ncclSuccess;
}

NCCL_PARAM(GroupSize, "P2P_SCHEDULE_GROUP_SIZE", NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
NCCL_PARAM(SaiP2pFabricGroupSchedule, "SAI_P2P_FABRIC_GROUP_SCHEDULE", -1);

static ncclResult_t ncclSaiRaggedShiftPhases(int nRankGroups, int nFabricGroups,
    const int* rankGroupToFabric, int** shiftOrderOut, uint8_t** phaseEndsOut,
    bool* useGreedyOrderOut, int* sumPhaseMaxOut, int* densestEdgeOut) {
  ncclResult_t ret = ncclSuccess;
  int *groupCounts = NULL, *shiftMax = NULL, *shiftCross = NULL;
  int *sortedShifts = NULL, *phaseSizes = NULL, *phaseLoads = NULL;
  int *phaseShifts = NULL, *shiftOrder = NULL;
  int *baselinePhaseLoad = NULL, *shiftLoads = NULL;
  uint8_t* phaseEnds = NULL;
  int edgeCount = 0, nPhases = 0;
  int fullPhases = 0, remainder = 0;
  int output = 0, sumPhaseMax = 0, densestEdge = 0;
  int baselineOutput = 0, baselinePhaseSize = 0, baselineScore = 0;
  int baselinePow2 = 0;
  size_t shiftLoadCount = 0;
  uint32_t baselineRound = 0, baselineDelta = 0;
  bool useGreedyOrder = false;
  if (shiftOrderOut == NULL || phaseEndsOut == NULL || useGreedyOrderOut == NULL ||
      rankGroupToFabric == NULL ||
      nRankGroups <= 0 || nFabricGroups <= 1 ||
      nFabricGroups > INT_MAX / nFabricGroups) return ncclInvalidArgument;
  *shiftOrderOut = NULL;
  *phaseEndsOut = NULL;
  *useGreedyOrderOut = false;

  edgeCount = nFabricGroups * nFabricGroups;
  nPhases = (nRankGroups + nFabricGroups - 1) / nFabricGroups;
  if ((size_t)nRankGroups > SIZE_MAX / (size_t)edgeCount ||
      (size_t)nPhases > SIZE_MAX / (size_t)edgeCount ||
      (size_t)nPhases > SIZE_MAX / (size_t)nFabricGroups) return ncclInvalidArgument;
  shiftLoadCount = (size_t)nRankGroups * edgeCount;
  NCCLCHECKGOTO(ncclCalloc(&groupCounts, nFabricGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&shiftLoads, shiftLoadCount), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&shiftMax, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&shiftCross, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sortedShifts, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&phaseSizes, nPhases), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&phaseLoads, (size_t)nPhases * edgeCount), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&phaseShifts, (size_t)nPhases * nFabricGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&shiftOrder, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&phaseEnds, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&baselinePhaseLoad, edgeCount), ret, fail);

  for (int group = 0; group < nRankGroups; group++) {
    int fabric = rankGroupToFabric[group];
    if (fabric < 0 || fabric >= nFabricGroups) {
      ret = ncclInvalidArgument;
      goto fail;
    }
    groupCounts[fabric]++;
  }
  for (int shift = 0; shift < nRankGroups; shift++) {
    sortedShifts[shift] = shift;
    int* load = shiftLoads + (size_t)shift * edgeCount;
    for (int source = 0; source < nRankGroups; source++) {
      int sourceFabric = rankGroupToFabric[source];
      int destinationFabric = rankGroupToFabric[(source + shift) % nRankGroups];
      if (sourceFabric == destinationFabric) continue;
      int edge = sourceFabric * nFabricGroups + destinationFabric;
      load[edge]++;
      shiftCross[shift]++;
      shiftMax[shift] = std::max(shiftMax[shift], load[edge]);
    }
  }
  std::sort(sortedShifts, sortedShifts + nRankGroups,
      [shiftMax, shiftCross](int left, int right) {
        if (shiftMax[left] != shiftMax[right]) return shiftMax[left] > shiftMax[right];
        if (shiftCross[left] != shiftCross[right]) return shiftCross[left] > shiftCross[right];
        return left < right;
      });

  fullPhases = nRankGroups / nFabricGroups;
  remainder = nRankGroups % nFabricGroups;
  for (int sorted = 0; sorted < nRankGroups; sorted++) {
    int shift = sortedShifts[sorted];
    int bestPhase = -1, bestIncrease = 0, bestMaximum = 0, bestSize = 0;
    uint64_t bestDeviation = 0;
    for (int phase = 0; phase < nPhases; phase++) {
      int capacity = phase < fullPhases ? nFabricGroups : remainder;
      if (capacity == 0) capacity = nFabricGroups;
      if (phaseSizes[phase] >= capacity) continue;
      int oldMaximum = 0, newMaximum = 0;
      uint64_t deviation = 0;
      int* current = phaseLoads + (size_t)phase * edgeCount;
      int* addition = shiftLoads + (size_t)shift * edgeCount;
      for (int sourceFabric = 0; sourceFabric < nFabricGroups; sourceFabric++) {
        for (int destinationFabric = 0; destinationFabric < nFabricGroups; destinationFabric++) {
          if (sourceFabric == destinationFabric) continue;
          int edge = sourceFabric * nFabricGroups + destinationFabric;
          int merged = current[edge] + addition[edge];
          oldMaximum = std::max(oldMaximum, current[edge]);
          newMaximum = std::max(newMaximum, merged);
          int64_t target = (int64_t)groupCounts[sourceFabric] * groupCounts[destinationFabric];
          int64_t delta = (int64_t)merged * nPhases - target;
          deviation += (uint64_t)(delta * delta);
        }
      }
      int increase = newMaximum - oldMaximum;
      bool better = bestPhase < 0 || increase < bestIncrease ||
          (increase == bestIncrease && newMaximum < bestMaximum) ||
          (increase == bestIncrease && newMaximum == bestMaximum && deviation < bestDeviation) ||
          (increase == bestIncrease && newMaximum == bestMaximum && deviation == bestDeviation &&
           phaseSizes[phase] < bestSize);
      if (better) {
        bestPhase = phase;
        bestIncrease = increase;
        bestMaximum = newMaximum;
        bestDeviation = deviation;
        bestSize = phaseSizes[phase];
      }
    }
    if (bestPhase < 0) {
      ret = ncclInternalError;
      goto fail;
    }
    int slot = phaseSizes[bestPhase]++;
    phaseShifts[(size_t)bestPhase * nFabricGroups + slot] = shift;
    int* destination = phaseLoads + (size_t)bestPhase * edgeCount;
    int* addition = shiftLoads + (size_t)shift * edgeCount;
    for (int edge = 0; edge < edgeCount; edge++) destination[edge] += addition[edge];
  }

  for (int sourceFabric = 0; sourceFabric < nFabricGroups; sourceFabric++) {
    for (int destinationFabric = 0; destinationFabric < nFabricGroups; destinationFabric++) {
      if (sourceFabric == destinationFabric) continue;
      densestEdge = std::max(densestEdge,
          groupCounts[sourceFabric] * groupCounts[destinationFabric]);
    }
  }
  for (int phase = 0; phase < nPhases; phase++) {
    int phaseMaximum = 0;
    int* load = phaseLoads + (size_t)phase * edgeCount;
    for (int edge = 0; edge < edgeCount; edge++) phaseMaximum = std::max(phaseMaximum, load[edge]);
    sumPhaseMax += phaseMaximum;
    for (int slot = 0; slot < phaseSizes[phase]; slot++) {
      shiftOrder[output++] = phaseShifts[(size_t)phase * nFabricGroups + slot];
    }
    if (output > 0) phaseEnds[output - 1] = 1;
  }
  if (output != nRankGroups) {
    ret = ncclInternalError;
    goto fail;
  }
  baselinePow2 = pow2Up(nRankGroups);
  do {
    if (baselineDelta < (uint32_t)nRankGroups) {
      sortedShifts[baselineOutput++] = (int)baselineDelta;
      for (int source = 0; source < nRankGroups; source++) {
        int sourceFabric = rankGroupToFabric[source];
        int destinationFabric = rankGroupToFabric[(source + baselineDelta) % nRankGroups];
        if (sourceFabric != destinationFabric) {
          baselinePhaseLoad[sourceFabric * nFabricGroups + destinationFabric]++;
        }
      }
      baselinePhaseSize++;
      if (baselinePhaseSize == nFabricGroups || baselineOutput == nRankGroups) {
        int phaseMaximum = 0;
        for (int edge = 0; edge < edgeCount; edge++) {
          phaseMaximum = std::max(phaseMaximum, baselinePhaseLoad[edge]);
          baselinePhaseLoad[edge] = 0;
        }
        baselineScore += phaseMaximum;
        baselinePhaseSize = 0;
      }
    }
    baselineRound++;
    baselineDelta = (baselineDelta + baselineRound) & (baselinePow2 - 1);
  } while (baselineRound != (uint32_t)baselinePow2);
  useGreedyOrder = sumPhaseMax < baselineScore;
  if (!useGreedyOrder) {
    memcpy(shiftOrder, sortedShifts, (size_t)nRankGroups * sizeof(int));
    memset(phaseEnds, 0, (size_t)nRankGroups * sizeof(uint8_t));
    for (int begin = 0; begin < nRankGroups; begin += nFabricGroups) {
      phaseEnds[std::min(nRankGroups, begin + nFabricGroups) - 1] = 1;
    }
    sumPhaseMax = baselineScore;
  }
  *shiftOrderOut = shiftOrder;
  *phaseEndsOut = phaseEnds;
  *useGreedyOrderOut = useGreedyOrder;
  if (sumPhaseMaxOut != NULL) *sumPhaseMaxOut = sumPhaseMax;
  if (densestEdgeOut != NULL) *densestEdgeOut = densestEdge;
  shiftOrder = NULL;
  phaseEnds = NULL;

fail:
  free(groupCounts);
  free(shiftLoads);
  free(shiftMax);
  free(shiftCross);
  free(sortedShifts);
  free(phaseSizes);
  free(phaseLoads);
  free(phaseShifts);
  free(baselinePhaseLoad);
  free(shiftOrder);
  free(phaseEnds);
  return ret;
}

static ncclResult_t ncclSaiBuildRaggedRoundOrder(struct ncclComm* comm,
    int nRankGroups, int rankGroupSize, const int* groupToNode) {
  ncclResult_t ret = ncclSuccess;
  int *rankGroupToFabric = NULL, *shiftOrder = NULL, *shiftToNativeBlock = NULL;
  int* roundOrder = NULL;
  uint8_t *shiftPhaseEnds = NULL, *phaseEnds = NULL, *seenRounds = NULL;
  bool useGreedyOrder = false;
  int sumPhaseMax = 0, densestEdge = 0;
  int nativeBlock = 0, logicalRound = 0;

  if (comm == NULL || groupToNode == NULL || nRankGroups <= 0 || rankGroupSize <= 0 ||
      nRankGroups > INT_MAX / rankGroupSize ||
      nRankGroups * rankGroupSize != comm->nRanks || comm->saiA2a.nFabricGroups <= 1 ||
      comm->saiA2a.nodeToFabricGroup == NULL) return ncclInternalError;

  NCCLCHECKGOTO(ncclCalloc(&rankGroupToFabric, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&shiftToNativeBlock, nRankGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&roundOrder, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&phaseEnds, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&seenRounds, comm->nRanks), ret, fail);

  for (int group = 0; group < nRankGroups; group++) {
    int node = groupToNode[group];
    if (node < 0 || node >= comm->nNodes) {
      ret = ncclInternalError;
      goto fail;
    }
    int fabric = comm->saiA2a.nodeToFabricGroup[node];
    if (fabric < 0 || fabric >= comm->saiA2a.nFabricGroups) {
      ret = ncclInternalError;
      goto fail;
    }
    rankGroupToFabric[group] = fabric;
    shiftToNativeBlock[group] = -1;
  }

  {
    int nRankGroupsPow2 = pow2Up(nRankGroups);
    uint32_t nativeRound = 0, nativeShift = 0;
    do {
      if (nativeShift < (uint32_t)nRankGroups) {
        shiftToNativeBlock[nativeShift] = nativeBlock++;
      }
      nativeRound++;
      nativeShift = (nativeShift + nativeRound) & (nRankGroupsPow2 - 1);
    } while (nativeRound != (uint32_t)nRankGroupsPow2);
  }
  if (nativeBlock != nRankGroups) {
    ret = ncclInternalError;
    goto fail;
  }

  NCCLCHECKGOTO(ncclSaiRaggedShiftPhases(
      nRankGroups, comm->saiA2a.nFabricGroups, rankGroupToFabric,
      &shiftOrder, &shiftPhaseEnds, &useGreedyOrder,
      &sumPhaseMax, &densestEdge), ret, fail);

  for (int localDelta = 0; localDelta < rankGroupSize; localDelta++) {
    for (int shiftIndex = 0; shiftIndex < nRankGroups; shiftIndex++) {
      int shift = shiftOrder[shiftIndex];
      if (shift < 0 || shift >= nRankGroups || shiftToNativeBlock[shift] < 0) {
        ret = ncclInternalError;
        goto fail;
      }
      int channelRound = shiftToNativeBlock[shift] * rankGroupSize + localDelta;
      if (channelRound < 0 || channelRound >= comm->nRanks || seenRounds[channelRound]) {
        ret = ncclInternalError;
        goto fail;
      }
      seenRounds[channelRound] = 1;
      roundOrder[logicalRound] = channelRound;
      phaseEnds[logicalRound] = shiftPhaseEnds[shiftIndex];
      logicalRound++;
    }
  }
  if (logicalRound != comm->nRanks) {
    ret = ncclInternalError;
    goto fail;
  }

  comm->saiA2a.raggedRoundOrder = roundOrder;
  comm->saiA2a.raggedPhaseEnds = phaseEnds;
  comm->saiA2a.raggedScheduleReady = true;
  roundOrder = NULL;
  phaseEnds = NULL;
  if (comm->rank == 0) {
    INFO(NCCL_GRAPH,
        "%s: using SAI ragged AlltoAll round map, topologyNodes %d fabricGroups %d maxNodesPerGroup %d order %s phaseEdgeScore %d lowerBound %d",
        __func__, comm->nNodes, comm->saiA2a.nFabricGroups,
        comm->saiA2a.maxFabricGroupNodes,
        useGreedyOrder ? "native-greedy" : "native-quadratic",
        sumPhaseMax, densestEdge);
  }

fail:
  if (ret != ncclSuccess && comm != NULL && comm->rank == 0) {
    INFO(NCCL_GRAPH,
        "%s: SAI ragged AlltoAll round map construction failed, error %d",
        __func__, ret);
  }
  free(rankGroupToFabric);
  free(shiftOrder);
  free(shiftToNativeBlock);
  free(roundOrder);
  free(shiftPhaseEnds);
  free(phaseEnds);
  free(seenRounds);
  return ret;
}

static int ncclP2pScheduleGroupSize(struct ncclComm* comm, int64_t configuredGroupSize) {
  if (comm->nNodes == 1) return comm->maxLocalRanks;
  if (configuredGroupSize <= 0 || configuredGroupSize > INT_MAX) return 0;
  int groupSize = (int)configuredGroupSize;
  for (int node = 0; node < comm->nNodes; node++) {
    int localRanks = comm->nodeRanks[node].localRanks;
    if (localRanks % groupSize != 0 || localRanks < groupSize) groupSize = gcd(groupSize, localRanks);
  }
  return groupSize;
}

static ncclResult_t ncclSaiP2pFabricGroupOrder(struct ncclComm* comm,
    int groupSize, int groupsPerNode, int nGroups,
    int* groupToNode, int* groupToLocal) {
  ncclResult_t ret = ncclSuccess;
  int *groupedNodes = NULL, *groupedLocals = NULL;
  int *fabricOffsets = NULL, *fabricNext = NULL, *rankGroupsPerFabric = NULL;
  int groupCount = 0;
  NCCLCHECKGOTO(ncclCalloc(&groupedNodes, nGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupedLocals, nGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&fabricOffsets, comm->saiA2a.nFabricGroups + 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&fabricNext, comm->saiA2a.nFabricGroups), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&rankGroupsPerFabric, comm->saiA2a.nFabricGroups), ret, fail);
  for (int fabricGroup = 0; fabricGroup < comm->saiA2a.nFabricGroups; fabricGroup++) {
    rankGroupsPerFabric[fabricGroup] =
        comm->saiA2a.fabricGroupCounts[fabricGroup] * groupsPerNode;
    fabricOffsets[fabricGroup + 1] =
        fabricOffsets[fabricGroup] + rankGroupsPerFabric[fabricGroup];
  }
  if (fabricOffsets[comm->saiA2a.nFabricGroups] != nGroups) {
    WARN("SAI fabric group rank-count mismatch: %d vs %d",
        fabricOffsets[comm->saiA2a.nFabricGroups], nGroups);
    ret = ncclInternalError;
    goto fail;
  }
  for (int n = 0; n < comm->nNodes; ++n) {
    if (comm->nodeRanks[n].localRanks % groupSize != 0) {
      WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d",
          groupSize, n, comm->nodeRanks[n].localRanks);
      ret = ncclInternalError;
      goto fail;
    }
    int fabricGroup = comm->saiA2a.nodeToFabricGroup[n];
    if (fabricGroup < 0 || fabricGroup >= comm->saiA2a.nFabricGroups) {
      ret = ncclInternalError;
      goto fail;
    }
    int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;
    for (int g = 0; g < nGroupsInNode; ++g) {
      int index = fabricOffsets[fabricGroup] + fabricNext[fabricGroup]++;
      if (index < 0 || index >= nGroups) {
        ret = ncclInternalError;
        goto fail;
      }
      groupedLocals[index] = g * groupSize;
      groupedNodes[index] = n;
    }
  }
  for (int index = 0; index < nGroups; index++) {
    groupToNode[groupCount] = groupedNodes[index];
    groupToLocal[groupCount] = groupedLocals[index];
    groupCount++;
  }

fail:
  free(groupedNodes);
  free(groupedLocals);
  free(fabricOffsets);
  free(fabricNext);
  free(rankGroupsPerFabric);
  return ret;
}

static bool ncclSaiRaggedScheduleCandidate(struct ncclComm* comm, int groupSize) {
  int64_t groupNodesParam =
      comm->saiA2a.config.field[ncclSaiA2aConfigGroupNodes];
  int64_t islandSize =
      comm->saiA2a.config.field[ncclSaiA2aConfigIslandSize];
  if (!comm->saiA2a.configConsistent ||
      comm->saiA2a.config.field[ncclSaiA2aConfigEnabled] == 0 ||
      comm->saiA2a.config.field[ncclSaiA2aConfigPlannerEnable] == 0 ||
      groupNodesParam <= 0 || groupNodesParam > INT_MAX ||
      islandSize <= 0 || islandSize > INT_MAX ||
      comm->saiA2a.nodeToFabricGroup == nullptr ||
      comm->saiA2a.fabricGroupCounts == nullptr ||
      !ncclSaiA2aRaggedLayoutAllowed(
          comm->nNodes, (int)groupNodesParam,
          comm->saiA2a.config.field[ncclSaiA2aConfigMultigroupEnable],
          comm->saiA2a.fabricMetadataValid, comm->saiA2a.fabricGroupsComplete,
          comm->saiA2a.nFabricGroups, comm->saiA2a.maxFabricGroupNodes)) return false;
  for (int n = 0; n < comm->nNodes; n++) {
    if (comm->nodeRanks[n].localRanks != comm->nodeRanks[0].localRanks ||
        comm->nodeRanks[n].localRanks % groupSize != 0) return false;
  }
  return comm->nodeRanks[0].localRanks == islandSize && groupSize == islandSize;
}

static ncclResult_t ncclP2pSchedule(struct ncclComm* comm, int groupSize) {
  struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
  if (groupSize <= 0) return ncclInvalidUsage;
  comm->p2pSchedGroupSize = groupSize;

  int local = comm->localRank % groupSize; // local id inside my group
  int localGroup = comm->localRank / groupSize;
  int group = -1;
  int nGroups = comm->nRanks / groupSize;
  int nGroupsPow2 = pow2Up(nGroups);

  int64_t saiGroupNodesParam = comm->saiA2a.config.field[ncclSaiA2aConfigGroupNodes];
  bool validSaiGroupNodes = saiGroupNodesParam > 0 && saiGroupNodesParam <= INT_MAX;
  int saiGroupNodes = validSaiGroupNodes ? (int)saiGroupNodesParam : 0;
  int groupsPerNode = nodeRanks[0].localRanks / groupSize;
  bool uniformLocalRanks = groupsPerNode > 0;
  for (int n = 0; n < comm->nNodes; n++) {
    uniformLocalRanks &= nodeRanks[n].localRanks == nodeRanks[0].localRanks &&
        nodeRanks[n].localRanks % groupSize == 0;
  }
  bool saiScheduleRequested = comm->saiA2a.configConsistent &&
      ncclSaiP2pFabricScheduleRequested(
          comm->saiA2a.config.field[ncclSaiA2aConfigEnabled] != 0,
          comm->saiA2a.config.field[ncclSaiA2aConfigP2pFabricSchedule] != 0);
  bool saiCompleteScheduleAllowed = validSaiGroupNodes && ncclSaiP2pFabricScheduleAllowed(
      comm->nNodes, saiGroupNodes,
      comm->saiA2a.config.field[ncclSaiA2aConfigMultigroupEnable],
      comm->saiA2a.fabricGroupsComplete);
  bool saiRaggedScheduleAllowed = validSaiGroupNodes && ncclSaiA2aRaggedLayoutAllowed(
      comm->nNodes, saiGroupNodes,
      comm->saiA2a.config.field[ncclSaiA2aConfigMultigroupEnable],
      comm->saiA2a.fabricMetadataValid, comm->saiA2a.fabricGroupsComplete,
      comm->saiA2a.nFabricGroups, comm->saiA2a.maxFabricGroupNodes);
  bool saiSchedule = saiScheduleRequested && saiCompleteScheduleAllowed &&
      uniformLocalRanks && comm->saiA2a.nodeToFabricGroup != nullptr &&
      comm->saiA2a.fabricGroupCounts != nullptr;
  int64_t saiIslandSize =
      comm->saiA2a.config.field[ncclSaiA2aConfigIslandSize];
  bool completeIslandRankGroups = saiIslandSize > 0 && saiIslandSize <= INT_MAX &&
      nodeRanks[0].localRanks == saiIslandSize && groupSize == saiIslandSize;
  bool saiRaggedSchedule = ncclSaiRaggedScheduleCandidate(comm, groupSize);
  bool useFabricMetadataOrder = saiSchedule;

  int *groupToNode = NULL, *groupToLocal = NULL;
  ncclResult_t ret = ncclCalloc(&groupToNode, nGroups); // node hosting the group
  if (ret != ncclSuccess) return ret;
  ret = ncclCalloc(&groupToLocal, nGroups); // local offset of the group
  if (ret != ncclSuccess) {
    free(groupToNode);
    return ret;
  }
  int groupCount = 0;
  if (useFabricMetadataOrder) {
    ret = ncclSaiP2pFabricGroupOrder(
        comm, groupSize, groupsPerNode, nGroups, groupToNode, groupToLocal);
    if (ret != ncclSuccess) {
      free(groupToNode);
      free(groupToLocal);
      return ret;
    }
    groupCount = nGroups;
  } else {
    for (int n = 0; n < comm->nNodes; ++n) {
      if (0 != comm->nodeRanks[n].localRanks % groupSize) {
        WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d", groupSize, n, comm->nodeRanks[n].localRanks);
        free(groupToNode); free(groupToLocal);
        return ncclInternalError;
      }
      int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;
      for (int g = 0; g < nGroupsInNode; ++g) {
        groupToLocal[groupCount] = g * groupSize;
        groupToNode[groupCount] = n;
        groupCount++;
      }
    }
  }
  for (int candidate = 0; candidate < groupCount; candidate++) {
    if (groupToNode[candidate] == comm->node &&
        groupToLocal[candidate] == localGroup * groupSize) group = candidate;
  }
  if (groupCount != nGroups || group < 0) {
    WARN("Group creation failed: count %d vs %d localGroup %d", groupCount, nGroups, group);
    free(groupToNode);
    free(groupToLocal);
    return ncclInternalError;
  }
  INFO(NCCL_GRAPH,"%s: group size used is %d",__func__,groupSize);

  int round = 0;
  if (saiSchedule) {
    int groupsPerFabric = saiGroupNodes * groupsPerNode;
    if (uniformLocalRanks && groupsPerFabric > 0 && nGroups % groupsPerFabric == 0) {
      int nFabricGroups = nGroups / groupsPerFabric;
      int nFabricGroupsPow2 = pow2Up(nFabricGroups);
      INFO(NCCL_GRAPH, "%s: using SAI fabric group schedule, groupNodes %d, groupsPerFabric %d, fabricGroups %d",
           __func__, saiGroupNodes, groupsPerFabric, nFabricGroups);
      for (int delta = 0; delta < groupSize; delta++) {
        for (int groupSkew = 0; groupSkew < groupsPerFabric; groupSkew++) {
          uint32_t fabricRound = 0, fabricDelta = 0;
          do {
            if (fabricDelta < nFabricGroups) {
              int groupDelta = fabricDelta * groupsPerFabric + groupSkew;
              int sendGroup = (group + groupDelta) % nGroups;
              int recvGroup = (group - groupDelta + nGroups) % nGroups;
              int sendNode = groupToNode[sendGroup];
              int recvNode = groupToNode[recvGroup];
              int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
              int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
              comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
              comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
              round += 1;
            }
            fabricRound += 1;
            fabricDelta = (fabricDelta + fabricRound) & (nFabricGroupsPow2 - 1);
          } while (fabricRound != nFabricGroupsPow2);
        }
      }
    } else {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled, localRanks/group layout is not uniform", __func__);
    }
  } else if (saiScheduleRequested && !saiRaggedSchedule) {
    if (!validSaiGroupNodes) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled, invalid groupNodes %ld",
           __func__, (long)saiGroupNodesParam);
    } else if (!ncclSaiA2aMultigroupAllowed(
        comm->saiA2a.config.field[ncclSaiA2aConfigMultigroupEnable],
        comm->saiA2a.fabricGroupsComplete)) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled by multigroup policy", __func__);
    } else if (!comm->saiA2a.fabricMetadataValid ||
        comm->saiA2a.nodeToFabricGroup == nullptr) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled, fabric-group metadata is unavailable", __func__);
    } else if (!uniformLocalRanks) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled, local rank groups are not uniform", __func__);
    } else if (saiRaggedScheduleAllowed && !completeIslandRankGroups) {
      INFO(NCCL_GRAPH, "%s: SAI ragged fabric schedule disabled, topology nodes are not complete configured islands", __func__);
    } else if (comm->nNodes < saiGroupNodes) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled below one configured group", __func__);
    } else if (comm->saiA2a.maxFabricGroupNodes > saiGroupNodes) {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled, observed group occupancy exceeds configured capacity", __func__);
    } else {
      INFO(NCCL_GRAPH, "%s: SAI fabric group schedule disabled for this fabric layout", __func__);
    }
  }
  if (round == 0) {
    uint32_t groupRound = 0, groupDelta = 0;
  // When enumerating peer deltas we use the quadratic formula (x*x+x)/2 mod N.
  // Since that formula only produces valid permutations when N is a pow of 2,
  // we let N = pow2Up(n) and filter out results greater-eq to n.
  // Example sequence for 16 ranks: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
    do {
      if (groupDelta < nGroups) { // Filter nonsensical group deltas
        int sendGroup = (group + groupDelta) % nGroups;
        int recvGroup = (group - groupDelta + nGroups) % nGroups;
        int sendNode = groupToNode[sendGroup];
        int recvNode = groupToNode[recvGroup];
        for (int delta = 0; delta < groupSize; delta++) {
          int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
          int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
          comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
          comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
          round += 1;
        }
      }
      groupRound += 1;
      groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1); // Quadratic update
    } while (groupRound != nGroupsPow2);
  }

  if (round != comm->nRanks) {
    WARN("P2p schedule creation has bugs.");
    free(groupToNode);
    free(groupToLocal);
    return ncclInternalError;
  }
  if (saiRaggedSchedule) {
    ncclResult_t mapResult = ncclSaiBuildRaggedRoundOrder(
        comm, nGroups, groupSize, groupToNode);
    if (mapResult != ncclSuccess) {
      free(groupToNode);
      free(groupToLocal);
      return mapResult;
    }
  }

  free(groupToNode);
  free(groupToLocal);
  return ncclSuccess;
}

static bool ncclSaiA2aAllocIslandScratch(struct ncclComm* comm,
    const struct ncclSaiA2aConfig* config, size_t* reserveBytesOut, void** scratchOut) {
  if (reserveBytesOut != nullptr) *reserveBytesOut = 0;
  if (scratchOut != nullptr) *scratchOut = nullptr;
  if (comm == nullptr || config == nullptr || scratchOut == nullptr) return false;

  size_t reserveBytes = ncclSaiA2aIslandScratchReserveBytes(config, comm->nRanks);
  if (reserveBytesOut != nullptr) *reserveBytesOut = reserveBytes;
  if (reserveBytes == 0) return true;

  cudaError_t status = cudaMalloc(scratchOut, reserveBytes);
  if (status != cudaSuccess) (void)cudaGetLastError();
  return status == cudaSuccess;
}

static ncclResult_t ncclBuildCommTopology(
    struct ncclComm* comm, struct ncclSaiLocalP2pInfo saiLocalP2pInfo[8]) {
  const int rank = comm->rank;
  const int nranks = comm->nRanks;
  bool localP2pEnabled = false;
  uint64_t localP2pRankPairs = 0;

  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // Keep collective graph search on the upstream NET baseline. The agreed
  // local P2P rank-pair mask is applied only after topology trimming and is
  // consumed by graph-less P2P transport selection.
  ncclTopoSaiSetLocalP2pSys(comm->topo, false, 0);
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  if (nranks == 8) {
    memset(saiLocalP2pInfo, 0, sizeof(*saiLocalP2pInfo) * 8);
    NCCLCHECK(ncclTopoSaiGetLocalP2pInfo(
        comm, comm->topo, saiLocalP2pInfo+rank));
    NCCLCHECK(bootstrapAllGather(
        comm->bootstrap, saiLocalP2pInfo, sizeof(*saiLocalP2pInfo)));
    bool policyConsistent = true;
    bool topologyConsistent = true;
    bool allEligible = true;
    int eligibleRanks = 0;
    for (int r = 0; r < nranks; r++) {
      if (saiLocalP2pInfo[r].eligible != 0) eligibleRanks++;
      else allEligible = false;
      if (saiLocalP2pInfo[r].policySignature !=
          saiLocalP2pInfo[0].policySignature) policyConsistent = false;
      if (saiLocalP2pInfo[r].topologyClass !=
          saiLocalP2pInfo[0].topologyClass ||
          saiLocalP2pInfo[r].rankPairs !=
          saiLocalP2pInfo[0].rankPairs) topologyConsistent = false;
    }
    int consensus = ncclSaiResolveLocalP2pConsensus(
        policyConsistent, allEligible, topologyConsistent);
    if (consensus == ncclSaiLocalP2pConsensusReject) {
      if (rank == 0) WARN("NCCL-SAI local P2P policy differs across ranks");
      return ncclInvalidUsage;
    }
    localP2pEnabled = consensus == ncclSaiLocalP2pConsensusEnabled;
    if (localP2pEnabled) localP2pRankPairs = saiLocalP2pInfo[0].rankPairs;
    if (rank == 0) {
      INFO(NCCL_INIT|NCCL_P2P,
        "NCCL-SAI automatic local P2P policy: enabled %d eligibleRanks %d/%d",
        localP2pEnabled ? 1 : 0, eligibleRanks, nranks);
    }
  }
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  ncclTopoSaiSetLocalP2pSys(
      comm->topo, localP2pEnabled, localP2pRankPairs);
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // We use two upstream mandatory AllGathers. NCCL-SAI adds fixed-size policy
  // votes only after the ordinary peer record has passed its schema check.
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;
  cpu_set_t affinitySave;
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN];
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT];
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  struct graphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
    int crossNic;
  };

  struct raggedBuildStatus {
    int result;
    int ready;
  };

  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
    struct ncclSaiA2aConfig saiA2aConfig;
    struct ncclSaiFabricGroupInfo saiFabricGroup;
    uint64_t saiIslandScratchReserveBytes;
    int64_t p2pScheduleGroupSize;
    int saiIslandScratchReady;
    int cpuArch;
    int cpuVendor;
    int localRanks;
  };

  struct ncclSaiLocalP2pInfo saiLocalP2pInfo[8];
  int nChannelsOrig;
  struct allGatherInfo *allGather3Data = NULL;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  uint64_t *saiNodeGroupIds = NULL, *saiGroupIds = NULL;
  int *saiGroupCounts = NULL;
  bool *saiNodeGroupSet = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int p2pLevel = -1;
  int p2pScheduleGroupSize = 0;
  bool raggedBuildStatusRequired = false;
  bool saiAutomaticConfigFallback = false;
  bool saiConfigMismatchFallback = false;
  bool saiExplicitConfigConsistent = true;
  bool saiIbPluginConsistent = true;
  bool saiIbPolicyConsistent = true;
  bool saiIbModeConsistent = true;
  bool saiIbAllLayoutsEligible = true;
  int saiIbInternalPlugin = 0;
  int saiIbEndpointMode = ncclSaiIbEndpointMergeUpstream;
  int saiIbEndpointConsensus = ncclSaiIbEndpointConsensusUpstream;
  int saiFabricValidRanks = 0, saiFabricAbsentRanks = 0, saiFabricInvalidRanks = 0;
  int saiIbPreflightEligibleRanks = 0;
  int saiRailEligibleRanks = 0;
  struct ncclSaiIbEndpointPolicyInfo* saiIbEndpointInfo = nullptr;
  struct ncclSaiRailInfo* saiRailInfo = nullptr;
  uint64_t saiIbEndpointGatherStart = 0;
  size_t saiIslandScratchReserveBytes = 0;
  void* saiIslandScratch = nullptr;

  timers[TIMER_INIT_ALLGATHER] = clockNano();
  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);
  __atomic_store_n(&comm->peerInfoValid, true, __ATOMIC_RELEASE);

  comm->cuMemSupport = 1;
  comm->saiA2a.spansMultiplePhysicalHosts = false;
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) {
      nNodes++;
      comm->saiA2a.spansMultiplePhysicalHosts = true;
    }
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // The ordinary peer record stays ABI-compatible through the schema check
  // above. Exchange the NCCL-SAI endpoint policy only after mixed builds have
  // already been rejected safely.
  NCCLCHECKGOTO(ncclCalloc(&saiIbEndpointInfo, nranks), ret, fail);
  if (comm->ncclNet == &ncclNetIb) {
    NCCLCHECKGOTO(ncclIbGetSaiEndpointPolicyInfo(
        saiIbEndpointInfo+rank), ret, fail);
    saiIbEndpointInfo[rank].internalIbPlugin = 1;
  } else {
    saiIbEndpointInfo[rank].mode = ncclSaiIbEndpointMergeUpstream;
  }
  saiIbEndpointGatherStart = clockNano();
  NCCLCHECKGOTO(bootstrapAllGather(
      comm->bootstrap, saiIbEndpointInfo, sizeof(*saiIbEndpointInfo)), ret, fail);
  timers[TIMER_INIT_ALLGATHER] += clockNano() - saiIbEndpointGatherStart;

  saiIbPluginConsistent = true;
  saiIbPolicyConsistent = true;
  saiIbModeConsistent = true;
  saiIbAllLayoutsEligible = true;
  saiIbInternalPlugin = saiIbEndpointInfo[0].internalIbPlugin;
  saiIbEndpointMode = saiIbEndpointInfo[0].mode;
  if (saiIbInternalPlugin != 0 && saiIbInternalPlugin != 1) {
    saiIbPluginConsistent = false;
  }
  if (saiIbEndpointMode < ncclSaiIbEndpointMergeExplicitDisabled ||
      saiIbEndpointMode > ncclSaiIbEndpointMergeAutomatic) {
    saiIbModeConsistent = false;
  }
  for (int r = 0; r < nranks; r++) {
    const struct ncclSaiIbEndpointPolicyInfo* peer = saiIbEndpointInfo+r;
    if ((peer->internalIbPlugin != 0 && peer->internalIbPlugin != 1) ||
        peer->internalIbPlugin != saiIbInternalPlugin) {
      saiIbPluginConsistent = false;
    }
    if (peer->policySignature != saiIbEndpointInfo[0].policySignature) {
      saiIbPolicyConsistent = false;
    }
    if (peer->mode != saiIbEndpointMode) {
      saiIbModeConsistent = false;
    }
    if (peer->layoutEligible != 0) {
      saiIbPreflightEligibleRanks++;
    } else {
      saiIbAllLayoutsEligible = false;
    }
  }
  saiIbEndpointConsensus = ncclSaiResolveIbEndpointConsensus(
      saiIbPluginConsistent, saiIbPolicyConsistent, saiIbModeConsistent,
      saiIbInternalPlugin != 0, saiIbEndpointMode,
      saiIbAllLayoutsEligible);
  if (saiIbEndpointConsensus == ncclSaiIbEndpointConsensusReject) {
    if (rank == 0) {
      WARN("NCCL-SAI IB endpoint policy or NET plugin differs across ranks");
    }
    ret = ncclInvalidUsage;
    goto fail;
  }
  comm->saiIbAutoPreservePhysicalEndpoints =
      saiIbEndpointConsensus == ncclSaiIbEndpointConsensusPreserve;
  if (rank == 0 && saiIbInternalPlugin) {
    INFO(NCCL_INIT|NCCL_ENV,
      "NCCL-SAI automatic IB endpoint probe: preserve %d eligibleRanks %d/%d mode %d",
      comm->saiIbAutoPreservePhysicalEndpoints ? 1 : 0,
      saiIbPreflightEligibleRanks, nranks, saiIbEndpointMode);
  }

  // Check for MNNVL support
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail);
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;

    comm->nvlsRegSupport = 1;
    for (int i = 0; i < nranks; i++) {
      comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
      comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
          (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      if (comm->nvlsRegSupport) {
        for (int j = i + 1; j < nranks; j++) {
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
            comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0;
            break;
          }
        }
      }
    }

    // Buffer Registration is not supported with MNNVL
    if (comm->MNNVL) comm->nvlsRegSupport = 0;
    else if (ncclParamSingleProcMemRegEnable()) comm->nvlsRegSupport = 1;

    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  timers[TIMER_INIT_TOPO] = clockNano();

  // Dump XML if requested by user
  const char* dumpXmlFile;
  dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
  if (dumpXmlFile) {
    NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);
  }

  // Build a physical-endpoint probe only after every rank agrees that the
  // automatic preflight is eligible. The full topology vote below either
  // accepts that probe or rebuilds every rank with upstream NIC fusion.
  NCCLCHECKGOTO(ncclBuildCommTopology(comm, saiLocalP2pInfo), ret, fail);
  if (saiIbInternalPlugin) {
    NCCLCHECKGOTO(ncclCalloc(&saiRailInfo, nranks), ret, fail);
    NCCLCHECKGOTO(ncclTopoSaiGetRailInfo(
        comm, comm->topo, saiRailInfo+rank), ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(
        comm->bootstrap, saiRailInfo, sizeof(*saiRailInfo)), ret, fail);

    bool railEnabled = ncclSaiRailInfoConsensus(
        saiRailInfo, nranks, &saiRailEligibleRanks);

    if (comm->saiIbAutoPreservePhysicalEndpoints && !railEnabled) {
      if (rank == 0) {
        INFO(NCCL_INIT|NCCL_ENV,
          "NCCL-SAI physical-endpoint probe rejected by the full topology "
          "vote; rebuilding with upstream NIC fusion");
      }
      comm->saiIbAutoPreservePhysicalEndpoints = false;
      ncclTopoFree(comm->topo);
      comm->topo = nullptr;
      NCCLCHECKGOTO(ncclBuildCommTopology(comm, saiLocalP2pInfo), ret, fail);
      ncclTopoSaiSetRailByChannel(comm->topo, false);
    } else {
      ncclTopoSaiSetRailByChannel(comm->topo, railEnabled);
    }

    if (rank == 0) {
      INFO(NCCL_INIT|NCCL_ENV,
        "NCCL-SAI automatic dual-rail policy: enabled %d eligibleRanks %d/%d "
        "port1Subnet 0x%lx port2Subnet 0x%lx",
        ncclTopoSaiRailByChannelEnabled(comm->topo) ? 1 : 0,
        saiRailEligibleRanks, nranks,
        (unsigned long)saiRailInfo[0].railSubnet[0],
        (unsigned long)saiRailInfo[0].railSubnet[1]);
    }
  } else {
    ncclTopoSaiSetRailByChannel(comm->topo, false);
  }
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);
  // Decide on comm's CPU architecture.
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  // Determine local CollNet support
  if (!collNetSupport(comm)) {
    comm->config.collnetEnable = 0;
  }

  // Determine local Nvls support
  NCCLCHECKGOTO(ncclNvlsInit(comm), ret, fail);

  timers[TIMER_INIT_GRAPHS] = clockNano();
  // Get rings and trees
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0;
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph->minChannels = 1;
  ringGraph->maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1;
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;
  collNetChainGraph->collNet = 1;
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;
  if (comm->config.collnetEnable) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }
  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];

  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // Because timers[[TIMER_INIT_ALLGATHER] already contains the timing of the first allgather,
  // we temporarily store the start time of the subsequent one in an as-of-yet unused CONNECT timer.
  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;
  ncclSaiA2aGetConfig(comm, &allGather3Data[rank].saiA2aConfig);
  ncclSaiA2aGetFabricGroupInfo(&allGather3Data[rank].saiFabricGroup);
  allGather3Data[rank].p2pScheduleGroupSize = ncclParamGroupSize();
  saiIslandScratchReserveBytes = 0;
  allGather3Data[rank].saiIslandScratchReady = ncclSaiA2aAllocIslandScratch(
      comm, &allGather3Data[rank].saiA2aConfig, &saiIslandScratchReserveBytes,
      &saiIslandScratch) ? 1 : 0;
  allGather3Data[rank].saiIslandScratchReserveBytes = saiIslandScratchReserveBytes;

  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  for (int r = 1; r < nranks; r++) {
    if (allGather3Data[0].saiA2aConfig.field[
            ncclSaiA2aConfigExplicitControlsPresent] !=
            allGather3Data[r].saiA2aConfig.field[
            ncclSaiA2aConfigExplicitControlsPresent] ||
        allGather3Data[0].saiA2aConfig.field[
            ncclSaiA2aConfigExplicitControlsSignature] !=
            allGather3Data[r].saiA2aConfig.field[
            ncclSaiA2aConfigExplicitControlsSignature]) {
      saiExplicitConfigConsistent = false;
      break;
    }
  }
  if (!saiExplicitConfigConsistent) {
    WARN("NCCL-SAI explicit configuration differs across ranks");
    ret = ncclInvalidUsage;
    goto fail;
  }
  // NCCL_SAI_DISABLE participates in the explicit-control signature above, so
  // this rank-local read is communicator-consistent and remains frozen after
  // initialization even if the process environment changes later.
  comm->saiA2a.globallyDisabled = ncclSaiGloballyDisabled();

  comm->saiA2a.configConsistent = true;
  for (int r = 1; r < nranks; r++) {
    if (memcmp(allGather3Data[0].saiA2aConfig.field, allGather3Data[r].saiA2aConfig.field,
        sizeof(allGather3Data[0].saiA2aConfig.field)) != 0) {
      comm->saiA2a.configConsistent = false;
      break;
    }
  }
  if (!comm->saiA2a.configConsistent) {
    // Rank-local automatic detection is an intersection, not an assertion.
    // Any disagreement disables the SAI schedule everywhere while preserving
    // communicator initialization and upstream behavior.
    if (rank == 0) {
      INFO(NCCL_INIT|NCCL_ENV,
        "NCCL-SAI automatic AlltoAll policy disabled: rank-local capabilities differ");
    }
    saiAutomaticConfigFallback = true;
    saiConfigMismatchFallback = true;
    comm->saiA2a.configConsistent = true;
    for (int r = 0; r < nranks; r++) {
      allGather3Data[r].saiIslandScratchReady = 0;
    }
  }
  if (!saiConfigMismatchFallback && comm->saiA2a.configConsistent &&
      !ncclTopoSaiRailByChannelEnabled(comm->topo) &&
      allGather3Data[0].saiA2aConfig.field[
          ncclSaiA2aConfigAutomaticPolicyMask] != 0) {
    if (rank == 0) {
      INFO(NCCL_INIT|NCCL_ENV,
        "NCCL-SAI automatic fabric policies disabled: dual-rail policy is unavailable");
    }
    saiAutomaticConfigFallback = true;
    int64_t automaticPolicyMask = allGather3Data[0].saiA2aConfig.field[
        ncclSaiA2aConfigAutomaticPolicyMask];
    if ((automaticPolicyMask &
        (ncclSaiA2aAutomaticBase | ncclSaiA2aAutomaticIsland)) != 0) {
      for (int r = 0; r < nranks; r++) {
        allGather3Data[r].saiIslandScratchReady = 0;
      }
    }
  }
  comm->saiA2a.config = saiConfigMismatchFallback ?
      allGather3Data[0].saiA2aConfig :
      allGather3Data[rank].saiA2aConfig;
  if (saiConfigMismatchFallback) {
    ncclSaiA2aApplyCanonicalFallback(&comm->saiA2a.config);
  } else if (saiAutomaticConfigFallback) {
    ncclSaiA2aApplyAutomaticFallback(&comm->saiA2a.config);
  }
  comm->saiA2a.islandScratchReady = true;
  for (int r = 0; r < nranks; r++) {
    if (allGather3Data[r].saiIslandScratchReady == 0 ||
        allGather3Data[r].saiIslandScratchReserveBytes != saiIslandScratchReserveBytes) {
      comm->saiA2a.islandScratchReady = false;
      break;
    }
  }
  if (comm->saiA2a.islandScratchReady && saiIslandScratchReserveBytes != 0) {
    comm->saiA2a.islandScratch = saiIslandScratch;
    comm->saiA2a.islandScratchBytes = saiIslandScratchReserveBytes;
    saiIslandScratch = nullptr;
  } else if (saiIslandScratch != nullptr) {
    (void)cudaFree(saiIslandScratch);
    saiIslandScratch = nullptr;
  }
  if (rank == 0 && allGather3Data[rank].saiIslandScratchReserveBytes != 0) {
    INFO(NCCL_INIT|NCCL_ENV,
      "NCCL-SAI island scratch reserve: bytes %llu ready %d",
      (unsigned long long)allGather3Data[rank].saiIslandScratchReserveBytes,
      comm->saiA2a.islandScratchReady ? 1 : 0);
  }

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[r] = node;

    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    if (comm->cpuVendor != allGather3Data[r].cpuVendor &&
        comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
  }

  comm->saiA2a.fabricMetadataValid = false;
  comm->saiA2a.fabricGroupsComplete = false;
  comm->saiA2a.raggedScheduleReady = false;
  comm->saiA2a.nFabricGroups = 0;
  comm->saiA2a.maxFabricGroupNodes = 0;
  saiFabricValidRanks = saiFabricAbsentRanks = saiFabricInvalidRanks = 0;
  for (int r = 0; r < nranks; r++) {
    switch (allGather3Data[r].saiFabricGroup.state) {
    case ncclSaiFabricGroupIdValid: saiFabricValidRanks++; break;
    case ncclSaiFabricGroupIdAbsent: saiFabricAbsentRanks++; break;
    default: saiFabricInvalidRanks++; break;
    }
  }
  if (saiFabricValidRanks == nranks) {
    NCCLCHECKGOTO(ncclCalloc(&saiNodeGroupIds, comm->nNodes), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&saiNodeGroupSet, comm->nNodes), ret, fail);
    bool nodeMetadataConsistent = true;
    for (int r = 0; r < nranks; r++) {
      int node = comm->rankToNode[r];
      uint64_t groupId = allGather3Data[r].saiFabricGroup.id;
      if (!saiNodeGroupSet[node]) {
        saiNodeGroupSet[node] = true;
        saiNodeGroupIds[node] = groupId;
      } else if (saiNodeGroupIds[node] != groupId) {
        nodeMetadataConsistent = false;
        break;
      }
    }
    if (nodeMetadataConsistent) {
      NCCLCHECKGOTO(ncclCalloc(&saiGroupIds, comm->nNodes), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&saiGroupCounts, comm->nNodes), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&comm->saiA2a.nodeToFabricGroup, comm->nNodes), ret, fail);
      for (int node = 0; node < comm->nNodes; node++) {
        int fabricGroup = 0;
        while (fabricGroup < comm->saiA2a.nFabricGroups &&
            saiGroupIds[fabricGroup] != saiNodeGroupIds[node]) fabricGroup++;
        if (fabricGroup == comm->saiA2a.nFabricGroups) {
          saiGroupIds[fabricGroup] = saiNodeGroupIds[node];
          comm->saiA2a.nFabricGroups++;
        }
        comm->saiA2a.nodeToFabricGroup[node] = fabricGroup;
        saiGroupCounts[fabricGroup]++;
      }
      comm->saiA2a.fabricMetadataValid = true;
      int64_t expectedNodes = comm->saiA2a.config.field[ncclSaiA2aConfigGroupNodes];
      bool groupsComplete = expectedNodes > 0 && expectedNodes <= INT_MAX &&
          comm->nNodes == comm->saiA2a.nFabricGroups * expectedNodes;
      for (int group = 0; group < comm->saiA2a.nFabricGroups; group++) {
        groupsComplete &= saiGroupCounts[group] == expectedNodes;
        comm->saiA2a.maxFabricGroupNodes = std::max(
            comm->saiA2a.maxFabricGroupNodes, saiGroupCounts[group]);
      }
      comm->saiA2a.fabricGroupsComplete = groupsComplete;
      comm->saiA2a.fabricGroupCounts = saiGroupCounts;
      saiGroupCounts = NULL;
      if (rank == 0) {
        INFO(NCCL_INIT|NCCL_ENV,
          "NCCL-SAI fabric metadata: groups %d topologyNodes %d expectedNodesPerGroup %ld maxObservedNodesPerGroup %d complete %d",
          comm->saiA2a.nFabricGroups, comm->nNodes, (long)expectedNodes,
          comm->saiA2a.maxFabricGroupNodes, groupsComplete ? 1 : 0);
      }
    } else if (rank == 0) {
      INFO(NCCL_INIT|NCCL_ENV,
        "NCCL-SAI fabric metadata ignored: ranks in one topology node reported different group IDs");
    }
  } else if ((saiFabricValidRanks != 0 || saiFabricInvalidRanks != 0) && rank == 0) {
    INFO(NCCL_INIT|NCCL_ENV,
      "NCCL-SAI fabric metadata ignored: validRanks %d absentRanks %d invalidRanks %d",
      saiFabricValidRanks, saiFabricAbsentRanks, saiFabricInvalidRanks);
  }
  free(saiNodeGroupIds); saiNodeGroupIds = NULL;
  free(saiNodeGroupSet); saiNodeGroupSet = NULL;
  free(saiGroupIds); saiGroupIds = NULL;
  free(saiGroupCounts); saiGroupCounts = NULL;

  // Alert the user to the presence of mixed CPUs. In the past this has caused
  // locks in some collective routines. This may help debug issues in the future.
  if (rank==0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  comm->minLocalRanks = INT_MAX;
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  p2pScheduleGroupSize = ncclP2pScheduleGroupSize(
      comm, allGather3Data[rank].p2pScheduleGroupSize);
  if (comm->saiA2a.config.field[ncclSaiA2aConfigEnabled] != 0) {
    if (p2pScheduleGroupSize <= 0) {
      WARN("NCCL-SAI invalid effective P2P schedule group size");
      ret = ncclInvalidUsage;
      goto fail;
    }
    for (int r = 0; r < nranks; r++) {
      if (ncclP2pScheduleGroupSize(
          comm, allGather3Data[r].p2pScheduleGroupSize) != p2pScheduleGroupSize) {
        WARN("NCCL-SAI effective P2P schedule group size differs across ranks");
        ret = ncclInvalidUsage;
        goto fail;
      }
    }
  }
  raggedBuildStatusRequired =
      ncclSaiRaggedScheduleCandidate(comm, p2pScheduleGroupSize);

  NCCLCHECKGOTO(initNvlDomainInfo(comm), ret, fail);

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d",
       comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
    comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
  }
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;

  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0;
    }
  }
  NCCLCHECKGOTO(ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink), ret, fail);
  comm->isOneRPN = (comm->maxLocalRanks == 1);

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret, fail);
  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler plugin context has to be initialized before proxy thread
  NCCLCHECKGOTO(ncclProfilerPluginInit(comm), ret, fail);

  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->isAllDirectP2p, &comm->directMode, &comm->isAllCudaP2p), ret, fail);
  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->shareResources) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();
  // Build p2p schedule
  comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, comm->nRanks);
  comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, comm->nRanks);
  {
    ncclResult_t scheduleResult = ncclP2pSchedule(comm, p2pScheduleGroupSize);
    if (!raggedBuildStatusRequired) {
      if (scheduleResult != ncclSuccess) {
        ret = scheduleResult;
        goto fail;
      }
    } else {
      struct raggedBuildStatus* status =
          reinterpret_cast<struct raggedBuildStatus*>(allGather3Data);
      status[rank].result = (int)scheduleResult;
      status[rank].ready = scheduleResult == ncclSuccess &&
          comm->saiA2a.raggedScheduleReady &&
          comm->saiA2a.raggedRoundOrder != NULL &&
          comm->saiA2a.raggedPhaseEnds != NULL;
      uint64_t statusStart = clockNano();
      NCCLCHECKGOTO(bootstrapAllGather(
          comm->bootstrap, status, sizeof(*status)), ret, fail);
      uint64_t statusElapsed = clockNano() - statusStart;
      timers[TIMER_INIT_ALLGATHER] += statusElapsed;
      timers[TIMER_INIT_CONNECT] += statusElapsed;
      for (int peer = 0; peer < nranks; peer++) {
        if (status[peer].result != (int)ncclSuccess) {
          ret = (ncclResult_t)status[peer].result;
          goto fail;
        }
        if (status[peer].ready == 0) {
          ret = ncclInternalError;
          goto fail;
        }
      }
    }
  }

  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (comm->runtimeConn) {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    // Attempt to setup NVLS, may silently fail and disable NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // Check if we can setup CollNet
    if (comm->config.collnetEnable) ncclCollNetSetup(comm, parent, graphs);
  } else {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // Connect Trees
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // Connect PAT only for communicators with 1 GPU per node
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // Attempt to setup NVLS, may silently fail and disable NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // And NVLS trees if needed
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // Check if we can setup CollNet
    if (comm->config.collnetEnable) {
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY+1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // Connect to local net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // Then to remote ones when using PXN
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    if (ncclParamNvbPreconnect()) {
      // Connect p2p when using NVB path
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r=0; r<nvbNpeers; r++) {
        int peer = nvbPeers[r];
        int sendRound=0, recvRound=0;
        while (comm->p2pSchedule[sendRound].sendRank != peer) sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer) recvRound++;
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1UL<<channelId);
          }
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1UL<<channelId);
          }
        }
      }

      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoInitTunerConstants(comm), ret, fail);
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail);
  if (comm->tuner) {
    NCCLCHECKGOTO(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog, &comm->nvlDomainInfo, &comm->tunerConstants), ret, fail);
  }
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable();
  comm->devrState.bigSize = 0;

  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() -  timers[TIMER_INIT_CONNECT];
  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (saiIslandScratch != nullptr) (void)cudaFree(saiIslandScratch);
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable()) ncclProxyShmUnlink(comm);
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(saiNodeGroupIds);
  free(saiNodeGroupSet);
  free(saiGroupIds);
  free(saiGroupCounts);
  free(saiIbEndpointInfo);
  free(saiRailInfo);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlinkUtilCentricSchedEnable, "NVLINK_UTIL_CENTRIC_SCHED_ENABLE", NCCL_CONFIG_UNDEF_INT);


#define NCCL_COMMINIT_FUNCNAME_LEN 128
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  int splitCount;
  // For Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // name of the function calling
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

typedef struct{
  int key;
  int color;
} commSplitInfo;
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // we assume excludeRanksList is sorted
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  int maxSharedMem = 0;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;

  timers[TIMER_INIT_TOTAL] = clockNano();
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  cudaArch = 100*archMajor + 10*archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  if (job->parent) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    if (job->excludeRanksCount) {
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
      // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
      if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    }
    // child hash obtained from (parent hash, split count, color)
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->splitCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d- Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info, no commId was used
    commIdHash = 0;
  } else {
    // obtain a unique hash using the first commId
    comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // update communicator state
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
  } else {
    // the name for the replay tool is ncclCommInitRank for all the variations
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);
  return res;
fail:
  comm->initState = res;
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  const char* collnetEnableEnv;
  int ctaPolicyEnv;
  int shrinkShareEnv;
  int nvlsCTAsEnv;
  int nChannelsPerNetPeerEnv;
  int nvlinkUtilCentricSchedEnableEnv;

  /* override configuration with env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    if (comm->config.cgaClusterSize != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=%d", cgaClusterSizeEnv);
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    INFO(NCCL_ENV, "NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    else {
      if (comm->config.minCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config minCTAs reset to NCCL_MIN_CTAS=%d", minCTAsEnv);
      comm->config.minCTAs = minCTAsEnv;
    }
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    else {
      if (comm->config.maxCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config maxCTAs reset to NCCL_MAX_CTAS=%d", maxCTAsEnv);
      comm->config.maxCTAs = maxCTAsEnv;
    }
  }

  /* override configuration with env variable. */
  nChannelsPerNetPeerEnv = ncclParamNChannelsPerNetPeer();
  if (nChannelsPerNetPeerEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nChannelsPerNetPeerEnv <= 0)
      INFO(NCCL_ENV, "NCCL_NCHANNELS_PER_NET_PEER %d is too low, leaving it set at %d", nChannelsPerNetPeerEnv, comm->config.nChannelsPerNetPeer);
    else {
      if (comm->config.nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nChannelsPerNetPeer reset to NCCL_NCHANNELS_PER_NET_PEER=%d", nChannelsPerNetPeerEnv);
      comm->config.nChannelsPerNetPeer = nChannelsPerNetPeerEnv;
    }
  }

  nvlinkUtilCentricSchedEnableEnv = ncclParamNvlinkUtilCentricSchedEnable();
  if (nvlinkUtilCentricSchedEnableEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nvlinkUtilCentricSchedEnableEnv != 0 && nvlinkUtilCentricSchedEnableEnv != 1)
      INFO(NCCL_ENV, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE %d is not valid, leaving it set at %d", nvlinkUtilCentricSchedEnableEnv, comm->config.nvlinkCentricSched);
    else {
      if (comm->config.nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=%d", nvlinkUtilCentricSchedEnableEnv);
      comm->config.nvlinkCentricSched = nvlinkUtilCentricSchedEnableEnv;
    }
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR)
      INFO(NCCL_ENV, "Comm config netName reset to NCCL_NET=%s", tmpNetName);
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.splitShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config splitShare reset to NCCL_COMM_SPLIT_SHARE_RESOURCES=%d", splitShareEnv);
    comm->config.splitShare = splitShareEnv;
  }
  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.shrinkShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config shrinkShare reset to NCCL_COMM_SHRINK_SHARE_RESOURCES=%d", shrinkShareEnv);
    comm->config.shrinkShare = shrinkShareEnv;
  }

  // NCCL_COLLNET_ENABLE needs to be reloaded each time for comm init
  // since users might change the env on the fly to enable/disable collnet
  collnetEnableEnv = ncclGetEnv("NCCL_COLLNET_ENABLE");
  if (collnetEnableEnv != NULL) {
    int collnetEnableInt = (int)strtol(collnetEnableEnv, NULL, 0);
    if (collnetEnableInt != NCCL_CONFIG_UNDEF_INT) {
      if (comm->config.collnetEnable != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config collnetEnable reset to NCCL_COLLNET_ENABLE=%d", collnetEnableInt);
      comm->config.collnetEnable = collnetEnableInt;
      INFO(NCCL_ENV, "NCCL_COLLNET_ENABLE set by environment to %d.", collnetEnableInt);
    }
  }

  ctaPolicyEnv = ncclParamCtaPolicy();
  if (ctaPolicyEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.CTAPolicy != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config CTAPolicy reset to NCCL_CTA_POLICY=%d", ctaPolicyEnv);
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  nvlsCTAsEnv = ncclParamNvlsChannels();
  if (nvlsCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=%d", nvlsCTAsEnv);
    comm->config.nvlsCTAs = nvlsCTAsEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    INFO(NCCL_ENV, "splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.collnetEnable != 1 && comm->config.collnetEnable != 0) {
    INFO(NCCL_ENV, "collnetEnable %d is not a valid value 0/1, set it to 0", comm->config.collnetEnable);
    comm->config.collnetEnable = 0;
  }

  if (comm->config.CTAPolicy < NCCL_CTA_POLICY_DEFAULT || comm->config.CTAPolicy > NCCL_CTA_POLICY_ZERO) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, NCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT && comm->config.nvlsCTAs <= 0) {
    INFO(NCCL_ENV, "nvlsCTAs %d is not a valid value, NCCL will decide the default value automatically", comm->config.nvlsCTAs);
    comm->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
  }

  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parnet) {
  memcpy(&childComm->config, &parnet->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* check version. */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 25, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 27, 0)) {
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
    if (internalConfigPtr->version < NCCL_VERSION(2, 28, 0)) {
      internalConfigPtr->nChannelsPerNetPeer = defaultConfig.nChannelsPerNetPeer;
      internalConfigPtr->nvlinkCentricSched = defaultConfig.nvlinkCentricSched;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT &&
    internalConfigPtr->minCTAs <= 0) ||
    (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT &&
      internalConfigPtr->maxCTAs <= 0) ||
    (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs, internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->collnetEnable != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->collnetEnable < 0 || internalConfigPtr->collnetEnable > 1)) {
    WARN("Invalid config collnetEnable attribute value %d", internalConfigPtr->collnetEnable);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->CTAPolicy < NCCL_CTA_POLICY_DEFAULT ||
    internalConfigPtr->CTAPolicy > NCCL_CTA_POLICY_ZERO)) {
    WARN("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->shrinkShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 && internalConfigPtr->shrinkShare != 1) {
    WARN("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlsCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlsCTAs <= 0) {
    WARN("Invalid config nvlsCTAs attribute value %d", internalConfigPtr->nvlsCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->nChannelsPerNetPeer <= 0 || internalConfigPtr->nChannelsPerNetPeer > MAXCHANNELS)) {
    WARN("Invalid config nChannelsPerNetPeer attribute value %d", internalConfigPtr->nChannelsPerNetPeer);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlinkCentricSched != 0 && internalConfigPtr->nvlinkCentricSched != 1) {
    WARN("Invalid config nvlinkCentricSched attribute value %d", internalConfigPtr->nvlinkCentricSched);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, trafficClass, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "Traffic class", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, NCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, collnetEnable, NCCL_CONFIG_UNDEF_INT, 0, "Collnet enable", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, NCCL_CONFIG_UNDEF_INT, NCCL_CTA_POLICY_DEFAULT, "CTA policy flags", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, NCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlsCTAs, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "nvlsCTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nChannelsPerNetPeer, NCCL_CONFIG_UNDEF_INT,
                      NCCL_CONFIG_UNDEF_INT, "nChannelsPerNetPeer", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlinkCentricSched, NCCL_CONFIG_UNDEF_INT, 0, "nvlinkCentricSched", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.trafficClass = internalConfigPtr->trafficClass;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.collnetEnable = internalConfigPtr->collnetEnable;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;
  comm->config.nvlsCTAs = internalConfigPtr->nvlsCTAs;
  comm->config.nChannelsPerNetPeer = internalConfigPtr->nChannelsPerNetPeer;
  comm->config.nvlinkCentricSched = internalConfigPtr->nvlinkCentricSched;
  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  free(job->commId);
  free(_job);
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument;
  }
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob* job = NULL;
  bool launchedJob = false;
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), res, fail);

  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static std::once_flag once;
    std::call_once(once, showVersion);
  }
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInProgress and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInProgress;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECK(cudaGetDevice(&cudaDev));

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));

  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  NVTX3_RANGE(NcclNvtxParamsCommInitAll);

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* duplicate device check. */
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms+i, ndev,1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  NVTX3_RANGE_ADD_PAYLOAD(CommInitAll, NcclNvtxParamsCommInitAllSchema,
    NVTX3_PAYLOAD(comms[0]->commHash, ndev));

exit:
  (void)cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;

  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankConfig);

  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankScalable);

  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d\n", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d\n", comm, comm->rank, ret);
    }

    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, true), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
    // And keep polling until all graphs referencing us die.
    while (comm->localPersistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->finalize(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }
  NCCLCHECK(commFree(comm));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommFinalize);

  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, NULL, free, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommFinalize, NcclNvtxParamsCommFinalizeSchema,
      NVTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* this is  the last call to ncclCommDestroy/Abort, we need to make sure all comms
       * in the process have been finalized before we free local resources. */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NCCL_NVTX3_FUNC_RANGE;
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_FUNC_WITH_PARAMS(CommDestroy, NcclNvtxParamsCommInitRank,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  NCCLCHECK(ncclGroupStartInternal());
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // Set abort flags
  if (comm->childAbortFlag != nullptr) {
    __atomic_store_n(comm->childAbortFlag, value, __ATOMIC_RELEASE);
    __atomic_store_n(comm->childAbortFlagDev, value, __ATOMIC_RELEASE);
  }
  __atomic_store_n(comm->abortFlag, value, __ATOMIC_RELEASE);
  __atomic_store_n(comm->abortFlagDev, value, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRevoke, ncclComm_t comm, int revokeFlags);
struct ncclCommRevokeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t commRevokeAsync(struct ncclAsyncJob* job_) {
  struct ncclCommRevokeAsyncJob* job = (struct ncclCommRevokeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(PtrCheck(comm, "CommRevokeAsync", "comm"), res, exit);
  INFO(NCCL_INIT, "CommRevokeAsync START comm %p rank %d nRanks %d nNodes %d localRank %d cudaDev %d",
      comm, comm->rank, comm->nRanks, comm->nNodes, comm->localRank, comm->cudaDev);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
  NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/true), res, exit);
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), res, exit);
  {
    ncclResult_t _tmpret = ncclSuccess;
    if ((_tmpret = ncclProxyStop(comm)) != ncclSuccess) {
      WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, _tmpret);
    }
    if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
      PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
      if (comm->proxyState->threadUDS) {
        // UDS support
        PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
      }
      // Mark threads as joined so later cleanup (e.g., commFree) won't join again
      comm->proxyState->thread = 0;
      comm->proxyState->threadUDS = 0;
    }
  }
  NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
exit:
  (void)ncclCommSetAsyncError(comm, res);
  INFO(NCCL_INIT, "CommRevokeAsync END comm %p result %d", comm, res);
  return res;
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommRevoke);

  if (comm == NULL) {
    return ncclSuccess;
  }
  // For now only NCCL_REVOKE_DEFAULT (0) is supported
  if (revokeFlags != NCCL_REVOKE_DEFAULT) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if destroy/finalize in progress
  if (comm->destroyFlag || comm->finalizeCalled) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if revoke in progress
  if (comm->revokedFlag) {
    return ncclInvalidArgument;
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  (void)setCommAbortFlags(comm,1);
  comm->revokedFlag = 1;
  (void)ncclCommEnsureReady(comm);
  comm->finalizeCalled = true;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommRevokeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commRevokeAsync, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &res));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke COMPLETE, result %d", comm, rank, nranks, cudaDev, comm->busId, res);
  return res;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, res);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommAbort);

  if (comm == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Abort START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  // Ask anything that might still be running on the device to quit
  NCCLCHECK(setCommAbortFlags(comm,1));
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommAbort, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  free(job);
}

// initializing a child communicator (for both split and shrink)
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color, int key, int* excludeRanksList, int excludeRanksCount,
                                          ncclConfig_t* config, const char* caller) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);
  if (isShrink) {
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList may not be sorted, need to sort it
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // ranks in excludeRanksList should not call into this function
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ? ncclInvalidArgument : ncclSuccess, res, exit);
  }
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);

  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (!isShrink && color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;

    // Set the shareResource field, this is used throughout the init and must be reset every time.
    // Never share resources if the parent communicator has been revoked.
    // If we shrink, we only reuse resources in default mode.
    comm->shareResources = !comm->revokedFlag && (isShrink ? (!(flags & NCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare);
    if (comm->shareResources) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  if (excludeRanksList) {
    // need to copy the list of ranks to exclude because the job is async
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // each split has to lead to a unique comm, so increment the splitCount
    job->splitCount = ++comm->splitCount;
    job->excludeRanksList = NULL;
  }
  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", caller);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/NULL, /*destructor=*/childCommCleanupJob, comm), res, fail);

exit:
  (void)cudaSetDevice(oldDev);
  return res;
fail:
  if (childComm) {
    if (!comm->shareResources) {
      if (childComm->abortFlag) free(childComm->abortFlag);
      if (childComm->abortFlagDev) ncclCudaHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t* config, int shrinkFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  // Handle error mode by setting abort flags and waiting for kernels to complete and unset the flags to avoid bootstrap issues
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank, excludeRanksList, excludeRanksCount, config, __func__), res, exit);

  if (*newcomm) NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema, NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  NVTX3_RANGE(NcclNvtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/NCCL_SHRINK_DEFAULT, color, key, NULL, 0, config, __func__), res, exit);

  if (*newcomm)
    NVTX3_RANGE_ADD_PAYLOAD(CommSplit, NcclNvtxParamsCommSplitSchema, NVTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, color, key));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);

  /* Check gin status */
  if (*asyncError == ncclSuccess && comm->sharedRes && comm->sharedRes->ginState.ncclGin) {
    struct ncclGinState* ginState = &comm->sharedRes->ginState;
    // Gin progress thread status
    if (ginState->needsProxyProgress) *asyncError = __atomic_load_n(&comm->sharedRes->ginState.asyncResult, __ATOMIC_ACQUIRE);
    // Gin side errors, also works when we have no GIN progress thread.
    if (*asyncError == ncclSuccess) {
      bool ginError;
      for (int c=0; c<comm->sharedRes->ginState.ginCommCount; c++) {
        NCCLCHECK(ncclGinQueryLastError(&comm->sharedRes->ginState, &ginError));
        if (ginError) {
          WARN("GIN Error on gin context %d\n", c);
          *asyncError = ncclRemoteError;
          break;
        }
      }
    }
  }

  /* if there is linked group job, we should complete it. */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
