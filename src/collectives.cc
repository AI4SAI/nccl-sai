/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "group.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"
#include "param.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

NCCL_PARAM(SaiA2aEnable, "SAI_A2A_ENABLE", -1);
NCCL_PARAM(SaiA2aLaneEnable, "SAI_A2A_LANE_ENABLE", 1);
NCCL_PARAM(SaiA2aLanes, "SAI_A2A_LANES", 8);
NCCL_PARAM(SaiA2aGroupNodes, "SAI_A2A_GROUP_NODES", 4);
NCCL_PARAM(SaiA2aMinPeerBytes, "SAI_A2A_MIN_PEER_BYTES", 131072);
NCCL_PARAM(SaiA2aMinRanks, "SAI_A2A_MIN_RANKS", 32);
NCCL_PARAM(SaiA2aLaneTrace, "SAI_A2A_LANE_TRACE", 0);
NCCL_PARAM(SaiA2aIslandEnable, "SAI_A2A_ISLAND_ENABLE", 1);
NCCL_PARAM(SaiA2aIslandSize, "SAI_A2A_ISLAND_SIZE", 4);
NCCL_PARAM(SaiA2aIslandMaxPeerBytes, "SAI_A2A_ISLAND_MAX_PEER_BYTES", 4096);
NCCL_PARAM(SaiA2aIslandMinRanks, "SAI_A2A_ISLAND_MIN_RANKS", 32);
NCCL_PARAM(SaiA2aIslandBulkLocalEnable, "SAI_A2A_ISLAND_BULK_LOCAL_ENABLE", 1);

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

static ncclResult_t ncclNativeAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

static bool saiA2aUniformLocalRanks(struct ncclComm* comm) {
  if (comm == nullptr || comm->nodeRanks == nullptr || comm->localRanks <= 0 || comm->nNodes <= 0) return false;
  for (int n = 0; n < comm->nNodes; n++) {
    if (comm->nodeRanks[n].localRanks != comm->localRanks) return false;
  }
  return true;
}

static void saiA2aSetReason(const char** reasonOut, const char* reason) {
  if (reasonOut != nullptr) *reasonOut = reason;
}

static bool saiA2aMulSize(size_t a, size_t b, size_t* result) {
  if (result == nullptr || (a != 0 && b > SIZE_MAX / a)) return false;
  *result = a * b;
  return true;
}

static bool saiA2aAddSize(size_t a, size_t b, size_t* result) {
  if (result == nullptr || b > SIZE_MAX - a) return false;
  *result = a + b;
  return true;
}

static bool saiA2aProfileDisabledValue(const char* value) {
  if (value == nullptr || value[0] == '\0') return true;
  return strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0 ||
      strcmp(value, "off") == 0 || strcmp(value, "OFF") == 0 ||
      strcmp(value, "none") == 0 || strcmp(value, "NONE") == 0 ||
      strcmp(value, "native") == 0 || strcmp(value, "NATIVE") == 0 ||
      strcmp(value, "upstream") == 0 || strcmp(value, "UPSTREAM") == 0;
}

static bool saiA2aFabricProfileEnabled() {
  return !saiA2aProfileDisabledValue(getenv("NCCL_SAI_FABRIC_PROFILE"));
}

static bool saiA2aFabricEnabled(const char** reasonOut) {
  int64_t enabled = ncclParamSaiA2aEnable();
  if (enabled == 0) { saiA2aSetReason(reasonOut, "sai_a2a_disabled"); return false; }
  if (enabled > 0) { saiA2aSetReason(reasonOut, "sai_a2a_explicit"); return true; }
  if (saiA2aFabricProfileEnabled()) {
    saiA2aSetReason(reasonOut, "sai_fabric_profile");
    return true;
  }
  saiA2aSetReason(reasonOut, "non_sai_fabric_profile");
  return false;
}

static bool saiA2aLaneEligible(struct ncclComm* comm, size_t peerBytes, int* lanesOut, int* groupNodesOut, const char** reasonOut) {
  if (comm == nullptr || lanesOut == nullptr || groupNodesOut == nullptr) { saiA2aSetReason(reasonOut, "bad_args"); return false; }
  if (ncclParamSaiA2aLaneEnable() == 0) { saiA2aSetReason(reasonOut, "disabled"); return false; }
  if (ncclGroupDepth != 0) { saiA2aSetReason(reasonOut, "inside_group"); return false; }
  if (comm->config.blocking == 0) { saiA2aSetReason(reasonOut, "nonblocking_comm"); return false; }
  if (comm->nRanks < ncclParamSaiA2aMinRanks()) { saiA2aSetReason(reasonOut, "too_few_ranks"); return false; }

  int64_t minPeerBytes = ncclParamSaiA2aMinPeerBytes();
  if (minPeerBytes < 0) minPeerBytes = 0;
  if (peerBytes < (size_t)minPeerBytes) { saiA2aSetReason(reasonOut, "small_peer_bytes"); return false; }

  int lanes = (int)ncclParamSaiA2aLanes();
  int groupNodes = (int)ncclParamSaiA2aGroupNodes();
  if (lanes < 2 || groupNodes <= 0) { saiA2aSetReason(reasonOut, "bad_lane_config"); return false; }
  if (comm->nNodes % groupNodes != 0) { saiA2aSetReason(reasonOut, "nonmultiple_group_nodes"); return false; }
  if (!saiA2aUniformLocalRanks(comm)) { saiA2aSetReason(reasonOut, "nonuniform_local_ranks"); return false; }
  if (comm->rankToNode == nullptr || comm->rankToLocalRank == nullptr) { saiA2aSetReason(reasonOut, "missing_rank_maps"); return false; }

  int groupRanks = groupNodes * comm->localRanks;
  if (groupRanks <= 0) { saiA2aSetReason(reasonOut, "bad_group_ranks"); return false; }
  if (lanes > groupRanks) lanes = groupRanks;
  *lanesOut = lanes;
  *groupNodesOut = groupNodes;
  saiA2aSetReason(reasonOut, "eligible");
  return true;
}

static void saiA2aTrace(struct ncclComm* comm, const char* path, const char* reason, size_t peerBytes, int lanes, int groupNodes) {
  if (ncclParamSaiA2aLaneTrace() == 0) return;
  int rank = comm == nullptr ? -1 : comm->rank;
  if (rank != 0) return;
  fprintf(stderr,
      "SAI_A2A_LANE_TRACE path=%s reason=%s rank=%d nranks=%d nnodes=%d localRanks=%d peerBytes=%zu lanes=%d groupNodes=%d groupDepth=%d blocking=%d\n",
      path, reason == nullptr ? "unknown" : reason, rank, comm == nullptr ? -1 : comm->nRanks,
      comm == nullptr ? -1 : comm->nNodes, comm == nullptr ? -1 : comm->localRanks, peerBytes, lanes, groupNodes,
      ncclGroupDepth, comm == nullptr ? -1 : comm->config.blocking);
}

static int saiA2aGroupLocalRank(struct ncclComm* comm, int rank, int groupNodes) {
  int node = comm->rankToNode[rank];
  int localRank = comm->rankToLocalRank[rank];
  return (node % groupNodes) * comm->localRanks + localRank;
}

static int saiA2aIslandRank(struct ncclComm* comm, int island, int islandLocal, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int node = island / islandsPerNode;
  int nodeIsland = island % islandsPerNode;
  return comm->nodeRanks[node].localRankToRank[nodeIsland * islandSize + islandLocal];
}

static bool saiA2aIslandGlobalContiguous(struct ncclComm* comm, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int nIslands = comm->nNodes * islandsPerNode;
  for (int island = 0; island < nIslands; island++) {
    for (int islandLocal = 0; islandLocal < islandSize; islandLocal++) {
      if (saiA2aIslandRank(comm, island, islandLocal, islandSize) != island * islandSize + islandLocal) return false;
    }
  }
  return true;
}

static bool saiA2aIslandEligible(struct ncclComm* comm, const void* sendbuff, const void* recvbuff,
    size_t peerBytes, int* islandSizeOut, const char** reasonOut) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || islandSizeOut == nullptr) { saiA2aSetReason(reasonOut, "bad_args"); return false; }
  if (ncclParamSaiA2aIslandEnable() == 0) { saiA2aSetReason(reasonOut, "island_disabled"); return false; }
  if (ncclGroupDepth != 0) { saiA2aSetReason(reasonOut, "inside_group"); return false; }
  if (comm->config.blocking == 0) { saiA2aSetReason(reasonOut, "nonblocking_comm"); return false; }
  if (comm->nRanks < ncclParamSaiA2aIslandMinRanks()) { saiA2aSetReason(reasonOut, "island_too_few_ranks"); return false; }
  if (peerBytes == 0) { saiA2aSetReason(reasonOut, "island_zero_peer_bytes"); return false; }
  int64_t maxPeerBytes = ncclParamSaiA2aIslandMaxPeerBytes();
  if (maxPeerBytes < 0) maxPeerBytes = 0;
  if (maxPeerBytes > 0 && peerBytes > (size_t)maxPeerBytes) { saiA2aSetReason(reasonOut, "island_large_peer_bytes"); return false; }
  int islandSize = (int)ncclParamSaiA2aIslandSize();
  if (islandSize < 2 || comm->localRanks < islandSize || (comm->localRanks % islandSize) != 0) {
    saiA2aSetReason(reasonOut, "island_bad_size");
    return false;
  }
  if (!saiA2aUniformLocalRanks(comm)) { saiA2aSetReason(reasonOut, "nonuniform_local_ranks"); return false; }
  if (comm->rankToNode == nullptr || comm->rankToLocalRank == nullptr) { saiA2aSetReason(reasonOut, "missing_rank_maps"); return false; }
  int islandsPerNode = comm->localRanks / islandSize;
  for (int n = 0; n < comm->nNodes; n++) {
    if (comm->nodeRanks[n].localRankToRank == nullptr) { saiA2aSetReason(reasonOut, "missing_local_rank_map"); return false; }
    for (int ni = 0; ni < islandsPerNode; ni++) {
      int firstRank = comm->nodeRanks[n].localRankToRank[ni * islandSize];
      for (int l = 1; l < islandSize; l++) {
        if (comm->nodeRanks[n].localRankToRank[ni * islandSize + l] != firstRank + l) {
          saiA2aSetReason(reasonOut, "island_noncontiguous_ranks");
          return false;
        }
      }
    }
  }
  *islandSizeOut = islandSize;
  saiA2aSetReason(reasonOut, "eligible");
  return true;
}

static bool saiA2aIslandBulkEligible(struct ncclComm* comm, int islandSize, const char** reasonOut) {
  if (ncclParamSaiA2aIslandBulkLocalEnable() == 0) { saiA2aSetReason(reasonOut, "island_bulk_disabled"); return false; }
  if (!saiA2aIslandGlobalContiguous(comm, islandSize)) { saiA2aSetReason(reasonOut, "island_noncontiguous_global_ranks"); return false; }
  saiA2aSetReason(reasonOut, "eligible");
  return true;
}

static ncclResult_t saiA2aCopySelf(const void* sendbuff, void* recvbuff, int rank, size_t peerBytes,
    cudaStream_t stream) {
  if (sendbuff == recvbuff || peerBytes == 0) return ncclSuccess;
  const char* src = (const char*)sendbuff + (size_t)rank * peerBytes;
  char* dst = (char*)recvbuff + (size_t)rank * peerBytes;
  cudaError_t err = cudaMemcpyAsync(dst, src, peerBytes, cudaMemcpyDeviceToDevice, stream);
  return err == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

static ncclResult_t saiA2aLaneAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream, size_t peerBytes, int lanes, int groupNodes) {
  ncclResult_t ret = saiA2aCopySelf(sendbuff, recvbuff, comm->rank, peerBytes, stream);
  if (ret != ncclSuccess) return ret;

  int myGroupLocal = saiA2aGroupLocalRank(comm, comm->rank, groupNodes);
  for (int phase = 0; phase < lanes; phase++) {
    bool hasPeer = false;
    for (int peer = 0; peer < comm->nRanks; peer++) {
      if (peer == comm->rank) continue;
      int peerGroupLocal = saiA2aGroupLocalRank(comm, peer, groupNodes);
      if (((myGroupLocal + peerGroupLocal) % lanes) == phase) {
        hasPeer = true;
        break;
      }
    }
    if (!hasPeer) continue;

    ncclResult_t phaseRet = ncclGroupStartInternal();
    if (phaseRet != ncclSuccess) return phaseRet;
    for (int peer = 0; peer < comm->nRanks; peer++) {
      if (peer == comm->rank) continue;
      int peerGroupLocal = saiA2aGroupLocalRank(comm, peer, groupNodes);
      if (((myGroupLocal + peerGroupLocal) % lanes) != phase) continue;
      const char* sendPtr = (const char*)sendbuff + (size_t)peer * peerBytes;
      char* recvPtr = (char*)recvbuff + (size_t)peer * peerBytes;
      ret = ncclSend(sendPtr, count, datatype, peer, comm, stream);
      if (ret != ncclSuccess && ret != ncclInProgress) {
        phaseRet = ret;
        break;
      }
      ret = ncclRecv(recvPtr, count, datatype, peer, comm, stream);
      if (ret != ncclSuccess && ret != ncclInProgress) {
        phaseRet = ret;
        break;
      }
    }
    ret = ncclGroupEndInternal();
    if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) return phaseRet;
    if (ret != ncclSuccess && ret != ncclInProgress) return ret;
  }

  return ncclSuccess;
}

static ncclResult_t saiA2aIslandAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream, size_t peerBytes, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int nIslands = comm->nNodes * islandsPerNode;
  int myNodeIsland = comm->localRank / islandSize;
  int myIslandLocal = comm->localRank % islandSize;
  int myIsland = comm->node * islandsPerNode + myNodeIsland;
  size_t blockCount, blockBytes, stageBytes;
  if (!saiA2aMulSize(count, (size_t)islandSize, &blockCount) ||
      !saiA2aMulSize(peerBytes, (size_t)islandSize, &blockBytes) ||
      !saiA2aMulSize(blockBytes, (size_t)nIslands, &stageBytes)) return ncclInvalidArgument;
  char* stage = nullptr;
  cudaError_t err = cudaMallocAsync((void**)&stage, stageBytes, stream);
  if (err != cudaSuccess) return ncclUnhandledCudaError;

  ncclResult_t ret = ncclSuccess;
  ncclResult_t phaseRet = ncclGroupStartInternal();
  if (phaseRet != ncclSuccess) { ret = phaseRet; goto fail; }
  for (int island = 0; island < nIslands; island++) {
    if (island == myIsland) continue;
    int peerRank = saiA2aIslandRank(comm, island, myIslandLocal, islandSize);
    int firstRank = saiA2aIslandRank(comm, island, 0, islandSize);
    const char* sendPtr = (const char*)sendbuff + (size_t)firstRank * peerBytes;
    char* recvPtr = stage + (size_t)island * blockBytes;
    ret = ncclSend(sendPtr, blockCount, datatype, peerRank, comm, stream);
    if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
    ret = ncclRecv(recvPtr, blockCount, datatype, peerRank, comm, stream);
    if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
  }
  ret = ncclGroupEndInternal();
  if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) { ret = phaseRet; goto fail; }
  if (ret != ncclSuccess && ret != ncclInProgress) goto fail;

  for (int island = 0; island < nIslands; island++) {
    int srcRank = saiA2aIslandRank(comm, island, myIslandLocal, islandSize);
    const char* src = nullptr;
    if (island == myIsland) src = (const char*)sendbuff + (size_t)comm->rank * peerBytes;
    else src = stage + (size_t)island * blockBytes + (size_t)myIslandLocal * peerBytes;
    char* dst = (char*)recvbuff + (size_t)srcRank * peerBytes;
    err = cudaMemcpyAsync(dst, src, peerBytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) { ret = ncclUnhandledCudaError; goto fail; }
  }

  phaseRet = ncclGroupStartInternal();
  if (phaseRet != ncclSuccess) { ret = phaseRet; goto fail; }
  for (int dstLocal = 0; dstLocal < islandSize; dstLocal++) {
    if (dstLocal == myIslandLocal) continue;
    int dstRank = saiA2aIslandRank(comm, myIsland, dstLocal, islandSize);
    for (int srcIsland = 0; srcIsland < nIslands; srcIsland++) {
      const char* sendPtr = nullptr;
      if (srcIsland == myIsland) sendPtr = (const char*)sendbuff + (size_t)dstRank * peerBytes;
      else sendPtr = stage + (size_t)srcIsland * blockBytes + (size_t)dstLocal * peerBytes;
      ret = ncclSend(sendPtr, count, datatype, dstRank, comm, stream);
      if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
    }
    if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) break;
  }
  if (phaseRet == ncclSuccess || phaseRet == ncclInProgress) {
    for (int srcLocal = 0; srcLocal < islandSize; srcLocal++) {
      if (srcLocal == myIslandLocal) continue;
      int srcPeer = saiA2aIslandRank(comm, myIsland, srcLocal, islandSize);
      for (int srcIsland = 0; srcIsland < nIslands; srcIsland++) {
        int srcRank = saiA2aIslandRank(comm, srcIsland, srcLocal, islandSize);
        char* recvPtr = (char*)recvbuff + (size_t)srcRank * peerBytes;
        ret = ncclRecv(recvPtr, count, datatype, srcPeer, comm, stream);
        if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
      }
      if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) break;
    }
  }
  ret = ncclGroupEndInternal();
  if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) { ret = phaseRet; goto fail; }
  if (ret != ncclSuccess && ret != ncclInProgress) goto fail;
  if (ret == ncclInProgress) ret = ncclSuccess;

fail:
  err = cudaFreeAsync(stage, stream);
  if (ret == ncclSuccess && err != cudaSuccess) ret = ncclUnhandledCudaError;
  return ret;
}

static int saiA2aIslandPeerSlot(int local, int myLocal) {
  return local < myLocal ? local : local - 1;
}

static ncclResult_t saiA2aIslandBulkLocalAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream, size_t peerBytes, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int nIslands = comm->nNodes * islandsPerNode;
  int myNodeIsland = comm->localRank / islandSize;
  int myIslandLocal = comm->localRank % islandSize;
  int myIsland = comm->node * islandsPerNode + myNodeIsland;
  size_t blockCount, blockBytes, stageBytes, localBulkBytes, localBufBytes, scratchBytes;
  if (!saiA2aMulSize(count, (size_t)islandSize, &blockCount) ||
      !saiA2aMulSize(peerBytes, (size_t)islandSize, &blockBytes) ||
      !saiA2aMulSize(blockBytes, (size_t)nIslands, &stageBytes) ||
      !saiA2aMulSize(peerBytes, (size_t)nIslands, &localBulkBytes) ||
      !saiA2aMulSize(localBulkBytes, (size_t)(islandSize - 1), &localBufBytes) ||
      !saiA2aAddSize(stageBytes, localBufBytes, &scratchBytes) ||
      !saiA2aAddSize(scratchBytes, localBufBytes, &scratchBytes)) return ncclInvalidArgument;
  char* scratch = nullptr;
  cudaError_t err = comm->memPool != nullptr ?
    cudaMallocFromPoolAsync((void**)&scratch, scratchBytes, comm->memPool, stream) :
    cudaMallocAsync((void**)&scratch, scratchBytes, stream);
  if (err != cudaSuccess) return ncclUnhandledCudaError;
  char* stage = scratch;
  char* localSend = stage + stageBytes;
  char* localRecv = localSend + localBufBytes;

  ncclResult_t ret = ncclSuccess;
  ncclResult_t phaseRet = ncclSuccess;
  int myIslandFirstRank = saiA2aIslandRank(comm, myIsland, 0, islandSize);
  err = cudaMemcpyAsync(stage + (size_t)myIsland * blockBytes,
      (const char*)sendbuff + (size_t)myIslandFirstRank * peerBytes,
      blockBytes, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) { ret = ncclUnhandledCudaError; goto fail; }

  phaseRet = ncclGroupStartInternal();
  if (phaseRet != ncclSuccess) { ret = phaseRet; goto fail; }
  for (int island = 0; island < nIslands; island++) {
    if (island == myIsland) continue;
    int peerRank = saiA2aIslandRank(comm, island, myIslandLocal, islandSize);
    int firstRank = saiA2aIslandRank(comm, island, 0, islandSize);
    const char* sendPtr = (const char*)sendbuff + (size_t)firstRank * peerBytes;
    char* recvPtr = stage + (size_t)island * blockBytes;
    ret = ncclSend(sendPtr, blockCount, datatype, peerRank, comm, stream);
    if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
    ret = ncclRecv(recvPtr, blockCount, datatype, peerRank, comm, stream);
    if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
  }
  ret = ncclGroupEndInternal();
  if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) { ret = phaseRet; goto fail; }
  if (ret != ncclSuccess && ret != ncclInProgress) goto fail;

  err = cudaMemcpy2DAsync((char*)recvbuff + (size_t)myIslandLocal * peerBytes, blockBytes,
      stage + (size_t)myIslandLocal * peerBytes, blockBytes,
      peerBytes, (size_t)nIslands, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) { ret = ncclUnhandledCudaError; goto fail; }

  for (int dstLocal = 0; dstLocal < islandSize; dstLocal++) {
    if (dstLocal == myIslandLocal) continue;
    int slot = saiA2aIslandPeerSlot(dstLocal, myIslandLocal);
    err = cudaMemcpy2DAsync(localSend + (size_t)slot * localBulkBytes, peerBytes,
        stage + (size_t)dstLocal * peerBytes, blockBytes,
        peerBytes, (size_t)nIslands, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) { ret = ncclUnhandledCudaError; goto fail; }
  }

  phaseRet = ncclGroupStartInternal();
  if (phaseRet != ncclSuccess) { ret = phaseRet; goto fail; }
  for (int dstLocal = 0; dstLocal < islandSize; dstLocal++) {
    if (dstLocal == myIslandLocal) continue;
    int slot = saiA2aIslandPeerSlot(dstLocal, myIslandLocal);
    int dstRank = saiA2aIslandRank(comm, myIsland, dstLocal, islandSize);
    ret = ncclSend(localSend + (size_t)slot * localBulkBytes, localBulkBytes, ncclInt8, dstRank, comm, stream);
    if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
  }
  if (phaseRet == ncclSuccess || phaseRet == ncclInProgress) {
    for (int srcLocal = 0; srcLocal < islandSize; srcLocal++) {
      if (srcLocal == myIslandLocal) continue;
      int slot = saiA2aIslandPeerSlot(srcLocal, myIslandLocal);
      int srcPeer = saiA2aIslandRank(comm, myIsland, srcLocal, islandSize);
      ret = ncclRecv(localRecv + (size_t)slot * localBulkBytes, localBulkBytes, ncclInt8, srcPeer, comm, stream);
      if (ret != ncclSuccess && ret != ncclInProgress) { phaseRet = ret; break; }
    }
  }
  ret = ncclGroupEndInternal();
  if (phaseRet != ncclSuccess && phaseRet != ncclInProgress) { ret = phaseRet; goto fail; }
  if (ret != ncclSuccess && ret != ncclInProgress) goto fail;

  for (int srcLocal = 0; srcLocal < islandSize; srcLocal++) {
    if (srcLocal == myIslandLocal) continue;
    int slot = saiA2aIslandPeerSlot(srcLocal, myIslandLocal);
    err = cudaMemcpy2DAsync((char*)recvbuff + (size_t)srcLocal * peerBytes, blockBytes,
        localRecv + (size_t)slot * localBulkBytes, peerBytes,
        peerBytes, (size_t)nIslands, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) { ret = ncclUnhandledCudaError; goto fail; }
  }
  if (ret == ncclInProgress) ret = ncclSuccess;

fail:
  err = cudaFreeAsync(scratch, stream);
  if (ret == ncclSuccess && err != cudaSuccess) ret = ncclUnhandledCudaError;
  return ret;
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));

  ncclResult_t ret = CommCheck(comm, "AlltoAll", "comm");
  if (ret != ncclSuccess) {
    saiA2aTrace(comm, "native", "comm_check_failed", 0, 0, 0);
    return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }
  if (comm->revokedFlag) {
    saiA2aTrace(comm, "native", "revoked", 0, 0, 0);
    return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }

  size_t typeBytes = ncclTypeSize(datatype);
  size_t peerBytes = 0;
  size_t totalBytes = 0;
  int lanes = 0;
  int groupNodes = 0;
  const char* reason = nullptr;
  bool inPlace = sendbuff == recvbuff;
  if (sendbuff == nullptr || recvbuff == nullptr || typeBytes == 0 ||
      !saiA2aMulSize(count, typeBytes, &peerBytes) ||
      !saiA2aMulSize(peerBytes, (size_t)comm->nRanks, &totalBytes)) {
    reason = "bad_buffer_type_or_size";
    saiA2aTrace(comm, "native", reason, peerBytes, lanes, groupNodes);
    return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }
  (void)totalBytes;
  if (!saiA2aFabricEnabled(&reason)) {
    saiA2aTrace(comm, "native", reason, peerBytes, lanes, groupNodes);
    return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
  }

  if (saiA2aLaneEligible(comm, peerBytes, &lanes, &groupNodes, &reason)) {
    saiA2aTrace(comm, "lane", reason, peerBytes, lanes, groupNodes);
    return saiA2aLaneAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream, peerBytes, lanes, groupNodes);
  }

  int islandSize = 0;
  const char* islandReason = nullptr;
  if (saiA2aIslandEligible(comm, sendbuff, recvbuff, peerBytes, &islandSize, &islandReason)) {
    const char* bulkReason = nullptr;
    if (saiA2aIslandBulkEligible(comm, islandSize, &bulkReason)) {
      saiA2aTrace(comm, "island_bulk", inPlace ? "eligible_in_place" : bulkReason, peerBytes, 0, islandSize);
      return saiA2aIslandBulkLocalAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream, peerBytes, islandSize);
    }
    if (inPlace) {
      saiA2aTrace(comm, "native", "island_in_place_no_bulk", peerBytes, 0, islandSize);
      return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
    }
    saiA2aTrace(comm, "island", islandReason, peerBytes, 0, islandSize);
    return saiA2aIslandAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream, peerBytes, islandSize);
  }

  saiA2aTrace(comm, "native", reason == nullptr ? islandReason : reason, peerBytes, lanes, groupNodes);
  return ncclNativeAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}
