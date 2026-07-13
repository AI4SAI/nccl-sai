/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "enqueue.h"
#include "graph.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"
#include "param.h"
#include "sai_profile.h"

#include <algorithm>
#include <limits.h>
#include <stdint.h>
#include <string.h>

NCCL_PARAM(SaiA2aEnable, "SAI_A2A_ENABLE", -1);
NCCL_PARAM(SaiA2aPlannerEnable, "SAI_A2A_PLANNER_ENABLE", -1);
NCCL_PARAM(SaiA2aPlannerRounds, "SAI_A2A_PLANNER_ROUNDS", -1);
NCCL_PARAM(SaiA2aGroupNodes, "SAI_A2A_GROUP_NODES", -1);
NCCL_PARAM(SaiA2aMultigroupEnable, "SAI_A2A_MULTIGROUP_ENABLE", -1);
NCCL_PARAM(SaiA2aMinPeerBytes, "SAI_A2A_MIN_PEER_BYTES", 131072);
NCCL_PARAM(SaiA2aMinRanks, "SAI_A2A_MIN_RANKS", 32);
NCCL_PARAM(SaiA2aIslandEnable, "SAI_A2A_ISLAND_ENABLE", -1);
NCCL_PARAM(SaiA2aIslandSize, "SAI_A2A_ISLAND_SIZE", 4);
NCCL_PARAM(SaiA2aIslandMaxPeerBytes, "SAI_A2A_ISLAND_MAX_PEER_BYTES", 4096);
NCCL_PARAM(SaiA2aIslandMinRanks, "SAI_A2A_ISLAND_MIN_RANKS", NCCL_SAI_A2A_ISLAND_MIN_RANKS_DEFAULT);
NCCL_PARAM(SaiA2aIslandScratchCapBytes, "SAI_A2A_ISLAND_SCRATCH_CAP_BYTES", 64 * 1024 * 1024);

extern int64_t ncclParamSaiP2pFabricGroupSchedule();

// Only controls that can change grouped AlltoAll/P2P scheduling belong in this
// signature. Local-P2P policy has its own communicator-wide signature and must
// not turn an otherwise unrelated AlltoAll configuration into invalid usage.
static const char* const saiA2aExplicitControlNames[] = {
    "NCCL_SAI_DISABLE",
    "NCCL_SAI_FABRIC_PROFILE",
    "NCCL_SAI_A2A_ENABLE",
    "NCCL_SAI_A2A_PLANNER_ENABLE",
    "NCCL_SAI_A2A_PLANNER_ROUNDS",
    "NCCL_SAI_A2A_GROUP_NODES",
    "NCCL_SAI_A2A_MULTIGROUP_ENABLE",
    "NCCL_SAI_A2A_MIN_PEER_BYTES",
    "NCCL_SAI_A2A_MIN_RANKS",
    "NCCL_SAI_A2A_ISLAND_ENABLE",
    "NCCL_SAI_A2A_ISLAND_SIZE",
    "NCCL_SAI_A2A_ISLAND_MAX_PEER_BYTES",
    "NCCL_SAI_A2A_ISLAND_MIN_RANKS",
    "NCCL_SAI_A2A_ISLAND_SCRATCH_CAP_BYTES",
    "NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE",
};

static bool saiA2aExplicitControlPresent() {
  for (size_t i = 0;
       i < sizeof(saiA2aExplicitControlNames) /
           sizeof(saiA2aExplicitControlNames[0]);
       i++) {
    if (ncclGetEnv(saiA2aExplicitControlNames[i]) != nullptr) return true;
  }
  return false;
}

static uint64_t saiA2aExplicitControlSignature() {
  uint64_t hash = UINT64_C(14695981039346656037);
  static const char domain[] = "nccl-sai:explicit-controls:v1";
  hash = ncclSaiHashString(hash, domain);
  if (ncclSaiGloballyDisabled()) {
    return ncclSaiHashString(hash, "canonical-disabled");
  }
  for (size_t i = 0;
       i < sizeof(saiA2aExplicitControlNames) /
           sizeof(saiA2aExplicitControlNames[0]);
       i++) {
    hash = ncclSaiHashString(hash, saiA2aExplicitControlNames[i]);
    hash = ncclSaiHashString(
        hash, ncclGetEnv(saiA2aExplicitControlNames[i]));
  }
  return hash;
}

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

static bool saiA2aFabricEnabled(int64_t enabled, bool fullMeshProfile) {
  if (ncclSaiGloballyDisabled() || !fullMeshProfile) return false;
  if (enabled == 0) return false;
  return true;
}

void ncclSaiA2aGetFabricGroupInfo(struct ncclSaiFabricGroupInfo* info) {
  if (info == nullptr) return;
  info->id = 0;
  info->state = ncclSaiFabricGroupIdAbsent;
  if (ncclSaiGloballyDisabled()) return;
  const char* value = ncclGetEnv("NCCL_SAI_FABRIC_GROUP_ID");
  if (value != nullptr) {
    if (!ncclSaiParseFabricGroupId(value, &info->id)) {
      info->state = ncclSaiFabricGroupIdInvalid;
      return;
    }
    info->state = ncclSaiFabricGroupIdValid;
    return;
  }
}

void ncclSaiA2aGetConfig(struct ncclComm* comm, struct ncclSaiA2aConfig* config) {
  if (config == nullptr) return;
  memset(config, 0, sizeof(*config));
  if (ncclSaiGloballyDisabled()) {
    config->field[ncclSaiA2aConfigVersion] = NCCL_SAI_INIT_SCHEMA_TAG;
    config->field[ncclSaiA2aConfigActivationSource] =
        ncclSaiA2aActivationDisabled;
    config->field[ncclSaiA2aConfigExplicitControlsPresent] =
        saiA2aExplicitControlPresent() ? 1 : 0;
    config->field[ncclSaiA2aConfigExplicitControlsSignature] =
        (int64_t)saiA2aExplicitControlSignature();
    return;
  }
  bool topologyEligible = ncclTopoSaiFullMeshTopologyEligible(comm);
  bool explicitFullMeshProfile =
      ncclSaiFullMeshProfileEnabled() && topologyEligible;
  // NCCL 2.28 has no stable cross-node hardware identity for the physical
  // fabric group. Do not infer it from scheduler metadata, rank order, host
  // names, or HCA strings. Grouped AlltoAll remains an explicit expert path
  // until a hardware-backed provider exists; transparent runtime behavior
  // stays upstream and allocates no SAI scratch.
  bool automaticFullMeshProfile = false;
  bool fullMeshProfile = explicitFullMeshProfile || automaticFullMeshProfile;
  int64_t a2aEnable = ncclParamSaiA2aEnable();
  bool a2aEnabled = saiA2aFabricEnabled(a2aEnable, fullMeshProfile);
  bool fullMeshDefaults = fullMeshProfile && a2aEnabled;
  int64_t automaticPolicyMask = 0;
  config->field[ncclSaiA2aConfigVersion] = NCCL_SAI_INIT_SCHEMA_TAG;
  config->field[ncclSaiA2aConfigEnabled] = a2aEnabled ? 1 : 0;
  int64_t plannerEnable = ncclParamSaiA2aPlannerEnable();
  const int64_t plannerEnableConfigured = plannerEnable;
  int64_t plannerRounds = ncclParamSaiA2aPlannerRounds();
  int64_t groupNodes = ncclParamSaiA2aGroupNodes();
  int64_t multigroupEnable = ncclParamSaiA2aMultigroupEnable();
  int64_t minPeerBytes = ncclParamSaiA2aMinPeerBytes();
  int64_t minRanks = ncclParamSaiA2aMinRanks();
  int64_t islandEnable = ncclParamSaiA2aIslandEnable();
  const int64_t islandEnableConfigured = islandEnable;
  int64_t islandSize = ncclParamSaiA2aIslandSize();
  int64_t islandMaxPeerBytes = ncclParamSaiA2aIslandMaxPeerBytes();
  int64_t islandMinRanks = ncclParamSaiA2aIslandMinRanks();
  int64_t islandScratchCapBytes = ncclParamSaiA2aIslandScratchCapBytes();
  int64_t p2pFabricSchedule = ncclParamSaiP2pFabricGroupSchedule();
  const int64_t p2pFabricScheduleConfigured = p2pFabricSchedule;
  if (plannerEnable < 0) {
    plannerEnable = fullMeshDefaults ? 1 : 0;
    if (ncclSaiA2aDefaultPolicyIsAutomatic(
        plannerEnableConfigured, plannerEnable != 0,
        automaticFullMeshProfile)) {
      automaticPolicyMask |= ncclSaiA2aAutomaticPlanner;
    }
  }
  bool autoPlannerRounds = plannerRounds == -1 && fullMeshDefaults;
  if (!autoPlannerRounds && (plannerRounds < 1 || plannerRounds > INT_MAX)) plannerRounds = 4;
  int64_t defaultGroupNodes = fullMeshProfile ? 16 : 4;
  if (groupNodes < 0) groupNodes = defaultGroupNodes;
  if (groupNodes < 1 || groupNodes > INT_MAX) groupNodes = defaultGroupNodes;
  if (minPeerBytes < 0) minPeerBytes = 0;
  if (minRanks < 0) minRanks = 0;
  if (islandEnable < 0) {
    islandEnable = fullMeshDefaults ? 1 : 0;
    if (ncclSaiA2aDefaultPolicyIsAutomatic(
        islandEnableConfigured, islandEnable != 0,
        automaticFullMeshProfile)) {
      automaticPolicyMask |= ncclSaiA2aAutomaticIsland;
    }
  }
  if (islandSize < 2 || islandSize > INT_MAX) islandSize = 4;
  if (islandMaxPeerBytes < 0) islandMaxPeerBytes = 0;
  if (islandMinRanks < 0) islandMinRanks = 0;
  if (islandScratchCapBytes < 0) islandScratchCapBytes = 0;
  if (p2pFabricSchedule < 0) {
    p2pFabricSchedule = fullMeshDefaults ? 1 : 0;
    if (ncclSaiA2aDefaultPolicyIsAutomatic(
        p2pFabricScheduleConfigured, p2pFabricSchedule != 0,
        automaticFullMeshProfile)) {
      automaticPolicyMask |= ncclSaiA2aAutomaticP2pSchedule;
    }
  }
  config->field[ncclSaiA2aConfigPlannerEnable] = plannerEnable != 0 ? 1 : 0;
  config->field[ncclSaiA2aConfigPlannerRounds] = plannerRounds;
  config->field[ncclSaiA2aConfigGroupNodes] = groupNodes;
  if (multigroupEnable < -1) multigroupEnable = -1;
  config->field[ncclSaiA2aConfigMultigroupEnable] = multigroupEnable > 0 ? 1 : multigroupEnable;
  config->field[ncclSaiA2aConfigMinPeerBytes] = minPeerBytes;
  config->field[ncclSaiA2aConfigMinRanks] = minRanks;
  config->field[ncclSaiA2aConfigIslandEnable] = islandEnable != 0 ? 1 : 0;
  config->field[ncclSaiA2aConfigIslandSize] = islandSize;
  config->field[ncclSaiA2aConfigIslandMaxPeerBytes] = islandMaxPeerBytes;
  config->field[ncclSaiA2aConfigIslandMinRanks] = islandMinRanks;
  config->field[ncclSaiA2aConfigIslandScratchCapBytes] = islandScratchCapBytes;
  config->field[ncclSaiA2aConfigP2pFabricSchedule] =
      ncclSaiP2pFabricScheduleRequested(
          a2aEnabled, p2pFabricSchedule != 0) ? 1 : 0;
  config->field[ncclSaiA2aConfigActivationSource] =
      ncclSaiA2aActivationSourceFor(
          a2aEnabled, a2aEnable, explicitFullMeshProfile);
  if (config->field[ncclSaiA2aConfigActivationSource] ==
      ncclSaiA2aActivationAutomatic) {
    automaticPolicyMask |= ncclSaiA2aAutomaticBase;
  }
  config->field[ncclSaiA2aConfigExplicitControlsPresent] =
      saiA2aExplicitControlPresent() ? 1 : 0;
  config->field[ncclSaiA2aConfigExplicitControlsSignature] =
      (int64_t)saiA2aExplicitControlSignature();
  config->field[ncclSaiA2aConfigAutomaticPolicyMask] = automaticPolicyMask;
}

static void saiA2aSetReason(const char** reasonOut, const char* reason) {
  if (reasonOut != nullptr) *reasonOut = reason;
}

static bool saiA2aMulSize(size_t a, size_t b, size_t* result) {
  if (result == nullptr || (a != 0 && b > SIZE_MAX / a)) return false;
  *result = a * b;
  return true;
}

size_t ncclSaiA2aIslandScratchReserveBytes(
    const struct ncclSaiA2aConfig* config, int nRanks) {
  if (config == nullptr || nRanks <= 0 ||
      config->field[ncclSaiA2aConfigEnabled] == 0 ||
      config->field[ncclSaiA2aConfigIslandEnable] == 0) return 0;

  int64_t minRanksParam = config->field[ncclSaiA2aConfigIslandMinRanks];
  if (minRanksParam > 0 && nRanks < minRanksParam) return 0;

  int64_t islandSizeParam = config->field[ncclSaiA2aConfigIslandSize];
  int64_t scratchCapParam = config->field[ncclSaiA2aConfigIslandScratchCapBytes];
  if (islandSizeParam < 2 || islandSizeParam > INT_MAX || scratchCapParam <= 0) return 0;
  int islandSize = (int)islandSizeParam;
  if (nRanks % islandSize != 0) return 0;

  size_t scratchCap = (uint64_t)scratchCapParam > SIZE_MAX ?
      SIZE_MAX : (size_t)scratchCapParam;
  int64_t maxPeerBytesParam = config->field[ncclSaiA2aConfigIslandMaxPeerBytes];
  if (maxPeerBytesParam <= 0) return scratchCap;

  size_t maxPeerBytes = (uint64_t)maxPeerBytesParam > SIZE_MAX ?
      SIZE_MAX : (size_t)maxPeerBytesParam;
  size_t scratchBytes = 0;
  if (!saiA2aMulSize(maxPeerBytes, (size_t)nRanks, &scratchBytes)) return scratchCap;
  return std::min(scratchBytes, scratchCap);
}

static bool saiA2aUniformLocalRanks(struct ncclComm* comm) {
  if (comm == nullptr || comm->nodeRanks == nullptr || comm->localRanks <= 0 || comm->nNodes <= 0) return false;
  for (int node = 0; node < comm->nNodes; node++) {
    if (comm->nodeRanks[node].localRanks != comm->localRanks) return false;
  }
  return true;
}

static int saiA2aIslandRank(
    struct ncclComm* comm, int island, int islandLocal, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int node = island / islandsPerNode;
  int nodeIsland = island % islandsPerNode;
  return comm->nodeRanks[node].localRankToRank[nodeIsland * islandSize + islandLocal];
}

static bool saiA2aIslandGlobalContiguous(struct ncclComm* comm, int islandSize) {
  int islandsPerNode = comm->localRanks / islandSize;
  int nIslands = comm->nRanks / islandSize;
  for (int island = 0; island < nIslands; island++) {
    for (int islandLocal = 0; islandLocal < islandSize; islandLocal++) {
      if (saiA2aIslandRank(comm, island, islandLocal, islandSize) !=
          island * islandSize + islandLocal) return false;
    }
  }
  return islandsPerNode > 0;
}

bool ncclSaiA2aIslandEligible(struct ncclComm* comm, size_t count, size_t peerBytes,
    struct ncclSaiA2aIslandLayout* layout,
    const char** reasonOut) {
  if (comm == nullptr || layout == nullptr) { saiA2aSetReason(reasonOut, "bad_args"); return false; }
  const struct ncclSaiA2aConfig* config = &comm->saiA2a.config;
  if (!comm->saiA2a.configConsistent) { saiA2aSetReason(reasonOut, "config_mismatch"); return false; }
  if (config->field[ncclSaiA2aConfigEnabled] == 0) { saiA2aSetReason(reasonOut, "profile_disabled"); return false; }
  if (config->field[ncclSaiA2aConfigIslandEnable] == 0) { saiA2aSetReason(reasonOut, "island_disabled"); return false; }
  if (!comm->saiA2a.islandScratchReady) {
    saiA2aSetReason(reasonOut, "island_scratch_unavailable");
    return false;
  }
  if (comm->nRanks < config->field[ncclSaiA2aConfigIslandMinRanks]) {
    saiA2aSetReason(reasonOut, "island_too_few_ranks");
    return false;
  }
  int64_t groupNodesParam = config->field[ncclSaiA2aConfigGroupNodes];
  if (groupNodesParam <= 0 || groupNodesParam > INT_MAX) {
    saiA2aSetReason(reasonOut, "island_invalid_fabric_group");
    return false;
  }
  if (!ncclSaiA2aGroupedLayoutAllowed(comm->nNodes, (int)groupNodesParam,
      config->field[ncclSaiA2aConfigMultigroupEnable],
      comm->saiA2a.fabricGroupsComplete)) {
    bool multigroup = comm->nNodes > groupNodesParam;
    saiA2aSetReason(reasonOut,
      multigroup && config->field[ncclSaiA2aConfigMultigroupEnable] == 0 ?
      "island_multigroup_disabled" : "island_fabric_group_metadata_unavailable");
    return false;
  }
  if (peerBytes == 0) { saiA2aSetReason(reasonOut, "island_zero_peer_bytes"); return false; }
  int64_t maxPeerBytes = config->field[ncclSaiA2aConfigIslandMaxPeerBytes];
  if (maxPeerBytes > 0 && peerBytes > (size_t)maxPeerBytes) {
    saiA2aSetReason(reasonOut, "island_large_peer_bytes");
    return false;
  }
  int64_t islandSizeParam = config->field[ncclSaiA2aConfigIslandSize];
  if (islandSizeParam < 2 || islandSizeParam > INT_MAX) {
    saiA2aSetReason(reasonOut, "island_bad_size");
    return false;
  }
  int islandSize = (int)islandSizeParam;
  if (comm->localRanks < islandSize || comm->nRanks % islandSize != 0 ||
      comm->localRanks % islandSize != 0) {
    saiA2aSetReason(reasonOut, "island_bad_size");
    return false;
  }
  if (!saiA2aUniformLocalRanks(comm)) { saiA2aSetReason(reasonOut, "nonuniform_local_ranks"); return false; }
  if (comm->nNodes > INT_MAX / comm->localRanks || comm->nNodes * comm->localRanks != comm->nRanks) {
    saiA2aSetReason(reasonOut, "nonuniform_local_ranks");
    return false;
  }
  if (comm->node < 0 || comm->node >= comm->nNodes ||
      comm->localRank < 0 || comm->localRank >= comm->localRanks) {
    saiA2aSetReason(reasonOut, "invalid_local_rank");
    return false;
  }
  int islandsPerNode = comm->localRanks / islandSize;
  for (int node = 0; node < comm->nNodes; node++) {
    if (comm->nodeRanks[node].localRankToRank == nullptr) {
      saiA2aSetReason(reasonOut, "missing_local_rank_map");
      return false;
    }
  }
  if (!saiA2aIslandGlobalContiguous(comm, islandSize)) {
    saiA2aSetReason(reasonOut, "island_noncontiguous_global_ranks");
    return false;
  }

  int nIslands = comm->nRanks / islandSize;
  if (!saiA2aMulSize(count, (size_t)islandSize, &layout->blockCount) ||
      !saiA2aMulSize(peerBytes, (size_t)islandSize, &layout->blockBytes) ||
      !saiA2aMulSize(layout->blockBytes, (size_t)nIslands, &layout->stageBytes)) {
    saiA2aSetReason(reasonOut, "island_bad_scratch_size");
    return false;
  }
  int64_t scratchCapBytes = config->field[ncclSaiA2aConfigIslandScratchCapBytes];
  if (scratchCapBytes <= 0 || layout->stageBytes > (size_t)scratchCapBytes ||
      layout->stageBytes > comm->saiA2a.islandScratchBytes || comm->saiA2a.islandScratch == nullptr) {
    saiA2aSetReason(reasonOut, "island_scratch_cap");
    return false;
  }

  layout->islandSize = islandSize;
  layout->nIslands = nIslands;
  layout->myIslandLocal = comm->localRank % islandSize;
  layout->myIsland = comm->node * islandsPerNode + comm->localRank / islandSize;
  saiA2aSetReason(reasonOut, "eligible");
  return true;
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

  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
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
