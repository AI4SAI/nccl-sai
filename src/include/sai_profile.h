/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SAI_PROFILE_H_
#define NCCL_SAI_PROFILE_H_

#include "param.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

// Keep one schema tag for both the peer-version handshake and the extended
// initialization all-gather. The low 20 bits remain the upstream NCCL version.
#define NCCL_SAI_INIT_SCHEMA_TAG 0x5bcu
#define NCCL_SAI_PEER_NCCL_VERSION_MASK 0x000fffffu
#define NCCL_SAI_A2A_ISLAND_MIN_RANKS_DEFAULT 576

enum ncclSaiA2aActivationSource {
  ncclSaiA2aActivationDisabled = 0,
  ncclSaiA2aActivationAutomatic = 1,
  ncclSaiA2aActivationExplicit = 2,
};
enum ncclSaiA2aAutomaticPolicy {
  ncclSaiA2aAutomaticBase = 1 << 0,
  ncclSaiA2aAutomaticPlanner = 1 << 1,
  ncclSaiA2aAutomaticIsland = 1 << 2,
  ncclSaiA2aAutomaticP2pSchedule = 1 << 3,
};

static inline int ncclSaiA2aActivationSourceFor(
    bool enabled, int64_t configuredEnable, bool explicitFullMeshProfile) {
  if (!enabled) return ncclSaiA2aActivationDisabled;
  if (configuredEnable > 0 || explicitFullMeshProfile) {
    return ncclSaiA2aActivationExplicit;
  }
  return ncclSaiA2aActivationAutomatic;
}

static inline bool ncclSaiA2aDefaultPolicyIsAutomatic(
    int64_t configuredValue, bool resolvedEnabled,
    bool automaticFullMeshProfile) {
  return configuredValue < 0 && resolvedEnabled && automaticFullMeshProfile;
}

static inline int64_t ncclSaiA2aPolicyValueAfterAutomaticFallback(
    int64_t value, int64_t automaticPolicyMask, int64_t policy) {
  return (automaticPolicyMask & policy) != 0 ? 0 : value;
}

static inline int ncclSaiA2aSourceAfterAutomaticFallback(
    int source, int64_t automaticPolicyMask) {
  return (automaticPolicyMask & ncclSaiA2aAutomaticBase) != 0 ?
      ncclSaiA2aActivationDisabled : source;
}

static inline uint32_t ncclSaiPeerVersion(uint32_t ncclVersion) {
  return (NCCL_SAI_INIT_SCHEMA_TAG << 20) | (ncclVersion & NCCL_SAI_PEER_NCCL_VERSION_MASK);
}

static inline int ncclSaiA2aPlannerRoundWindow(int64_t configuredRounds,
    int topologyNodes, int groupNodes) {
  if (configuredRounds > 0 && configuredRounds <= INT_MAX) return (int)configuredRounds;
  if (topologyNodes <= 0 || groupNodes <= 0 || topologyNodes % groupNodes != 0) return 4;
  int fabricGroups = topologyNodes / groupNodes;
  int completeCycles = (4 + fabricGroups - 1) / fabricGroups;
  return fabricGroups * completeCycles;
}

static inline int ncclSaiA2aPlannerRoundLimit(
    int roundBegin, int nRounds, int roundWindow) {
  if (roundBegin < 0 || roundBegin > nRounds) return roundBegin;
  int remaining = nRounds - roundBegin;
  if (roundWindow <= 0 || roundWindow >= remaining) return nRounds;
  return roundBegin + roundWindow;
}

static inline bool ncclSaiA2aPlannerGroupsComplete(int topologyNodes, int groupNodes) {
  return topologyNodes > 0 && groupNodes > 0 && topologyNodes >= groupNodes &&
      topologyNodes % groupNodes == 0;
}

static inline bool ncclSaiA2aMultigroupAllowed(int64_t mode, bool fabricGroupsComplete) {
  if (mode == 0) return false;
  // A positive value may request the separately validated ragged path. It
  // does not make incomplete metadata valid for a grouped layout.
  if (mode > 0) return true;
  return fabricGroupsComplete;
}

static inline bool ncclSaiA2aGroupedLayoutAllowed(int topologyNodes, int groupNodes,
    int64_t multigroupMode, bool fabricGroupsComplete) {
  if (topologyNodes <= 0 || groupNodes <= 0 || topologyNodes < groupNodes) return false;
  if (!fabricGroupsComplete) return false;
  return topologyNodes == groupNodes || multigroupMode != 0;
}

static inline bool ncclSaiA2aRaggedLayoutAllowed(int topologyNodes, int groupNodes,
    int64_t multigroupMode, bool fabricMetadataValid, bool fabricGroupsComplete,
    int fabricGroups, int maxFabricGroupNodes) {
  return topologyNodes >= groupNodes && groupNodes > 0 && multigroupMode > 0 &&
      fabricMetadataValid && !fabricGroupsComplete && fabricGroups > 1 &&
      maxFabricGroupNodes > 0 && maxFabricGroupNodes <= groupNodes;
}

static inline bool ncclSaiP2pFabricScheduleAllowed(int topologyNodes, int groupNodes,
    int64_t multigroupMode, bool fabricGroupsComplete) {
  return fabricGroupsComplete && topologyNodes > groupNodes && groupNodes > 0 &&
      topologyNodes % groupNodes == 0 &&
      ncclSaiA2aMultigroupAllowed(multigroupMode, fabricGroupsComplete);
}

static inline bool ncclSaiP2pFabricScheduleRequested(
    bool baseA2aEnabled, bool scheduleRequested) {
  return baseA2aEnabled && scheduleRequested;
}

static inline bool ncclSaiParseFabricGroupId(const char* value, uint64_t* id) {
  if (value == nullptr || id == nullptr || value[0] == '\0' || value[0] == '-' || value[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') return false;
  *id = (uint64_t)parsed;
  return true;
}

static inline bool ncclSaiProfileFamily(const char* value, const char* family) {
  if (value == nullptr || family == nullptr || value[0] == '\0' || family[0] == '\0') return false;
  size_t familyLen = strlen(family);
  if (strncasecmp(value, family, familyLen) != 0) return false;
  if (value[familyLen] == '\0') return true;
  if (value[familyLen] != '-') return false;
  const char* variant = value + familyLen + 1;
  if (*variant == '\0' || !(('a' <= *variant && *variant <= 'z') ||
      ('A' <= *variant && *variant <= 'Z') || ('0' <= *variant && *variant <= '9'))) return false;
  for (const char* p = variant + 1; *p != '\0'; p++) {
    if (!(('a' <= *p && *p <= 'z') || ('A' <= *p && *p <= 'Z') ||
          ('0' <= *p && *p <= '9') || *p == '-' || *p == '_' || *p == '.')) return false;
  }
  return true;
}

static inline bool ncclSaiGloballyDisabled() {
  const char* value = ncclGetEnv("NCCL_SAI_DISABLE");
  return value != nullptr && strcmp(value, "1") == 0;
}

static inline bool ncclSaiLocalP2pAutomaticLevelAllowed(
    int resolvedUserLevel, int unsetLevel, int nvlinkLevel) {
  return resolvedUserLevel == unsetLevel || resolvedUserLevel == nvlinkLevel;
}

static inline int ncclSaiRankPairIndex(int nRanks, int rank1, int rank2) {
  if (nRanks < 2 || nRanks > 11 || rank1 < 0 || rank1 >= nRanks ||
      rank2 < 0 || rank2 >= nRanks || rank1 == rank2) return -1;
  if (rank1 > rank2) {
    int swap = rank1;
    rank1 = rank2;
    rank2 = swap;
  }
  int index = 0;
  for (int first = 0; first < rank1; first++) {
    index += nRanks - first - 1;
  }
  index += rank2 - rank1 - 1;
  return index < 64 ? index : -1;
}

static inline bool ncclSaiRankPairSelected(
    uint64_t rankPairs, int nRanks, int rank1, int rank2) {
  int index = ncclSaiRankPairIndex(nRanks, rank1, rank2);
  return index >= 0 && (rankPairs & (UINT64_C(1) << index)) != 0;
}

static inline bool ncclSaiA2aPlannerCanRun(
    bool configConsistent, bool enabled, bool plannerEnabled,
    bool islandEnabled) {
  return configConsistent && enabled && (plannerEnabled || islandEnabled);
}

// Profiles are expert-only labels for grouped AlltoAll/P2P experiments. They
// are never consulted by automatic rail or local-P2P capability detection.
// Keep the one implemented expert layout fail-closed.
static inline bool ncclSaiFullMeshProfileEnabled() {
  if (ncclSaiGloballyDisabled()) return false;
  const char* value = ncclGetEnv("NCCL_SAI_FABRIC_PROFILE");
  return ncclSaiProfileFamily(value, "ultrapod-fullmesh");
}

static inline bool ncclSaiAutomaticRailCapabilityEnabled(
    bool topologyEligible) {
  return topologyEligible && !ncclSaiGloballyDisabled();
}

static inline bool ncclSaiRailByChannelRequested(
    bool topologyEligible, int crossNic) {
  return crossNic == 0 &&
      ncclSaiAutomaticRailCapabilityEnabled(topologyEligible);
}

static inline uint64_t ncclSaiHashString(uint64_t hash, const char* value) {
  if (value == nullptr) {
    hash ^= UINT64_C(0xff);
    return hash * UINT64_C(1099511628211);
  }
  for (const unsigned char* p = (const unsigned char*)value; *p != '\0'; p++) {
    hash ^= *p;
    hash *= UINT64_C(1099511628211);
  }
  hash ^= 0;
  return hash * UINT64_C(1099511628211);
}

static inline uint64_t ncclSaiRailPolicySignature() {
  uint64_t hash = UINT64_C(14695981039346656037);
  static const char domain[] = "nccl-sai:rail-policy:v1";
  hash = ncclSaiHashString(hash, domain);
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_SAI_DISABLE"));
  return hash;
}

struct ncclSaiRailInfo {
  uint64_t policySignature;
  uint64_t topologyClass;
  uint64_t railSubnet[2];
  int eligible;
};

static inline bool ncclSaiRailInfoComplete(
    const struct ncclSaiRailInfo* info) {
  return info != nullptr && info->eligible == 1 &&
      info->policySignature != 0 && info->topologyClass != 0 &&
      info->railSubnet[0] != 0 && info->railSubnet[1] != 0 &&
      info->railSubnet[0] != info->railSubnet[1];
}

static inline bool ncclSaiRailInfoConsensus(
    const struct ncclSaiRailInfo* info, int ranks, int* eligibleRanks) {
  if (eligibleRanks != nullptr) *eligibleRanks = 0;
  if (info == nullptr || ranks <= 0) return false;
  bool enabled = true;
  for (int rank = 0; rank < ranks; rank++) {
    if (info[rank].eligible == 1) {
      if (eligibleRanks != nullptr) (*eligibleRanks)++;
    } else {
      enabled = false;
    }
    if (!ncclSaiRailInfoComplete(info+rank) ||
        info[rank].policySignature != info[0].policySignature ||
        info[rank].topologyClass != info[0].topologyClass ||
        info[rank].railSubnet[0] != info[0].railSubnet[0] ||
        info[rank].railSubnet[1] != info[0].railSubnet[1]) {
      enabled = false;
    }
  }
  return enabled;
}

static inline bool ncclSaiAccumulateRailSubnet(
    int port, uint64_t subnetPrefix, uint64_t railSubnet[2]) {
  if (railSubnet == nullptr || subnetPrefix == 0 ||
      (port != 1 && port != 2)) return false;
  uint64_t* slot = railSubnet + (port - 1);
  if (*slot == 0) *slot = subnetPrefix;
  return *slot == subnetPrefix;
}

static inline bool ncclSaiRailSubnetsComplete(
    const uint64_t railSubnet[2]) {
  return railSubnet != nullptr && railSubnet[0] != 0 &&
      railSubnet[1] != 0 && railSubnet[0] != railSubnet[1];
}

static inline int ncclSaiRailPathCapabilityClass(
    int pathType, int minimumType, int maximumType) {
  return pathType >= minimumType && pathType <= maximumType ? maximumType : -1;
}

static inline bool ncclSaiRailPathPairEligible(
    int firstType, float firstBw, int secondType, float secondBw,
    int minimumType, int maximumType) {
  return ncclSaiRailPathCapabilityClass(
      firstType, minimumType, maximumType) >= 0 &&
      ncclSaiRailPathCapabilityClass(
          secondType, minimumType, maximumType) >= 0 &&
      firstType == secondType && firstBw > 0 && firstBw == secondBw;
}

static inline bool ncclSaiRailPathClassesCompatible(
    int referenceType, float referenceBw, int candidateType,
    float candidateBw, int minimumType, int maximumType) {
  int referenceClass = ncclSaiRailPathCapabilityClass(
      referenceType, minimumType, maximumType);
  int candidateClass = ncclSaiRailPathCapabilityClass(
      candidateType, minimumType, maximumType);
  return referenceClass >= 0 && referenceClass == candidateClass &&
      referenceBw > 0 && referenceBw == candidateBw;
}

// The internal IB plugin has three endpoint-fusion states. Explicit
// NCCL_IB_MERGE_NICS=0 keeps the upstream disable behavior, any explicit
// upstream fusion control keeps upstream behavior, and only a completely
// unset policy may inspect the hardware for automatic endpoint preservation.
enum ncclSaiIbEndpointMergeMode {
  ncclSaiIbEndpointMergeExplicitDisabled = 0,
  ncclSaiIbEndpointMergeUpstream = 1,
  ncclSaiIbEndpointMergeAutomatic = 2,
};

struct ncclSaiIbEndpointPolicyInfo {
  uint64_t policySignature;
  int internalIbPlugin;
  int mode;
  int layoutEligible;
};

enum ncclSaiIbEndpointConsensusResult {
  ncclSaiIbEndpointConsensusReject = -1,
  ncclSaiIbEndpointConsensusUpstream = 0,
  ncclSaiIbEndpointConsensusPreserve = 1,
};

static inline uint64_t ncclSaiIbEndpointPolicySignature() {
  uint64_t hash = UINT64_C(14695981039346656037);
  static const char domain[] = "nccl-sai:ib-endpoint-policy:v1";
  hash = ncclSaiHashString(hash, domain);
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_IB_MERGE_NICS"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_NET_MERGE_LEVEL"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_NET_FORCE_MERGE"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_CROSS_NIC"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_NETDEVS_POLICY"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_SAI_DISABLE"));
  return hash;
}

static inline int ncclSaiResolveIbEndpointConsensus(
    bool pluginConsistent, bool policyConsistent, bool modeConsistent,
    bool internalIbPlugin, int mode, bool allLayoutsEligible) {
  if (!pluginConsistent || !policyConsistent || !modeConsistent) {
    return ncclSaiIbEndpointConsensusReject;
  }
  if (!internalIbPlugin || mode != ncclSaiIbEndpointMergeAutomatic) {
    return ncclSaiIbEndpointConsensusUpstream;
  }
  return allLayoutsEligible ? ncclSaiIbEndpointConsensusPreserve :
      ncclSaiIbEndpointConsensusUpstream;
}

static inline enum ncclSaiIbEndpointMergeMode
ncclSaiIbEndpointMergeControlMode(
    bool internalIbPlugin, bool mergeNicsExplicit, int mergeNics,
    bool mergeLevelExplicit, bool forceMergeExplicit, bool saiDisabled,
    int crossNic, bool automaticNetDevsPolicy) {
  if (!internalIbPlugin) return ncclSaiIbEndpointMergeUpstream;
  if (mergeNicsExplicit) {
    return mergeNics == 0 ? ncclSaiIbEndpointMergeExplicitDisabled :
        ncclSaiIbEndpointMergeUpstream;
  }
  if (mergeLevelExplicit || forceMergeExplicit || saiDisabled) {
    return ncclSaiIbEndpointMergeUpstream;
  }
  if (crossNic != 0 || !automaticNetDevsPolicy) {
    return ncclSaiIbEndpointMergeUpstream;
  }
  return ncclSaiIbEndpointMergeAutomatic;
}

struct ncclSaiIbEndpointPreflight {
  const char* adapterPath;
  uint64_t subnetPrefix;
  int port;
  int speed;
  int maxQp;
  int activeMtu;
  int provider;
  int dataDirect;
  int infiniband;
};

// This is intentionally only a cheap local preflight. After communicator-wide
// agreement it may request a temporary physical-endpoint probe, but the full
// topology/rank/GDR/path vote must accept that probe or rebuild with upstream
// NIC fusion.
static inline bool ncclSaiIbEndpointLayoutEligible(
    const struct ncclSaiIbEndpointPreflight* endpoints, int endpointCount) {
  if (endpoints == nullptr || endpointCount != 8) return false;

  uint64_t railSubnet[2] = { 0, 0 };
  const char* adapterPath[4] = { nullptr, nullptr, nullptr, nullptr };
  int adapterMembers[4] = { 0, 0, 0, 0 };
  int adapterPortMask[4] = { 0, 0, 0, 0 };
  int adapterCount = 0;
  const struct ncclSaiIbEndpointPreflight* reference = endpoints;

  for (int endpoint = 0; endpoint < endpointCount; endpoint++) {
    const struct ncclSaiIbEndpointPreflight* shape = endpoints + endpoint;
    if (shape->adapterPath == nullptr || shape->adapterPath[0] == '\0' ||
        shape->subnetPrefix == 0 ||
        (shape->port != 1 && shape->port != 2) || shape->speed <= 0 ||
        shape->maxQp <= 0 || shape->activeMtu <= 0 ||
        shape->dataDirect < 0 || shape->dataDirect > 1 ||
        shape->infiniband != 1 || shape->speed != reference->speed ||
        shape->maxQp != reference->maxQp ||
        shape->activeMtu != reference->activeMtu ||
        shape->provider != reference->provider ||
        shape->dataDirect != reference->dataDirect ||
        shape->infiniband != reference->infiniband ||
        !ncclSaiAccumulateRailSubnet(
            shape->port, shape->subnetPrefix, railSubnet)) {
      return false;
    }

    int adapter = -1;
    for (int group = 0; group < adapterCount; group++) {
      if (strcmp(adapterPath[group], shape->adapterPath) == 0) {
        adapter = group;
        break;
      }
    }
    if (adapter == -1) {
      if (adapterCount == 4) return false;
      adapter = adapterCount++;
      adapterPath[adapter] = shape->adapterPath;
    }

    const int portBit = 1 << (shape->port - 1);
    if ((adapterPortMask[adapter] & portBit) != 0) return false;
    adapterPortMask[adapter] |= portBit;
    adapterMembers[adapter]++;
  }

  if (adapterCount != 4 || !ncclSaiRailSubnetsComplete(railSubnet)) {
    return false;
  }
  for (int adapter = 0; adapter < adapterCount; adapter++) {
    if (adapterMembers[adapter] != 2 || adapterPortMask[adapter] != 0x3) {
      return false;
    }
  }
  return true;
}

static inline uint64_t ncclSaiLocalP2pPolicySignature() {
  uint64_t hash = UINT64_C(14695981039346656037);
  static const char domain[] = "nccl-sai:local-p2p-policy:v1";
  hash = ncclSaiHashString(hash, domain);
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_SAI_DISABLE"));
  hash = ncclSaiHashString(
      hash, ncclGetEnv("NCCL_SAI_LOCAL_P2P_SYS_ENABLE"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_P2P_DISABLE"));
  hash = ncclSaiHashString(hash, ncclGetEnv("NCCL_P2P_LEVEL"));
  return hash;
}

struct ncclSaiLocalP2pInfo {
  uint64_t policySignature;
  uint64_t topologyClass;
  uint64_t rankPairs;
  int eligible;
};

enum ncclSaiLocalP2pConsensusResult {
  ncclSaiLocalP2pConsensusReject = -1,
  ncclSaiLocalP2pConsensusDisabled = 0,
  ncclSaiLocalP2pConsensusEnabled = 1,
};

static inline int ncclSaiResolveLocalP2pConsensus(
    bool policyConsistent, bool allEligible, bool topologyConsistent) {
  if (!policyConsistent) return ncclSaiLocalP2pConsensusReject;
  return allEligible && topologyConsistent ?
      ncclSaiLocalP2pConsensusEnabled : ncclSaiLocalP2pConsensusDisabled;
}

static inline int ncclSaiPopcount64(uint64_t value) {
  int count = 0;
  while (value != 0) {
    value &= value - 1;
    count++;
  }
  return count;
}

static inline int ncclSaiFirstSet64(uint64_t value) {
  if (value == 0) return -1;
  int index = 0;
  while ((value & UINT64_C(1)) == 0) {
    value >>= 1;
    index++;
  }
  return index;
}

static inline bool ncclSaiFourGpuCliqueMasks(
    int nGpus, const uint64_t* peerMasks) {
  if (peerMasks == nullptr || nGpus < 4 || nGpus > 16 || nGpus % 4 != 0) {
    return false;
  }
  const uint64_t validMask = (UINT64_C(1) << nGpus) - 1;
  for (int gpu = 0; gpu < nGpus; gpu++) {
    const uint64_t self = UINT64_C(1) << gpu;
    const uint64_t peers = peerMasks[gpu];
    if ((peers & self) != 0 || (peers & ~validMask) != 0 ||
        ncclSaiPopcount64(peers) != 3) return false;
    const uint64_t clique = peers | self;
    if (ncclSaiPopcount64(clique) != 4) return false;
    for (int peer = 0; peer < nGpus; peer++) {
      if ((clique & (UINT64_C(1) << peer)) == 0) continue;
      if ((peerMasks[peer] | (UINT64_C(1) << peer)) != clique) return false;
      if ((peerMasks[peer] & self) == 0 && peer != gpu) return false;
    }
  }
  return true;
}

struct ncclSaiNetEndpointShape {
  uint64_t asic;
  uint64_t pciId;
  int port;
  float bw;
  float latency;
  int maxChannels;
  int gdrSupport;
  int collSupport;
};

static inline bool ncclSaiSymmetricDualPortAdapters(
    const struct ncclSaiNetEndpointShape* endpoints, int endpointCount,
    int* adapterGroup) {
  if (endpoints == nullptr || endpointCount != 8) return false;
  bool grouped[8] = { false };
  uint64_t groupPciId[4] = { 0, 0, 0, 0 };
  int groups = 0;
  for (int endpoint = 0; endpoint < endpointCount; endpoint++) {
    const struct ncclSaiNetEndpointShape* shape = endpoints + endpoint;
    if (shape->asic == 0 || shape->pciId == 0 ||
        (shape->port != 1 && shape->port != 2) || shape->bw <= 0 ||
        shape->latency < 0 || shape->maxChannels <= 0 ||
        shape->gdrSupport <= 0 ||
        shape->bw != endpoints[0].bw ||
        shape->latency != endpoints[0].latency ||
        shape->maxChannels != endpoints[0].maxChannels ||
        shape->gdrSupport != endpoints[0].gdrSupport ||
        shape->collSupport != endpoints[0].collSupport) return false;
    if (grouped[endpoint]) continue;
    if (groups >= 4) return false;
    int port1 = -1;
    int port2 = -1;
    int members = 0;
    for (int peer = endpoint; peer < endpointCount; peer++) {
      if (endpoints[peer].asic != shape->asic) continue;
      if (endpoints[peer].pciId != shape->pciId) return false;
      grouped[peer] = true;
      if (adapterGroup != nullptr) adapterGroup[peer] = groups;
      members++;
      if (endpoints[peer].port == 1) {
        if (port1 != -1) return false;
        port1 = peer;
      } else if (endpoints[peer].port == 2) {
        if (port2 != -1) return false;
        port2 = peer;
      } else {
        return false;
      }
    }
    if (members != 2 || port1 < 0 || port2 < 0) return false;
    for (int prior = 0; prior < groups; prior++) {
      if (groupPciId[prior] == shape->pciId) return false;
    }
    groupPciId[groups++] = shape->pciId;
  }
  return groups == 4;
}

static inline bool ncclSaiOrderDualPortRails(
    int firstPort, int secondPort, int* port1Index, int* port2Index) {
  if (port1Index == nullptr || port2Index == nullptr) return false;
  if (firstPort == 1 && secondPort == 2) {
    *port1Index = 0;
    *port2Index = 1;
    return true;
  }
  if (firstPort == 2 && secondPort == 1) {
    *port1Index = 1;
    *port2Index = 0;
    return true;
  }
  return false;
}

static inline bool ncclSaiGraphRailByChannelEnabled(
    bool policyEnabled, int crossNic, bool collNet) {
  return policyEnabled && crossNic == 0 && !collNet;
}

// The qualified SAI topology has two independent rails selected by channel
// parity.  Keep at least one P2P channel per rail for each NET peer; otherwise
// the upstream scale heuristic can collapse a 16-rank communicator to one
// channel per peer and leave grouped Send/Recv bandwidth on the table.
// Callers apply this only when the user did not explicitly configure
// nChannelsPerNetPeer.
static inline int ncclSaiP2pNetChannelsPerPeer(
    bool railByChannelEnabled, int upstreamChannels) {
  return railByChannelEnabled && upstreamChannels < 2 ? 2 : upstreamChannels;
}

// The supported dual-port topology exposes exactly two local network rails per
// GPU. Keep this override fail-closed so every other topology uses the upstream
// local-NET index.
static inline bool ncclSaiSelectLocalNetByChannel(
    bool policyEnabled, int channelId, int localNetCount, int* localNetIndex) {
  if (localNetIndex == nullptr || channelId < 0 || localNetCount != 2 ||
      !policyEnabled) return false;
  *localNetIndex = channelId % 2;
  return true;
}

static inline bool ncclSaiSelectGraphNetByChannel(bool policyEnabled, int channelId,
    const int64_t* localNets, int localNetCount, int64_t* selectedNetId) {
  if (localNets == nullptr || selectedNetId == nullptr) return false;

  int localNetIndex = 0;
  if (!ncclSaiSelectLocalNetByChannel(
          policyEnabled, channelId, localNetCount, &localNetIndex)) return false;
  *selectedNetId = localNets[localNetIndex];
  return true;
}

#endif
