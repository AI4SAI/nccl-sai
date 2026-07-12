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
#define NCCL_SAI_INIT_SCHEMA_TAG 0x5b7u
#define NCCL_SAI_PEER_NCCL_VERSION_MASK 0x000fffffu
#define NCCL_SAI_A2A_ISLAND_MIN_RANKS_DEFAULT 576

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
  if (mode > 0) return true;
  return fabricGroupsComplete;
}

static inline bool ncclSaiA2aGroupedLayoutAllowed(int topologyNodes, int groupNodes,
    int64_t multigroupMode, bool fabricGroupsComplete) {
  if (topologyNodes <= 0 || groupNodes <= 0) return false;
  if (multigroupMode > 0) return true;
  if (!fabricGroupsComplete) return false;
  return topologyNodes <= groupNodes || multigroupMode != 0;
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

static inline bool ncclSaiParseFabricGroupId(const char* value, uint64_t* id) {
  if (value == nullptr || id == nullptr || value[0] == '\0' || value[0] == '-' || value[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') return false;
  *id = (uint64_t)parsed;
  return true;
}

static inline bool ncclSaiSlurmFabricGroupId(
    const char* address, const char* pattern, uint64_t* id) {
  if (address == nullptr || pattern == nullptr || id == nullptr ||
      address[0] == '\0' || pattern[0] == '\0') return false;

  const char* addressPart = address;
  const char* patternPart = pattern;
  size_t leafSwitchPrefixLen = 0;
  bool sawNode = false;
  while (true) {
    const char* addressEnd = strchr(addressPart, '.');
    const char* patternEnd = strchr(patternPart, '.');
    size_t addressLen = addressEnd == nullptr ? strlen(addressPart) :
        (size_t)(addressEnd - addressPart);
    size_t patternLen = patternEnd == nullptr ? strlen(patternPart) :
        (size_t)(patternEnd - patternPart);
    if (addressLen == 0 || patternLen == 0 ||
        (addressEnd == nullptr) != (patternEnd == nullptr)) return false;

    bool isSwitch = patternLen == 6 && strncmp(patternPart, "switch", 6) == 0;
    bool isNode = patternLen == 4 && strncmp(patternPart, "node", 4) == 0;
    if (!isSwitch && !isNode) return false;
    if (isNode) {
      if (patternEnd != nullptr || leafSwitchPrefixLen == 0) return false;
      sawNode = true;
    } else {
      if (sawNode) return false;
      leafSwitchPrefixLen = (size_t)((addressPart + addressLen) - address);
    }

    if (addressEnd == nullptr) break;
    addressPart = addressEnd + 1;
    patternPart = patternEnd + 1;
  }
  if (!sawNode || leafSwitchPrefixLen == 0) return false;

  uint64_t hash = UINT64_C(14695981039346656037);
  static const char domain[] = "nccl-sai:slurm-topology:";
  for (size_t i = 0; i < sizeof(domain) - 1; i++) {
    hash ^= (unsigned char)domain[i];
    hash *= UINT64_C(1099511628211);
  }
  for (size_t i = 0; i < leafSwitchPrefixLen; i++) {
    hash ^= (unsigned char)address[i];
    hash *= UINT64_C(1099511628211);
  }
  *id = hash;
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

// Keep profile activation fail-closed. New product-family names must be added
// deliberately; explicit NCCL_SAI_*_ENABLE controls remain available for
// experimental or third-party layouts.
static inline bool ncclSaiFabricProfileEnabled() {
  const char* value = ncclGetEnv("NCCL_SAI_FABRIC_PROFILE");
  return ncclSaiProfileFamily(value, "ultrapod") || ncclSaiProfileFamily(value, "slimpod");
}

static inline bool ncclSaiFullMeshProfileEnabled() {
  const char* value = ncclGetEnv("NCCL_SAI_FABRIC_PROFILE");
  return ncclSaiProfileFamily(value, "ultrapod-fullmesh");
}

// The validated full-mesh topology exposes exactly two local network rails per
// GPU. Keep this override fail-closed so every other profile and topology uses
// the upstream local-NET index selected by ncclTopoGetLocalNet.
static inline bool ncclSaiSelectLocalNetByChannel(
    int channelId, int localNetCount, int* localNetIndex) {
  if (localNetIndex == nullptr || channelId < 0 || localNetCount != 2 ||
      !ncclSaiFullMeshProfileEnabled()) return false;
  *localNetIndex = channelId % 2;
  return true;
}

#endif
