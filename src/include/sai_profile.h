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
#define NCCL_SAI_INIT_SCHEMA_TAG 0x5a8u
#define NCCL_SAI_PEER_NCCL_VERSION_MASK 0x000fffffu

static inline uint32_t ncclSaiPeerVersion(uint32_t ncclVersion) {
  return (NCCL_SAI_INIT_SCHEMA_TAG << 20) | (ncclVersion & NCCL_SAI_PEER_NCCL_VERSION_MASK);
}

static inline int ncclSaiProfileMinNchannels(bool fullMeshDefaults, bool explicitMin,
    int effectiveMaxNchannels) {
  if (!fullMeshDefaults || explicitMin || effectiveMaxNchannels <= 0) return 0;
  return effectiveMaxNchannels < 8 ? effectiveMaxNchannels : 8;
}

static inline int ncclSaiA2aPlannerRoundWindow(int64_t configuredRounds,
    int topologyNodes, int fabricNodes) {
  if (configuredRounds > 0 && configuredRounds <= INT_MAX) return (int)configuredRounds;
  if (topologyNodes <= 0 || fabricNodes <= 0 || topologyNodes % fabricNodes != 0) return 4;
  int fabricGroups = topologyNodes / fabricNodes;
  int completeCycles = (4 + fabricGroups - 1) / fabricGroups;
  return fabricGroups * completeCycles;
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

#endif
