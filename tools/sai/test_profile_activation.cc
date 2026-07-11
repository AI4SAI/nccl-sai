/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sai_profile.h"

#include <stdio.h>
#include <string.h>

static const char* testProfile = nullptr;

const char* ncclGetEnv(const char* name) {
  return strcmp(name, "NCCL_SAI_FABRIC_PROFILE") == 0 ? testProfile : nullptr;
}

struct ProfileCase {
  const char* value;
  bool enabled;
};

int main() {
  const uint32_t ncclVersion = 22809;
  const uint32_t peerVersion = ncclSaiPeerVersion(ncclVersion);
  if ((peerVersion >> 20) != NCCL_SAI_INIT_SCHEMA_TAG ||
      (peerVersion & NCCL_SAI_PEER_NCCL_VERSION_MASK) != ncclVersion) {
    fprintf(stderr, "peer version schema packing failed\n");
    return 1;
  }
  if (ncclSaiProfileMinNchannels(true, false, 32) != 8 ||
      ncclSaiProfileMinNchannels(true, false, 4) != 4 ||
      ncclSaiProfileMinNchannels(true, true, 32) != 0 ||
      ncclSaiProfileMinNchannels(false, false, 32) != 0) {
    fprintf(stderr, "profile channel floor clipping failed\n");
    return 1;
  }
  if (ncclSaiA2aPlannerRoundWindow(-1, 16, 16) != 4 ||
      ncclSaiA2aPlannerRoundWindow(-1, 32, 16) != 4 ||
      ncclSaiA2aPlannerRoundWindow(-1, 48, 16) != 6 ||
      ncclSaiA2aPlannerRoundWindow(-1, 256, 16) != 16 ||
      ncclSaiA2aPlannerRoundWindow(-1, 336, 16) != 21 ||
      ncclSaiA2aPlannerRoundWindow(8, 256, 16) != 8 ||
      ncclSaiA2aPlannerRoundWindow(-1, 8, 16) != 4) {
    fprintf(stderr, "planner round-window selection failed\n");
    return 1;
  }
  if (!ncclSaiA2aPlannerGroupsComplete(16, 16) ||
      !ncclSaiA2aPlannerGroupsComplete(256, 16) ||
      !ncclSaiA2aPlannerGroupsComplete(272, 16) ||
      ncclSaiA2aPlannerGroupsComplete(8, 16) ||
      ncclSaiA2aPlannerGroupsComplete(270, 16) ||
      ncclSaiA2aPlannerGroupsComplete(16, 0)) {
    fprintf(stderr, "planner complete-group guard failed\n");
    return 1;
  }
  if (!ncclSaiA2aMultigroupAllowed(-1, true) ||
      ncclSaiA2aMultigroupAllowed(-1, false) ||
      ncclSaiA2aMultigroupAllowed(0, true) ||
      !ncclSaiA2aMultigroupAllowed(1, false)) {
    fprintf(stderr, "multigroup metadata gating failed\n");
    return 1;
  }
  uint64_t fabricGroupId = 0;
  if (!ncclSaiParseFabricGroupId("0", &fabricGroupId) || fabricGroupId != 0 ||
      !ncclSaiParseFabricGroupId("18446744073709551615", &fabricGroupId) ||
      fabricGroupId != UINT64_MAX || ncclSaiParseFabricGroupId("", &fabricGroupId) ||
      ncclSaiParseFabricGroupId("-1", &fabricGroupId) ||
      ncclSaiParseFabricGroupId("12x", &fabricGroupId) ||
      ncclSaiParseFabricGroupId(nullptr, &fabricGroupId)) {
    fprintf(stderr, "fabric group ID parsing failed\n");
    return 1;
  }

  const ProfileCase cases[] = {
    {nullptr, false},
    {"", false},
    {"ultrapod", true},
    {"ULTRAPOD", true},
    {"ultrapod-fullmesh", true},
    {"ultrapod-clos", true},
    {"ultrapod-site-a", true},
    {"ultrapod-site_a.2", true},
    {"ultrapod-", false},
    {"ultrapod--site", false},
    {"ultrapod- ", false},
    {"ultrapod-site/a", false},
    {"ultrapod_site_a", false},
    {"ultrapods", false},
    {"slimpod", true},
    {"SlimPOD-site-b", true},
    {"0", false},
    {"false", false},
    {"upstream", false},
    {"custom", false},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    testProfile = cases[i].value;
    bool actual = ncclSaiFabricProfileEnabled();
    if (actual != cases[i].enabled) {
      fprintf(stderr, "profile case %zu failed: value=%s expected=%d actual=%d\n",
          i, testProfile == nullptr ? "(null)" : testProfile, cases[i].enabled, actual);
      return 1;
    }
  }

  if (ncclSaiProfileFamily("ultrapod", nullptr) || ncclSaiProfileFamily("ultrapod", "")) {
    fprintf(stderr, "invalid family names must not match\n");
    return 1;
  }

  testProfile = "ultrapod-fullmesh";
  if (!ncclSaiFullMeshProfileEnabled()) {
    fprintf(stderr, "full-mesh profile was not recognized\n");
    return 1;
  }
  testProfile = "ultrapod-fullmesh-site_a.2";
  if (!ncclSaiFullMeshProfileEnabled()) {
    fprintf(stderr, "full-mesh profile variant was not recognized\n");
    return 1;
  }
  testProfile = "ultrapod-clos";
  if (ncclSaiFullMeshProfileEnabled()) {
    fprintf(stderr, "CLOS profile was misclassified as full-mesh\n");
    return 1;
  }
  testProfile = "slimpod";
  if (ncclSaiFullMeshProfileEnabled()) {
    fprintf(stderr, "SlimPOD profile was misclassified as full-mesh\n");
    return 1;
  }

  printf("profile activation checks passed: %zu\n", sizeof(cases) / sizeof(cases[0]));
  return 0;
}
