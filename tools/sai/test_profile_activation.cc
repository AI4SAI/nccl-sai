/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sai_profile.h"

#include <stdio.h>
#include <string.h>

static const char* testProfile = nullptr;
static const char* testMergeNics = nullptr;

const char* ncclGetEnv(const char* name) {
  if (strcmp(name, "NCCL_SAI_FABRIC_PROFILE") == 0) return testProfile;
  if (strcmp(name, "NCCL_IB_MERGE_NICS") == 0) return testMergeNics;
  return nullptr;
}

struct ProfileCase {
  const char* value;
  bool enabled;
};

struct LocalNetCase {
  const char* profile;
  const char* mergeNics;
  int channelId;
  int localNetCount;
  int upstreamIndex;
  bool selected;
  int expectedIndex;
};

struct GraphNetCase {
  const char* profile;
  const char* mergeNics;
  int channelId;
  int64_t graphNetId;
  int localNetCount;
  bool selected;
  int64_t expectedNetId;
};

int main() {
  const uint32_t ncclVersion = 22809;
  if (NCCL_SAI_INIT_SCHEMA_TAG != 0x5b7u) {
    fprintf(stderr, "initialization schema tag was not advanced\n");
    return 1;
  }
  if (NCCL_SAI_A2A_ISLAND_MIN_RANKS_DEFAULT != 576) {
    fprintf(stderr, "island scale guard changed unexpectedly\n");
    return 1;
  }
  const uint32_t peerVersion = ncclSaiPeerVersion(ncclVersion);
  if ((peerVersion >> 20) != NCCL_SAI_INIT_SCHEMA_TAG ||
      (peerVersion & NCCL_SAI_PEER_NCCL_VERSION_MASK) != ncclVersion) {
    fprintf(stderr, "peer version schema packing failed\n");
    return 1;
  }
  if (ncclSaiA2aPlannerRoundWindow(-1, 16, 16) != 4 ||
      ncclSaiA2aPlannerRoundWindow(-1, 32, 16) != 4 ||
      ncclSaiA2aPlannerRoundWindow(-1, 48, 16) != 6 ||
      ncclSaiA2aPlannerRoundWindow(-1, 256, 16) != 16 ||
      ncclSaiA2aPlannerRoundWindow(-1, 272, 16) != 17 ||
      ncclSaiA2aPlannerRoundWindow(8, 256, 16) != 8 ||
      ncclSaiA2aPlannerRoundWindow(-1, 8, 16) != 4) {
    fprintf(stderr, "planner round-window selection failed\n");
    return 1;
  }
  if (ncclSaiA2aPlannerRoundLimit(10, 100, 4) != 14 ||
      ncclSaiA2aPlannerRoundLimit(10, 100, 0) != 100 ||
      ncclSaiA2aPlannerRoundLimit(90, 100, 9) != 99 ||
      ncclSaiA2aPlannerRoundLimit(90, 100, 10) != 100 ||
      ncclSaiA2aPlannerRoundLimit(INT_MAX - 10, INT_MAX, 4) != INT_MAX - 6 ||
      ncclSaiA2aPlannerRoundLimit(INT_MAX - 2, INT_MAX, 4) != INT_MAX ||
      ncclSaiA2aPlannerRoundLimit(1, INT_MAX, INT_MAX) != INT_MAX ||
      ncclSaiA2aPlannerRoundLimit(INT_MAX, INT_MAX, 1) != INT_MAX ||
      ncclSaiA2aPlannerRoundLimit(-1, 100, 4) != -1 ||
      ncclSaiA2aPlannerRoundLimit(101, 100, 4) != 101) {
    fprintf(stderr, "planner round-limit overflow guard failed\n");
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
  if (ncclSaiA2aGroupedLayoutAllowed(16, 16, -1, false) ||
      !ncclSaiA2aGroupedLayoutAllowed(16, 16, -1, true) ||
      !ncclSaiA2aGroupedLayoutAllowed(16, 16, 0, true) ||
      !ncclSaiA2aGroupedLayoutAllowed(32, 16, -1, true) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 16, -1, false) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 16, 0, true) ||
      !ncclSaiA2aGroupedLayoutAllowed(32, 16, 1, false) ||
      !ncclSaiA2aGroupedLayoutAllowed(8, 16, 1, false) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 0, -1, true)) {
    fprintf(stderr, "grouped-layout eligibility failed\n");
    return 1;
  }
  if (!ncclSaiA2aRaggedLayoutAllowed(16, 16, 1, true, false, 4, 4) ||
      !ncclSaiA2aRaggedLayoutAllowed(30, 16, 1, true, false, 3, 12) ||
      ncclSaiA2aRaggedLayoutAllowed(16, 16, -1, true, false, 4, 4) ||
      ncclSaiA2aRaggedLayoutAllowed(15, 16, 1, true, false, 3, 8) ||
      ncclSaiA2aRaggedLayoutAllowed(16, 16, 0, true, false, 4, 4) ||
      ncclSaiA2aRaggedLayoutAllowed(16, 16, 1, false, false, 4, 4) ||
      ncclSaiA2aRaggedLayoutAllowed(16, 16, 1, true, true, 1, 16) ||
      ncclSaiA2aRaggedLayoutAllowed(20, 16, 1, true, false, 2, 17) ||
      ncclSaiA2aRaggedLayoutAllowed(20, 16, 1, true, false, 1, 10)) {
    fprintf(stderr, "ragged-layout eligibility failed\n");
    return 1;
  }
  if (ncclSaiP2pFabricScheduleAllowed(16, 16, -1, true) ||
      !ncclSaiP2pFabricScheduleAllowed(32, 16, -1, true) ||
      !ncclSaiP2pFabricScheduleAllowed(272, 16, -1, true) ||
      ncclSaiP2pFabricScheduleAllowed(32, 16, -1, false) ||
      ncclSaiP2pFabricScheduleAllowed(32, 16, 1, false) ||
      ncclSaiP2pFabricScheduleAllowed(32, 16, 0, true) ||
      ncclSaiP2pFabricScheduleAllowed(24, 16, -1, true)) {
    fprintf(stderr, "P2P fabric schedule eligibility failed\n");
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
  uint64_t slurmGroupA1 = 0, slurmGroupA2 = 0, slurmGroupB = 0;
  if (!ncclSaiSlurmFabricGroupId(
          "root.fabric.group-a.node-1", "switch.switch.switch.node", &slurmGroupA1) ||
      !ncclSaiSlurmFabricGroupId(
          "root.fabric.group-a.node-2", "switch.switch.switch.node", &slurmGroupA2) ||
      !ncclSaiSlurmFabricGroupId(
          "root.fabric.group-b.node-3", "switch.switch.switch.node", &slurmGroupB) ||
      slurmGroupA1 != slurmGroupA2 || slurmGroupA1 == slurmGroupB ||
      ncclSaiSlurmFabricGroupId(
          "root.fabric.node-1", "switch.switch.switch.node", &slurmGroupA1) ||
      ncclSaiSlurmFabricGroupId(
          "root.fabric.group-a.node-1", "switch.switch.node", &slurmGroupA1) ||
      ncclSaiSlurmFabricGroupId(
          "root..group-a.node-1", "switch.switch.switch.node", &slurmGroupA1) ||
      ncclSaiSlurmFabricGroupId(
          "root.fabric.group-a.node-1", "switch.block.switch.node", &slurmGroupA1) ||
      ncclSaiSlurmFabricGroupId(
          "root.fabric.group-a", "switch.switch.switch", &slurmGroupA1) ||
      ncclSaiSlurmFabricGroupId(nullptr, nullptr, &slurmGroupA1)) {
    fprintf(stderr, "Slurm fabric group parsing failed\n");
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

  const LocalNetCase localNetCases[] = {
    {"ultrapod-fullmesh", "0", 0, 2, 1, true, 0},
    {"ultrapod-fullmesh", "0", 1, 2, 0, true, 1},
    {"ultrapod-fullmesh", "0", 6, 2, 1, true, 0},
    {"ultrapod-fullmesh-site_a.2", "0", 7, 2, 0, true, 1},
    {"ultrapod-fullmesh", nullptr, 1, 2, 0, false, 0},
    {"ultrapod-fullmesh", "1", 1, 2, 0, false, 0},
    {"ultrapod-fullmesh", "00", 1, 2, 0, false, 0},
    {nullptr, "0", 1, 2, 0, false, 0},
    {"custom", "0", 0, 2, 1, false, 1},
    {"ultrapod", "0", 1, 2, 0, false, 0},
    {"slimpod", "0", 0, 2, 1, false, 1},
    {"ultrapod-fullmesh", "0", 1, 1, 0, false, 0},
    {"ultrapod-fullmesh", "0", 2, 3, 2, false, 2},
    {"ultrapod-fullmesh", "0", 3, 4, 3, false, 3},
    {"ultrapod-fullmesh", "0", -1, 2, 1, false, 1},
  };

  for (size_t i = 0; i < sizeof(localNetCases) / sizeof(localNetCases[0]); i++) {
    const LocalNetCase& test = localNetCases[i];
    testProfile = test.profile;
    testMergeNics = test.mergeNics;
    int localNetIndex = test.upstreamIndex;
    bool selected = ncclSaiSelectLocalNetByChannel(
        test.channelId, test.localNetCount, &localNetIndex);
    if (selected != test.selected || localNetIndex != test.expectedIndex) {
      fprintf(stderr,
          "local NET case %zu failed: profile=%s merge=%s channel=%d count=%d "
          "expected_selected=%d actual_selected=%d expected_index=%d actual_index=%d\n",
          i, testProfile == nullptr ? "(null)" : testProfile,
          testMergeNics == nullptr ? "(null)" : testMergeNics, test.channelId,
          test.localNetCount, test.selected, selected, test.expectedIndex,
          localNetIndex);
      return 1;
    }
  }

  testProfile = "ultrapod-fullmesh";
  testMergeNics = "0";
  if (ncclSaiSelectLocalNetByChannel(0, 2, nullptr)) {
    fprintf(stderr, "null local NET output unexpectedly selected a rail\n");
    return 1;
  }

  const int64_t localNets[] = {0x10, 0x20, 0x30};
  const GraphNetCase graphNetCases[] = {
    {"ultrapod-fullmesh", "0", 0, 0x10, 2, true, 0x10},
    {"ultrapod-fullmesh", "0", 1, 0x10, 2, true, 0x20},
    {"ultrapod-fullmesh", "0", 6, 0x20, 2, true, 0x10},
    {"ultrapod-fullmesh", "0", 1, 0x30, 2, false, 0x30},
    {"ultrapod-fullmesh", "1", 1, 0x10, 2, false, 0x10},
    {"ultrapod-fullmesh", "0", 1, 0x10, 3, false, 0x10},
    {"ultrapod", "0", 1, 0x10, 2, false, 0x10},
  };

  for (size_t i = 0; i < sizeof(graphNetCases) / sizeof(graphNetCases[0]); i++) {
    const GraphNetCase& test = graphNetCases[i];
    testProfile = test.profile;
    testMergeNics = test.mergeNics;
    int64_t selectedNetId = test.graphNetId;
    bool selected = ncclSaiSelectGraphNetByChannel(test.channelId,
        test.graphNetId, localNets, test.localNetCount, &selectedNetId);
    if (selected != test.selected || selectedNetId != test.expectedNetId) {
      fprintf(stderr,
          "graph NET case %zu failed: profile=%s merge=%s channel=%d "
          "graph_net=%lx expected_selected=%d actual_selected=%d "
          "expected_net=%lx actual_net=%lx\n",
          i, testProfile == nullptr ? "(null)" : testProfile,
          testMergeNics == nullptr ? "(null)" : testMergeNics,
          test.channelId, (unsigned long)test.graphNetId, test.selected, selected,
          (unsigned long)test.expectedNetId, (unsigned long)selectedNetId);
      return 1;
    }
  }

  if (ncclSaiSelectGraphNetByChannel(0, 0x10, nullptr, 2, nullptr)) {
    fprintf(stderr, "null graph NET inputs unexpectedly selected a rail\n");
    return 1;
  }

  printf("profile activation checks passed: %zu profiles, %zu local NET cases, "
         "%zu graph NET cases\n",
      sizeof(cases) / sizeof(cases[0]),
      sizeof(localNetCases) / sizeof(localNetCases[0]),
      sizeof(graphNetCases) / sizeof(graphNetCases[0]));
  return 0;
}
