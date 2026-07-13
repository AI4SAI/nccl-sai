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
static const char* testCrossNic = nullptr;
static const char* testHca = nullptr;
static const char* testSaiDisable = nullptr;
static const char* testLocalP2pEnable = nullptr;
static const char* testP2pDisable = nullptr;
static const char* testP2pLevel = nullptr;

const char* ncclGetEnv(const char* name) {
  if (strcmp(name, "NCCL_SAI_FABRIC_PROFILE") == 0) return testProfile;
  if (strcmp(name, "NCCL_IB_MERGE_NICS") == 0) return testMergeNics;
  if (strcmp(name, "NCCL_CROSS_NIC") == 0) return testCrossNic;
  if (strcmp(name, "NCCL_IB_HCA") == 0) return testHca;
  if (strcmp(name, "NCCL_SAI_DISABLE") == 0) return testSaiDisable;
  if (strcmp(name, "NCCL_SAI_LOCAL_P2P_SYS_ENABLE") == 0) return testLocalP2pEnable;
  if (strcmp(name, "NCCL_P2P_DISABLE") == 0) return testP2pDisable;
  if (strcmp(name, "NCCL_P2P_LEVEL") == 0) return testP2pLevel;
  return nullptr;
}

struct LocalNetCase {
  const char* profile;
  const char* mergeNics;
  bool topologyEligible;
  int channelId;
  int localNetCount;
  int upstreamIndex;
  bool selected;
  int expectedIndex;
};

struct GraphNetCase {
  const char* profile;
  const char* mergeNics;
  bool topologyEligible;
  int channelId;
  int64_t graphNetId;
  int localNetCount;
  bool selected;
  int64_t expectedNetId;
};

struct GraphRailActivationCase {
  const char* profile;
  const char* mergeNics;
  bool topologyEligible;
  int crossNic;
  bool collNet;
  bool enabled;
};

int main() {
  const uint32_t ncclVersion = 22809;
  if (NCCL_SAI_INIT_SCHEMA_TAG != 0x5bbu) {
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
  if (ncclSaiA2aActivationSourceFor(false, -1, false) !=
          ncclSaiA2aActivationDisabled ||
      ncclSaiA2aActivationSourceFor(true, -1, false) !=
          ncclSaiA2aActivationAutomatic ||
      ncclSaiA2aActivationSourceFor(true, 1, false) !=
          ncclSaiA2aActivationExplicit ||
      ncclSaiA2aActivationSourceFor(true, -1, true) !=
          ncclSaiA2aActivationExplicit ||
      !ncclSaiA2aDefaultPolicyIsAutomatic(-1, true, true) ||
      ncclSaiA2aDefaultPolicyIsAutomatic(0, true, true) ||
      ncclSaiA2aDefaultPolicyIsAutomatic(-1, false, true) ||
      ncclSaiA2aDefaultPolicyIsAutomatic(-1, true, false)) {
    fprintf(stderr, "automatic policy provenance failed\n");
    return 1;
  }
  if (!ncclSaiLocalP2pAutomaticLevelAllowed(-2, -2, 1) ||
      !ncclSaiLocalP2pAutomaticLevelAllowed(1, -2, 1) ||
      ncclSaiLocalP2pAutomaticLevelAllowed(0, -2, 1) ||
      ncclSaiLocalP2pAutomaticLevelAllowed(2, -2, 1) ||
      ncclSaiLocalP2pAutomaticLevelAllowed(3, -2, 1) ||
      ncclSaiLocalP2pAutomaticLevelAllowed(4, -2, 1)) {
    fprintf(stderr, "automatic local P2P upstream-level compatibility failed\n");
    return 1;
  }
  if (ncclSaiResolveLocalP2pConsensus(false, true, true) !=
          ncclSaiLocalP2pConsensusReject ||
      ncclSaiResolveLocalP2pConsensus(true, false, true) !=
          ncclSaiLocalP2pConsensusDisabled ||
      ncclSaiResolveLocalP2pConsensus(true, true, false) !=
          ncclSaiLocalP2pConsensusDisabled ||
      ncclSaiResolveLocalP2pConsensus(true, true, true) !=
          ncclSaiLocalP2pConsensusEnabled) {
    fprintf(stderr, "automatic local P2P consensus resolution failed\n");
    return 1;
  }
  int64_t selectiveMask =
      ncclSaiA2aAutomaticPlanner | ncclSaiA2aAutomaticIsland;
  if (ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, selectiveMask, ncclSaiA2aAutomaticBase) != 1 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, selectiveMask, ncclSaiA2aAutomaticPlanner) != 0 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, selectiveMask, ncclSaiA2aAutomaticIsland) != 0 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, selectiveMask, ncclSaiA2aAutomaticP2pSchedule) != 1 ||
      ncclSaiA2aSourceAfterAutomaticFallback(
          ncclSaiA2aActivationExplicit, selectiveMask) !=
          ncclSaiA2aActivationExplicit) {
    fprintf(stderr, "selective automatic fallback failed\n");
    return 1;
  }
  int64_t fullAutomaticMask =
      ncclSaiA2aAutomaticBase | ncclSaiA2aAutomaticPlanner |
      ncclSaiA2aAutomaticIsland | ncclSaiA2aAutomaticP2pSchedule;
  if (ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, fullAutomaticMask, ncclSaiA2aAutomaticBase) != 0 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, fullAutomaticMask, ncclSaiA2aAutomaticPlanner) != 0 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, fullAutomaticMask, ncclSaiA2aAutomaticIsland) != 0 ||
      ncclSaiA2aPolicyValueAfterAutomaticFallback(
          1, fullAutomaticMask, ncclSaiA2aAutomaticP2pSchedule) != 0 ||
      ncclSaiA2aSourceAfterAutomaticFallback(
          ncclSaiA2aActivationAutomatic, fullAutomaticMask) !=
          ncclSaiA2aActivationDisabled) {
    fprintf(stderr, "full automatic fallback failed\n");
    return 1;
  }
  if (ncclSaiA2aGroupedLayoutAllowed(16, 16, -1, false) ||
      !ncclSaiA2aGroupedLayoutAllowed(16, 16, -1, true) ||
      !ncclSaiA2aGroupedLayoutAllowed(16, 16, 0, true) ||
      !ncclSaiA2aGroupedLayoutAllowed(32, 16, -1, true) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 16, -1, false) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 16, 0, true) ||
      ncclSaiA2aGroupedLayoutAllowed(32, 16, 1, false) ||
      !ncclSaiA2aGroupedLayoutAllowed(32, 16, 1, true) ||
      ncclSaiA2aGroupedLayoutAllowed(8, 16, 1, false) ||
      ncclSaiA2aGroupedLayoutAllowed(8, 16, 1, true) ||
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
  if (ncclSaiP2pFabricScheduleRequested(false, false) ||
      ncclSaiP2pFabricScheduleRequested(false, true) ||
      ncclSaiP2pFabricScheduleRequested(true, false) ||
      !ncclSaiP2pFabricScheduleRequested(true, true)) {
    fprintf(stderr, "P2P fabric schedule base-policy gating failed\n");
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
  // Normal production activation requires no NCCL_SAI_* variable and does not
  // use scheduler metadata or HCA strings as activation signals. The strict
  // runtime topology classifier is the activation gate.
  testProfile = nullptr;
  testHca = nullptr;
  testMergeNics = nullptr;
  testCrossNic = nullptr;
  if (!ncclSaiAutomaticRailCapabilityEnabled(true) ||
      ncclSaiAutomaticRailCapabilityEnabled(false) ||
      !ncclSaiRailByChannelRequested(true, 0)) {
    fprintf(stderr, "zero-SAI-variable automatic activation failed\n");
    return 1;
  }
  testSaiDisable = "1";
  if (ncclSaiAutomaticRailCapabilityEnabled(true) ||
      ncclSaiRailByChannelRequested(true, 0)) {
    fprintf(stderr, "global SAI disable did not override automatic activation\n");
    return 1;
  }
  testSaiDisable = nullptr;
  uint64_t automaticSignature = ncclSaiRailPolicySignature();
  uint64_t localP2pSignature = ncclSaiLocalP2pPolicySignature();
  testHca = "mlx5_0:1,mlx5_1:1";
  testMergeNics = "1";
  testProfile = "custom-stale-label";
  if (!ncclSaiAutomaticRailCapabilityEnabled(true) ||
      ncclSaiAutomaticRailCapabilityEnabled(false) ||
      !ncclSaiRailByChannelRequested(true, 0) ||
      automaticSignature != ncclSaiRailPolicySignature() ||
      localP2pSignature != ncclSaiLocalP2pPolicySignature()) {
    fprintf(stderr, "external metadata incorrectly became an automatic-policy signal\n");
    return 1;
  }
  testCrossNic = "0";
  if (automaticSignature != ncclSaiRailPolicySignature() ||
      ncclSaiRailByChannelRequested(true, 2)) {
    fprintf(stderr, "effective cross-NIC gating failed\n");
    return 1;
  }
  testLocalP2pEnable = "0";
  if (localP2pSignature == ncclSaiLocalP2pPolicySignature()) {
    fprintf(stderr, "local P2P policy control was absent from its signature\n");
    return 1;
  }
  testLocalP2pEnable = nullptr;
  testP2pLevel = "NVL";
  if (localP2pSignature == ncclSaiLocalP2pPolicySignature()) {
    fprintf(stderr, "upstream P2P policy was absent from local policy signature\n");
    return 1;
  }
  testP2pLevel = nullptr;
  testProfile = "upstream";
  if (!ncclSaiAutomaticRailCapabilityEnabled(true) ||
      !ncclSaiRailByChannelRequested(true, 0) ||
      automaticSignature != ncclSaiRailPolicySignature() ||
      localP2pSignature != ncclSaiLocalP2pPolicySignature()) {
    fprintf(stderr, "legacy profile label incorrectly changed automatic policy\n");
    return 1;
  }
  int port1Index = -1;
  int port2Index = -1;
  if (!ncclSaiOrderDualPortRails(1, 2, &port1Index, &port2Index) ||
      port1Index != 0 || port2Index != 1 ||
      !ncclSaiOrderDualPortRails(2, 1, &port1Index, &port2Index) ||
      port1Index != 1 || port2Index != 0 ||
      ncclSaiOrderDualPortRails(1, 1, &port1Index, &port2Index) ||
      ncclSaiOrderDualPortRails(1, 2, nullptr, &port2Index)) {
    fprintf(stderr, "dual-port rail ordering failed\n");
    return 1;
  }
  uint64_t railSubnet[2] = {0, 0};
  if (!ncclSaiAccumulateRailSubnet(1, 0x1111, railSubnet) ||
      !ncclSaiAccumulateRailSubnet(2, 0x2222, railSubnet) ||
      !ncclSaiAccumulateRailSubnet(1, 0x1111, railSubnet) ||
      !ncclSaiRailSubnetsComplete(railSubnet) ||
      ncclSaiAccumulateRailSubnet(2, 0x3333, railSubnet) ||
      ncclSaiAccumulateRailSubnet(3, 0x4444, railSubnet)) {
    fprintf(stderr, "dual-rail subnet validation failed\n");
    return 1;
  }
  const uint64_t sameSubnet[2] = {0x1111, 0x1111};
  const uint64_t missingSubnet[2] = {0x1111, 0};
  if (ncclSaiRailSubnetsComplete(sameSubnet) ||
      ncclSaiRailSubnetsComplete(missingSubnet)) {
    fprintf(stderr, "invalid rail subnet set was accepted\n");
    return 1;
  }

  uint64_t cliqueMasks[16] = { 0 };
  for (int gpu = 0; gpu < 16; gpu++) {
    int base = (gpu / 4) * 4;
    cliqueMasks[gpu] = (UINT64_C(0xf) << base) ^ (UINT64_C(1) << gpu);
  }
  if (!ncclSaiFourGpuCliqueMasks(4, cliqueMasks) ||
      !ncclSaiFourGpuCliqueMasks(8, cliqueMasks) ||
      !ncclSaiFourGpuCliqueMasks(16, cliqueMasks) ||
      ncclSaiFourGpuCliqueMasks(3, cliqueMasks)) {
    fprintf(stderr, "valid four-GPU clique shape was rejected\n");
    return 1;
  }
  uint64_t savedMask = cliqueMasks[0];
  cliqueMasks[0] &= ~(UINT64_C(1) << 3);
  if (ncclSaiFourGpuCliqueMasks(4, cliqueMasks)) {
    fprintf(stderr, "incomplete NVLink island was accepted\n");
    return 1;
  }
  cliqueMasks[0] = savedMask;
  cliqueMasks[0] |= UINT64_C(1) << 4;
  if (ncclSaiFourGpuCliqueMasks(8, cliqueMasks)) {
    fprintf(stderr, "cross-island NVLink edge was accepted\n");
    return 1;
  }
  cliqueMasks[0] = savedMask;

  struct ncclSaiNetEndpointShape endpointShape[8];
  for (int endpoint = 0; endpoint < 8; endpoint++) {
    endpointShape[endpoint].asic = 0x100 + endpoint / 2;
    endpointShape[endpoint].pciId = 0x200 + endpoint / 2;
    endpointShape[endpoint].port = endpoint % 2 + 1;
    endpointShape[endpoint].bw = 12.5f;
    endpointShape[endpoint].latency = 0.0f;
    endpointShape[endpoint].maxChannels = 32;
    endpointShape[endpoint].gdrSupport = 1;
    endpointShape[endpoint].collSupport = 0;
  }
  int adapterGroup[8];
  if (!ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup) ||
      adapterGroup[0] != adapterGroup[1] ||
      adapterGroup[0] == adapterGroup[2] ||
      ncclSaiSymmetricDualPortAdapters(endpointShape, 4, adapterGroup)) {
    fprintf(stderr, "valid symmetric dual-port adapter shape was rejected\n");
    return 1;
  }
  endpointShape[1].port = 1;
  if (ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup)) {
    fprintf(stderr, "duplicate physical port was accepted\n");
    return 1;
  }
  endpointShape[1].port = 2;
  endpointShape[2].pciId = endpointShape[0].pciId;
  endpointShape[3].pciId = endpointShape[0].pciId;
  if (ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup)) {
    fprintf(stderr, "reused PCI adapter identity was accepted\n");
    return 1;
  }
  endpointShape[2].pciId = 0x201;
  endpointShape[3].pciId = 0x201;
  endpointShape[6].bw = 25.0f;
  if (ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup)) {
    fprintf(stderr, "asymmetric adapter bandwidth was accepted\n");
    return 1;
  }
  endpointShape[6].bw = 12.5f;
  endpointShape[5].collSupport = 1;
  if (ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup)) {
    fprintf(stderr, "asymmetric adapter collective capability was accepted\n");
    return 1;
  }
  endpointShape[5].collSupport = 0;
  endpointShape[7].gdrSupport = 0;
  if (ncclSaiSymmetricDualPortAdapters(endpointShape, 8, adapterGroup)) {
    fprintf(stderr, "non-GDR endpoint was accepted\n");
    return 1;
  }
  testHca = nullptr;
  testMergeNics = nullptr;
  testCrossNic = nullptr;
  testSaiDisable = nullptr;
  testLocalP2pEnable = nullptr;
  testP2pDisable = nullptr;
  testP2pLevel = nullptr;
  testProfile = nullptr;

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
  testProfile = "unrelated-profile";
  if (ncclSaiFullMeshProfileEnabled()) {
    fprintf(stderr, "SlimPOD profile was misclassified as full-mesh\n");
    return 1;
  }

  const GraphRailActivationCase graphRailActivationCases[] = {
    {"ultrapod-fullmesh", "0", true, 0, false, true},
    {"ultrapod-fullmesh-site_a.2", "0", true, 0, false, true},
    {"ultrapod-fullmesh", nullptr, true, 0, false, true},
    {"ultrapod-fullmesh", "1", false, 0, false, false},
    {"ultrapod-fullmesh", "0", true, 1, false, false},
    {"ultrapod-fullmesh", "0", true, 2, false, false},
    {"ultrapod-fullmesh", "0", true, 0, true, false},
    {"ultrapod", "0", true, 0, false, true},
    {nullptr, "0", true, 0, false, true},
    {nullptr, "0", false, 0, false, false},
  };

  for (size_t i = 0;
       i < sizeof(graphRailActivationCases) /
           sizeof(graphRailActivationCases[0]);
       i++) {
    const GraphRailActivationCase& test = graphRailActivationCases[i];
    testProfile = test.profile;
    testMergeNics = test.mergeNics;
    bool policyEnabled = ncclSaiRailByChannelRequested(
        test.topologyEligible, test.crossNic);
    bool actual = ncclSaiGraphRailByChannelEnabled(
        policyEnabled, test.crossNic, test.collNet);
    if (actual != test.enabled) {
      fprintf(stderr,
          "graph rail activation case %zu failed: profile=%s merge=%s "
          "cross_nic=%d coll_net=%d expected=%d actual=%d\n",
          i, testProfile == nullptr ? "(null)" : testProfile,
          testMergeNics == nullptr ? "(null)" : testMergeNics,
          test.crossNic, test.collNet, test.enabled, actual);
      return 1;
    }
  }

  const LocalNetCase localNetCases[] = {
    {"ultrapod-fullmesh", "0", true, 0, 2, 1, true, 0},
    {"ultrapod-fullmesh", "0", true, 1, 2, 0, true, 1},
    {"ultrapod-fullmesh", "0", true, 6, 2, 1, true, 0},
    {"ultrapod-fullmesh-site_a.2", "0", true, 7, 2, 0, true, 1},
    {"ultrapod-fullmesh", nullptr, true, 1, 2, 0, true, 1},
    {"ultrapod-fullmesh", "1", false, 1, 2, 0, false, 0},
    {"ultrapod-fullmesh", "00", true, 1, 2, 0, true, 1},
    {nullptr, "0", true, 1, 2, 0, true, 1},
    {"custom", "0", true, 0, 2, 1, true, 0},
    {"ultrapod", "0", true, 1, 2, 0, true, 1},
    {"unrelated-profile", "0", true, 0, 2, 1, true, 0},
    {"ultrapod-fullmesh", "0", true, 1, 1, 0, false, 0},
    {"ultrapod-fullmesh", "0", true, 2, 3, 2, false, 2},
    {"ultrapod-fullmesh", "0", true, 3, 4, 3, false, 3},
    {"ultrapod-fullmesh", "0", true, -1, 2, 1, false, 1},
  };

  for (size_t i = 0; i < sizeof(localNetCases) / sizeof(localNetCases[0]); i++) {
    const LocalNetCase& test = localNetCases[i];
    testProfile = test.profile;
    testMergeNics = test.mergeNics;
    bool policyEnabled = ncclSaiRailByChannelRequested(
        test.topologyEligible, 0);
    int localNetIndex = test.upstreamIndex;
    bool selected = ncclSaiSelectLocalNetByChannel(
        policyEnabled, test.channelId, test.localNetCount, &localNetIndex);
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
  if (ncclSaiSelectLocalNetByChannel(true, 0, 2, nullptr)) {
    fprintf(stderr, "null local NET output unexpectedly selected a rail\n");
    return 1;
  }

  const int64_t localNets[] = {0x10, 0x20, 0x30};
  const GraphNetCase graphNetCases[] = {
    {"ultrapod-fullmesh", "0", true, 0, 0x10, 2, true, 0x10},
    {"ultrapod-fullmesh", "0", true, 1, 0x10, 2, true, 0x20},
    {"ultrapod-fullmesh", "0", true, 6, 0x20, 2, true, 0x10},
    {"ultrapod-fullmesh", "0", true, 1, 0x30, 2, true, 0x20},
    {"ultrapod-fullmesh", "1", false, 1, 0x10, 2, false, 0x10},
    {"ultrapod-fullmesh", "0", true, 1, 0x10, 3, false, 0x10},
    {"ultrapod", "0", true, 1, 0x10, 2, true, 0x20},
    {nullptr, nullptr, true, 1, 0x10, 2, true, 0x20},
  };

  for (size_t i = 0; i < sizeof(graphNetCases) / sizeof(graphNetCases[0]); i++) {
    const GraphNetCase& test = graphNetCases[i];
    testProfile = test.profile;
    testMergeNics = test.mergeNics;
    bool policyEnabled = ncclSaiRailByChannelRequested(
        test.topologyEligible, 0);
    int64_t selectedNetId = test.graphNetId;
    bool selected = ncclSaiSelectGraphNetByChannel(
        policyEnabled, test.channelId, localNets,
        test.localNetCount, &selectedNetId);
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

  if (ncclSaiSelectGraphNetByChannel(true, 0, nullptr, 2, nullptr)) {
    fprintf(stderr, "null graph NET inputs unexpectedly selected a rail\n");
    return 1;
  }

  printf("SAI activation checks passed: "
         "%zu graph rail activation cases, %zu local NET cases, "
         "%zu graph NET cases\n",
      sizeof(graphRailActivationCases) / sizeof(graphRailActivationCases[0]),
      sizeof(localNetCases) / sizeof(localNetCases[0]),
      sizeof(graphNetCases) / sizeof(graphNetCases[0]));
  return 0;
}
