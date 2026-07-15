/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sai_profile.h"
#include "topo_bcm.h"

#include <stdio.h>
#include <string.h>

static const char* testProfile = nullptr;
static const char* testMergeNics = nullptr;
static const char* testMergeLevel = nullptr;
static const char* testForceMerge = nullptr;
static const char* testCrossNic = nullptr;
static const char* testNetDevsPolicy = nullptr;
static const char* testHca = nullptr;
static const char* testSaiDisable = nullptr;
static const char* testLocalP2pEnable = nullptr;
static const char* testP2pDisable = nullptr;
static const char* testP2pLevel = nullptr;

const char* ncclGetEnv(const char* name) {
  if (strcmp(name, "NCCL_SAI_FABRIC_PROFILE") == 0) return testProfile;
  if (strcmp(name, "NCCL_IB_MERGE_NICS") == 0) return testMergeNics;
  if (strcmp(name, "NCCL_NET_MERGE_LEVEL") == 0) return testMergeLevel;
  if (strcmp(name, "NCCL_NET_FORCE_MERGE") == 0) return testForceMerge;
  if (strcmp(name, "NCCL_CROSS_NIC") == 0) return testCrossNic;
  if (strcmp(name, "NCCL_NETDEVS_POLICY") == 0) return testNetDevsPolicy;
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
  if (NCCL_SAI_INIT_SCHEMA_TAG != 0x5bcu) {
    fprintf(stderr, "initialization schema tag was not advanced\n");
    return 1;
  }
  if (NCCL_SAI_A2A_ISLAND_MIN_RANKS_DEFAULT != 576) {
    fprintf(stderr, "island scale guard changed unexpectedly\n");
    return 1;
  }
  if (ncclTopoBcmGen(UINT64_C(0x1000c0101000a096), 0) != 4 ||
      ncclTopoBcmGen(UINT64_C(0x1000c0101000a096), 1) != 4 ||
      ncclTopoBcmGen(UINT64_C(0x1000c0121000a096), 0) != 4 ||
      ncclTopoBcmGen(UINT64_C(0x1000c0121000a096), 1) != 4 ||
      ncclTopoBcmGen(UINT64_C(0x1000c03010000000), 0) != 5 ||
      ncclTopoBcmGen(UINT64_C(0x1000c03010001000), 1) != 5 ||
      ncclTopoBcmGen(UINT64_C(0x1000c0111000a096), 0) != 0) {
    fprintf(stderr, "BCM switch generation detection failed\n");
    return 1;
  }
  if (ncclSaiRailPathCapabilityClass(4, 4, 5) != 5 ||
      ncclSaiRailPathCapabilityClass(5, 4, 5) != 5 ||
      ncclSaiRailPathCapabilityClass(3, 4, 5) != -1 ||
      ncclSaiRailPathCapabilityClass(6, 4, 5) != -1 ||
      !ncclSaiRailPathPairEligible(4, 12.0f, 4, 12.0f, 4, 5) ||
      !ncclSaiRailPathPairEligible(5, 12.0f, 5, 12.0f, 4, 5) ||
      ncclSaiRailPathPairEligible(4, 12.0f, 5, 12.0f, 4, 5) ||
      ncclSaiRailPathPairEligible(4, 12.0f, 4, 6.0f, 4, 5) ||
      ncclSaiRailPathPairEligible(4, 0.0f, 4, 0.0f, 4, 5) ||
      ncclSaiRailPathPairEligible(3, 12.0f, 3, 12.0f, 4, 5) ||
      ncclSaiRailPathPairEligible(6, 12.0f, 6, 12.0f, 4, 5) ||
      !ncclSaiRailPathClassesCompatible(5, 12.0f, 4, 12.0f, 4, 5) ||
      !ncclSaiRailPathClassesCompatible(4, 12.0f, 5, 12.0f, 4, 5) ||
      ncclSaiRailPathClassesCompatible(5, 12.0f, 6, 12.0f, 4, 5) ||
      ncclSaiRailPathClassesCompatible(5, 0.0f, 4, 0.0f, 4, 5) ||
      ncclSaiRailPathClassesCompatible(5, 12.0f, 4, 6.0f, 4, 5)) {
    fprintf(stderr, "rail path qualification changed unexpectedly\n");
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
  if (ncclSaiP2pNetChannelsPerPeer(false, 1) != 1 ||
      ncclSaiP2pNetChannelsPerPeer(false, 2) != 2 ||
      ncclSaiP2pNetChannelsPerPeer(true, 1) != 2 ||
      ncclSaiP2pNetChannelsPerPeer(true, 2) != 2 ||
      ncclSaiP2pNetChannelsPerPeer(true, 4) != 4) {
    fprintf(stderr, "automatic dual-rail P2P channel floor failed\n");
    return 1;
  }
  uint64_t localP2pRankPairs =
      (UINT64_C(1) << ncclSaiRankPairIndex(8, 0, 7)) |
      (UINT64_C(1) << ncclSaiRankPairIndex(8, 3, 7));
  if (ncclSaiRankPairIndex(8, 0, 1) != 0 ||
      ncclSaiRankPairIndex(8, 0, 7) != 6 ||
      ncclSaiRankPairIndex(8, 1, 2) != 7 ||
      ncclSaiRankPairIndex(8, 3, 7) != 21 ||
      ncclSaiRankPairIndex(8, 4, 4) != -1 ||
      !ncclSaiRankPairSelected(localP2pRankPairs, 8, 7, 0) ||
      !ncclSaiRankPairSelected(localP2pRankPairs, 8, 3, 7) ||
      ncclSaiRankPairSelected(localP2pRankPairs, 8, 1, 7)) {
    fprintf(stderr, "local P2P rank-pair mask failed\n");
    return 1;
  }
  if (!ncclSaiA2aPlannerCanRun(true, true, true, false) ||
      !ncclSaiA2aPlannerCanRun(true, true, false, true) ||
      ncclSaiA2aPlannerCanRun(false, true, true, true) ||
      ncclSaiA2aPlannerCanRun(true, false, true, true) ||
      ncclSaiA2aPlannerCanRun(true, true, false, false)) {
    fprintf(stderr, "AlltoAll planner fast-path predicate failed\n");
    return 1;
  }
  if (ncclSaiDenseP2pCrossEpochAllowed(0, 0) ||
      ncclSaiDenseP2pCrossEpochAllowed(16, 16) ||
      ncclSaiDenseP2pCrossEpochAllowed(16, 0) ||
      !ncclSaiDenseP2pCrossEpochAllowed(32, 16) ||
      !ncclSaiDenseP2pCrossEpochAllowed(1024, 16)) {
    fprintf(stderr, "dense P2P cross-epoch scope failed\n");
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
  testMergeLevel = nullptr;
  testForceMerge = nullptr;
  testCrossNic = nullptr;
  testNetDevsPolicy = nullptr;
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

  struct ncclSaiRailInfo railInfo[2] = {};
  for (int rank = 0; rank < 2; rank++) {
    railInfo[rank].policySignature = 0x100;
    railInfo[rank].topologyClass = 0x200;
    railInfo[rank].railSubnet[0] = 0x1111;
    railInfo[rank].railSubnet[1] = 0x2222;
    railInfo[rank].eligible = 1;
  }
  int railEligibleRanks = 0;
  if (!ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks) ||
      railEligibleRanks != 2 ||
      ncclSaiRailInfoConsensus(nullptr, 2, &railEligibleRanks) ||
      ncclSaiRailInfoConsensus(railInfo, 0, &railEligibleRanks)) {
    fprintf(stderr, "valid rail consensus failed\n");
    return 1;
  }
  railInfo[1].eligible = 0;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks) ||
      railEligibleRanks != 1) {
    fprintf(stderr, "ineligible rail rank was accepted\n");
    return 1;
  }
  railInfo[1].eligible = 2;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks) ||
      railEligibleRanks != 1) {
    fprintf(stderr, "non-boolean rail eligibility was accepted\n");
    return 1;
  }
  railInfo[1].eligible = 1;
  railInfo[1].policySignature++;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "mismatched rail policy was accepted\n");
    return 1;
  }
  railInfo[1].policySignature = railInfo[0].policySignature;
  railInfo[1].topologyClass++;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "mismatched rail topology was accepted\n");
    return 1;
  }
  railInfo[1].topologyClass = railInfo[0].topologyClass;
  railInfo[1].railSubnet[0]++;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "mismatched rail-0 subnet was accepted\n");
    return 1;
  }
  railInfo[1].railSubnet[0] = railInfo[0].railSubnet[0];
  railInfo[1].railSubnet[1]++;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "mismatched rail-1 subnet was accepted\n");
    return 1;
  }
  railInfo[1].railSubnet[1] = railInfo[0].railSubnet[1];
  for (int rank = 0; rank < 2; rank++) railInfo[rank].policySignature = 0;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "missing rail policy identity was accepted\n");
    return 1;
  }
  for (int rank = 0; rank < 2; rank++) railInfo[rank].policySignature = 0x100;
  for (int rank = 0; rank < 2; rank++) railInfo[rank].topologyClass = 0;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "missing rail topology identity was accepted\n");
    return 1;
  }
  for (int rank = 0; rank < 2; rank++) railInfo[rank].topologyClass = 0x200;
  for (int rank = 0; rank < 2; rank++) railInfo[rank].railSubnet[1] = 0x1111;
  if (ncclSaiRailInfoConsensus(railInfo, 2, &railEligibleRanks)) {
    fprintf(stderr, "identical rail subnets were accepted\n");
    return 1;
  }
  for (int rank = 0; rank < 2; rank++) railInfo[rank].railSubnet[1] = 0x2222;

  struct ncclSaiIbEndpointPreflight ibEndpoints[9] = {};
  static const char* ibAdapterPaths[5] = {
    "/sys/devices/pci0000:00/0000:01:00.0",
    "/sys/devices/pci0000:00/0000:02:00.0",
    "/sys/devices/pci0000:80/0000:81:00.0",
    "/sys/devices/pci0000:80/0000:82:00.0",
    "/sys/devices/pci0000:c0/0000:c1:00.0",
  };
  for (int endpoint = 0; endpoint < 9; endpoint++) {
    ibEndpoints[endpoint].adapterPath = ibAdapterPaths[endpoint / 2];
    ibEndpoints[endpoint].port = endpoint % 2 + 1;
    ibEndpoints[endpoint].subnetPrefix =
        ibEndpoints[endpoint].port == 1 ? 0x1111 : 0x2222;
    ibEndpoints[endpoint].speed = 100000;
    ibEndpoints[endpoint].maxQp = 262144;
    ibEndpoints[endpoint].activeMtu = 5;
    ibEndpoints[endpoint].provider = 1;
    ibEndpoints[endpoint].dataDirect = 0;
    ibEndpoints[endpoint].infiniband = 1;
  }

  enum ncclSaiIbEndpointMergeMode ibMergeMode =
      ncclSaiIbEndpointMergeControlMode(
          true, false, 1, false, false, false, 0, true);
  if (ibMergeMode != ncclSaiIbEndpointMergeAutomatic ||
      !ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8) ||
      ncclSaiIbEndpointLayoutEligible(ibEndpoints, 7)) {
    fprintf(stderr, "automatic IB endpoint-probe decision failed\n");
    return 1;
  }

  ibMergeMode = ncclSaiIbEndpointMergeControlMode(
      true, true, 0, false, false, false, 2, false);
  if (ibMergeMode != ncclSaiIbEndpointMergeExplicitDisabled) {
    fprintf(stderr, "explicit NCCL_IB_MERGE_NICS=0 was not preserved\n");
    return 1;
  }
  ibMergeMode = ncclSaiIbEndpointMergeControlMode(
      true, true, 1, false, false, false, 0, true);
  if (ibMergeMode != ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "explicit NCCL_IB_MERGE_NICS=1 was not preserved\n");
    return 1;
  }
  if (ncclSaiIbEndpointMergeControlMode(
          true, false, 1, true, false, false, 0, true) !=
          ncclSaiIbEndpointMergeUpstream ||
      ncclSaiIbEndpointMergeControlMode(
          true, false, 1, false, true, false, 0, true) !=
          ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "explicit upstream NET fusion controls were not preserved\n");
    return 1;
  }
  ibMergeMode = ncclSaiIbEndpointMergeControlMode(
      true, false, 1, false, false, true, 0, true);
  if (ibMergeMode != ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "NCCL_SAI_DISABLE did not restore upstream IB fusion\n");
    return 1;
  }
  if (ncclSaiIbEndpointMergeControlMode(
          false, false, 1, false, false, false, 0, true) !=
          ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "external NET plugin entered SAI endpoint preflight\n");
    return 1;
  }
  if (ncclSaiIbEndpointMergeControlMode(
          true, false, 1, false, false, false, 1, true) !=
          ncclSaiIbEndpointMergeUpstream ||
      ncclSaiIbEndpointMergeControlMode(
          true, false, 1, false, false, false, 2, true) !=
          ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "nonzero effective NCCL_CROSS_NIC entered endpoint preflight\n");
    return 1;
  }
  if (ncclSaiIbEndpointMergeControlMode(
          true, false, 1, false, false, false, 0, false) !=
          ncclSaiIbEndpointMergeUpstream) {
    fprintf(stderr, "non-AUTO or invalid NETDEVS policy entered endpoint preflight\n");
    return 1;
  }

  if (ncclSaiResolveIbEndpointConsensus(
          true, true, true, true, ncclSaiIbEndpointMergeAutomatic, true) !=
          ncclSaiIbEndpointConsensusPreserve ||
      ncclSaiResolveIbEndpointConsensus(
          true, true, true, true, ncclSaiIbEndpointMergeAutomatic, false) !=
          ncclSaiIbEndpointConsensusUpstream ||
      ncclSaiResolveIbEndpointConsensus(
          true, true, true, true, ncclSaiIbEndpointMergeUpstream, true) !=
          ncclSaiIbEndpointConsensusUpstream ||
      ncclSaiResolveIbEndpointConsensus(
          true, true, true, false, ncclSaiIbEndpointMergeAutomatic, true) !=
          ncclSaiIbEndpointConsensusUpstream ||
      ncclSaiResolveIbEndpointConsensus(
          false, true, true, true, ncclSaiIbEndpointMergeAutomatic, true) !=
          ncclSaiIbEndpointConsensusReject ||
      ncclSaiResolveIbEndpointConsensus(
          true, false, true, true, ncclSaiIbEndpointMergeAutomatic, true) !=
          ncclSaiIbEndpointConsensusReject ||
      ncclSaiResolveIbEndpointConsensus(
          true, true, false, true, ncclSaiIbEndpointMergeAutomatic, true) !=
          ncclSaiIbEndpointConsensusReject) {
    fprintf(stderr, "communicator-wide IB endpoint consensus failed\n");
    return 1;
  }

  testMergeNics = nullptr;
  testMergeLevel = nullptr;
  testForceMerge = nullptr;
  testCrossNic = nullptr;
  testNetDevsPolicy = nullptr;
  testSaiDisable = nullptr;
  uint64_t endpointPolicySignature = ncclSaiIbEndpointPolicySignature();
  testMergeNics = "0";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "merge-NIC control was absent from endpoint policy signature\n");
    return 1;
  }
  testMergeNics = nullptr;
  testMergeLevel = "LOC";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "merge-level control was absent from endpoint policy signature\n");
    return 1;
  }
  testMergeLevel = nullptr;
  testForceMerge = "mlx5_0:1,mlx5_0:2";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "force-merge control was absent from endpoint policy signature\n");
    return 1;
  }
  testForceMerge = nullptr;
  testCrossNic = "0";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "cross-NIC control was absent from endpoint policy signature\n");
    return 1;
  }
  testCrossNic = nullptr;
  testNetDevsPolicy = "AUTO";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "NET-device policy was absent from endpoint policy signature\n");
    return 1;
  }
  testNetDevsPolicy = nullptr;
  testSaiDisable = "1";
  if (endpointPolicySignature == ncclSaiIbEndpointPolicySignature()) {
    fprintf(stderr, "SAI disable was absent from endpoint policy signature\n");
    return 1;
  }
  testSaiDisable = nullptr;

  if (!ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8) ||
      ncclSaiIbEndpointLayoutEligible(ibEndpoints, 7) ||
      ncclSaiIbEndpointLayoutEligible(ibEndpoints, 9)) {
    fprintf(stderr, "IB endpoint-count eligibility failed\n");
    return 1;
  }
  const char* savedAdapterPath0 = ibEndpoints[0].adapterPath;
  ibEndpoints[0].adapterPath = nullptr;
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "missing normalized PCI adapter path was accepted\n");
    return 1;
  }
  ibEndpoints[0].adapterPath = "";
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "empty normalized PCI adapter path was accepted\n");
    return 1;
  }
  ibEndpoints[0].adapterPath = savedAdapterPath0;
  uint64_t savedSubnet = ibEndpoints[1].subnetPrefix;
  for (int endpoint = 1; endpoint < 8; endpoint += 2) {
    ibEndpoints[endpoint].subnetPrefix = 0x1111;
  }
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "single-subnet IB layout was accepted\n");
    return 1;
  }
  for (int endpoint = 1; endpoint < 8; endpoint += 2) {
    ibEndpoints[endpoint].subnetPrefix = savedSubnet;
  }

  int savedPort = ibEndpoints[1].port;
  ibEndpoints[1].port = 1;
  ibEndpoints[1].subnetPrefix = 0x1111;
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "duplicate IB adapter port was accepted\n");
    return 1;
  }
  ibEndpoints[1].port = savedPort;
  ibEndpoints[1].subnetPrefix = savedSubnet;

  const char* savedAdapterPath = ibEndpoints[6].adapterPath;
  ibEndpoints[6].adapterPath = ibEndpoints[4].adapterPath;
  ibEndpoints[7].adapterPath = ibEndpoints[4].adapterPath;
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "wrong IB adapter count was accepted\n");
    return 1;
  }
  ibEndpoints[6].adapterPath = savedAdapterPath;
  ibEndpoints[7].adapterPath = savedAdapterPath;

  int savedSpeed = ibEndpoints[7].speed;
  ibEndpoints[7].speed *= 2;
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "mixed-speed IB layout was accepted\n");
    return 1;
  }
  ibEndpoints[7].speed = savedSpeed;

  int savedMtu = ibEndpoints[7].activeMtu;
  ibEndpoints[7].activeMtu--;
  if (ncclSaiIbEndpointLayoutEligible(ibEndpoints, 8)) {
    fprintf(stderr, "mixed-capability IB layout was accepted\n");
    return 1;
  }
  ibEndpoints[7].activeMtu = savedMtu;

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
