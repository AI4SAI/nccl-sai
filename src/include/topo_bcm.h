/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_BCM_H_
#define NCCL_TOPO_BCM_H_

#include <stdint.h>

// BCM Gen4 switches can expose the same full-bandwidth fabric through
// multiple device IDs. They all need the same two-level flattening before
// graph search. Gen5 encodes the hierarchy level in the device identity.
static inline int ncclTopoBcmGen(uint64_t id, int level) {
  const uint64_t masked = id & UINT64_C(0xfffffffffffff000);
  if (masked == UINT64_C(0x1000c0101000a000) ||
      masked == UINT64_C(0x1000c0121000a000)) return 4;
  if (masked == (UINT64_C(0x1000c03010000000) |
      (uint64_t)level * UINT64_C(0x1000))) return 5;
  return 0;
}

#endif
