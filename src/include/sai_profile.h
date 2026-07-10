/*************************************************************************
 * Copyright (c) 2026, AI4SAI CONTRIBUTORS. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SAI_PROFILE_H_
#define NCCL_SAI_PROFILE_H_

#include "param.h"

#include <stddef.h>
#include <string.h>
#include <strings.h>

static inline bool ncclSaiProfileFamily(const char* value, const char* family) {
  if (value == nullptr || family == nullptr || value[0] == '\0' || family[0] == '\0') return false;
  size_t familyLen = strlen(family);
  if (strncasecmp(value, family, familyLen) != 0) return false;
  return value[familyLen] == '\0' || (value[familyLen] == '-' && value[familyLen+1] != '\0');
}

// Keep profile activation fail-closed. New product-family names must be added
// deliberately; explicit NCCL_SAI_*_ENABLE controls remain available for
// experimental or third-party layouts.
static inline bool ncclSaiFabricProfileEnabled() {
  const char* value = ncclGetEnv("NCCL_SAI_FABRIC_PROFILE");
  return ncclSaiProfileFamily(value, "ultrapod") || ncclSaiProfileFamily(value, "slimpod");
}

#endif
