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
  const ProfileCase cases[] = {
    {nullptr, false},
    {"", false},
    {"ultrapod", true},
    {"ULTRAPOD", true},
    {"ultrapod-site-a", true},
    {"ultrapod-", false},
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

  printf("profile activation checks passed: %zu\n", sizeof(cases) / sizeof(cases[0]));
  return 0;
}
