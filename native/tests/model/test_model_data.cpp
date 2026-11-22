// model_data.h constants against weights.py / model.py / fast.py (CPU only): the visit plan, ring/full cache
// indices, tap visits and the projection layout.
#include "model/model_data.h"

#include <cstdio>

using namespace ingine::model;

int main() {
  bool ok = true;
  auto expect = [&](bool c, const char* what) {
    if (!c) std::printf("FAIL %s\n", what);
    ok &= c;
  };
  // weights.visit_plan: layers 0-5, 6-17 three times (passes 0, 1, 2), 18-23
  int v = 0;
  for (int l = 0; l < 6; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == 0, "visits 0-5");
  for (int p = 0; p < 3; ++p)
    for (int l = 6; l < 18; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == p, "loop visits");
  for (int l = 18; l < 24; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == 0, "visits 18-23");
  expect(v == NUM_VISITS, "48 visits");
  int nf = 0, ns = 0;
  for (int i = 0; i < NUM_VISITS; ++i) {
    const bool full = is_full_layer(VISITS[i].layer);
    expect(full ? FULL_INDEX[i] == nf++ && RING_INDEX[i] == -1 : RING_INDEX[i] == ns++ && FULL_INDEX[i] == -1,
           "cache indices");
  }
  expect(nf == NUM_FULL_VISITS && ns == NUM_SLIDING_VISITS, "8 full / 40 sliding visits");
  // model.TAP_VISITS = (5, 23, 41): layer 5, layer 11 of pass 1, layer 17 of pass 2
  expect(VISITS[TAP_VISITS[0]].layer == 5 && VISITS[TAP_VISITS[1]].layer == 11 && VISITS[TAP_VISITS[1]].pass == 1 &&
             VISITS[TAP_VISITS[2]].layer == 17 && VISITS[TAP_VISITS[2]].pass == 2,
         "tap visits");
  // SPEC row order: q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 | v 1024 | k_dyn 32 | v_dyn 32
  const int begin[] = {0, 3072, 6144, 6528, 6552, 7576, 8600, 8632, 8664};
  for (int s = 0; s <= NUM_SEGS; ++s) expect(SEG_BEGIN[s] == begin[s], "projection layout");
  expect(PROJ_ROWS_SLIDING == 8600 && PROJ_ROWS_FULL == 8664, "projection rows");
  std::printf("%s model_data constants\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
