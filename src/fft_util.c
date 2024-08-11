/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Modifications copyright 2024 Kevin Ahrendt.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "fft_util.h"

#include <stdio.h>

#ifdef USE_ESP32
#include <esp_heap_caps.h>
#endif

#include "kiss_fftr.h"

int FftPopulateState(struct FftState *state, size_t input_size)
{
  state->input_size = input_size;
  state->fft_size = 1;
  while (state->fft_size < state->input_size)
  {
    state->fft_size <<= 1;
  }

#ifdef USE_ESP32
  state->input =
      heap_caps_malloc(state->fft_size * sizeof(*state->input), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (state->input == NULL)
#endif
  {
    state->input = malloc(state->fft_size * sizeof(*state->input));
  }
  if (state->input == NULL)
  {
    fprintf(stderr, "Failed to alloc fft input buffer\n");
    return 0;
  }

#ifdef USE_ESP32
  state->output = heap_caps_malloc((state->fft_size / 2 + 1) * sizeof(*state->output) * 2,
                                   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (state->output == NULL)
#endif
  {
    state->output = malloc((state->fft_size / 2 + 1) * sizeof(*state->output) * 2);
  }
  if (state->output == NULL)
  {
    fprintf(stderr, "Failed to alloc fft output buffer\n");
    return 0;
  }

  // Ask kissfft how much memory it wants.
  size_t scratch_size = 0;
  kiss_fftr_cfg kfft_cfg = kiss_fftr_alloc(state->fft_size, 0, NULL, &scratch_size);
  if (kfft_cfg != NULL)
  {
    fprintf(stderr, "Kiss memory sizing failed.\n");
    return 0;
  }

#ifdef USE_ESP32
  state->scratch = heap_caps_malloc(scratch_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (state->scratch == NULL)
#endif
  {
    state->scratch = malloc(scratch_size);
  }

  if (state->scratch == NULL)
  {
    fprintf(stderr, "Failed to alloc fft scratch buffer\n");
    return 0;
  }
  state->scratch_size = scratch_size;
  // Let kissfft configure the scratch space we just allocated
  kfft_cfg = kiss_fftr_alloc(state->fft_size, 0, state->scratch, &scratch_size);
  if (kfft_cfg != state->scratch)
  {
    fprintf(stderr, "Kiss memory preallocation strategy failed.\n");
    return 0;
  }
  return 1;
}

void FftFreeStateContents(struct FftState *state)
{
  free(state->input);
  free(state->output);
  free(state->scratch);
}
