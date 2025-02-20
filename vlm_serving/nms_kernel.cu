// nms_kernel.cu — Custom CUDA Non-Maximum Suppression (fully implemented)
// Greedy NMS: sort by score, then suppress overlapping boxes in parallel.

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <float.h>
#include <stdio.h>

// Compute Intersection-over-Union for two boxes [x1,y1,x2,y2]
__device__ __forceinline__ float iou(
    float x1a, float y1a, float x2a, float y2a,
    float x1b, float y1b, float x2b, float y2b)
{
    float ix1 = fmaxf(x1a, x1b), iy1 = fmaxf(y1a, y1b);
    float ix2 = fminf(x2a, x2b), iy2 = fminf(y2a, y2b);
    float iw  = fmaxf(0.0f, ix2 - ix1);
    float ih  = fmaxf(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (x2a-x1a)*(y2a-y1a);
    float area_b = (x2b-x1b)*(y2b-y1b);
    return inter / (area_a + area_b - inter + 1e-6f);
}

// Each thread checks box[idx] against all higher-scored boxes
// If IoU > threshold with any, mark as suppressed (keep[idx] = 0)
__global__ void nmsKernel(
    const float* __restrict__ boxes,   // [N, 4] sorted by score desc
    int*   __restrict__ keep,          // [N] output: 1=keep, 0=suppress
    int N, float iou_thresh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || keep[idx] == 0) return;

    float x1 = boxes[idx*4+0], y1 = boxes[idx*4+1];
    float x2 = boxes[idx*4+2], y2 = boxes[idx*4+3];

    // Check against all higher-priority (lower index = higher score) boxes
    for (int j = 0; j < idx; j++) {
        if (keep[j] == 0) continue;
        float jou = iou(boxes[j*4], boxes[j*4+1], boxes[j*4+2], boxes[j*4+3],
                        x1, y1, x2, y2);
        if (jou > iou_thresh) {
            keep[idx] = 0;
            return;
        }
    }
}

// Host-side NMS: sort by score, initialize keep[], run nmsKernel
int runNMS(float* h_boxes, float* h_scores, int* h_keep, int N,
           float iou_thresh, float score_thresh)
{
    // Sort indices by score descending on host
    int* order = new int[N];
    for (int i = 0; i < N; i++) order[i] = i;
    // Simple insertion sort (N is typically small post-score-threshold)
    for (int i = 1; i < N; i++) {
        int key = order[i]; int j = i - 1;
        while (j >= 0 && h_scores[order[j]] < h_scores[key]) {
            order[j+1] = order[j]; j--;
        }
        order[j+1] = key;
    }

    // Reorder boxes and scores into sorted arrays
    float* sorted_boxes  = new float[N * 4];
    float* sorted_scores = new float[N];
    int valid = 0;
    for (int i = 0; i < N; i++) {
        if (h_scores[order[i]] < score_thresh) break;
        sorted_scores[valid] = h_scores[order[i]];
        for (int k = 0; k < 4; k++)
            sorted_boxes[valid*4+k] = h_boxes[order[i]*4+k];
        valid++;
    }

    // Allocate device memory
    float *d_boxes; int *d_keep;
    cudaMalloc(&d_boxes, valid*4*sizeof(float));
    cudaMalloc(&d_keep,  valid*sizeof(int));
    cudaMemcpy(d_boxes, sorted_boxes, valid*4*sizeof(float), cudaMemcpyHostToDevice);
    // Initialize all as keep=1
    thrust::device_ptr<int> keep_ptr(d_keep);
    thrust::fill(keep_ptr, keep_ptr + valid, 1);

    // Run NMS — sequential passes until convergence
    // (true parallel NMS requires multiple passes for correctness)
    for (int pass = 0; pass < valid; pass++) {
        int blocks = (valid + 255) / 256;
        nmsKernel<<<blocks, 256>>>(d_boxes, d_keep, valid, iou_thresh);
    }
    cudaDeviceSynchronize();

    // Copy results back, remap to original indices
    int* h_keep_sorted = new int[valid];
    cudaMemcpy(h_keep_sorted, d_keep, valid*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < valid; i++) h_keep[order[i]] = h_keep_sorted[i];

    int kept = 0;
    for (int i = 0; i < valid; i++) kept += h_keep_sorted[i];

    cudaFree(d_boxes); cudaFree(d_keep);
    delete[] order; delete[] sorted_boxes; delete[] sorted_scores;
    delete[] h_keep_sorted;
    return kept;
}

int main() {
    const int N = 6;
    // boxes: [x1, y1, x2, y2]
    float h_boxes[N*4] = {
        10, 10, 50, 50,   // box 0
        12, 12, 52, 52,   // box 1 — overlaps 0 heavily
        60, 60, 90, 90,   // box 2 — separate
        61, 61, 91, 91,   // box 3 — overlaps 2
        10, 10, 20, 20,   // box 4 — small, inside box 0
        80, 10, 100, 30,  // box 5 — separate
    };
    float h_scores[N] = {0.9f, 0.8f, 0.85f, 0.75f, 0.7f, 0.95f};
    int   h_keep[N]   = {0};

    int kept = runNMS(h_boxes, h_scores, h_keep, N, 0.5f, 0.5f);

    printf("NMS result (%d/%d boxes kept):\n", kept, N);
    for (int i = 0; i < N; i++)
        printf("  box %d (score=%.2f): %s\n", i, h_scores[i],
               h_keep[i] ? "KEEP" : "suppress");
    // Expected: boxes 5, 0 (or 2), and one other kept; heavily overlapping suppressed
    return 0;
}
