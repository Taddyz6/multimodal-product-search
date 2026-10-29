# Performance measurement boundaries

Historical local CPU measurements use an Intel Core i5-1038NG7, four physical/eight logical cores, macOS 14.8.7 and four numerical threads. They are not averaged with Colab, T4 or A100 results.

| Boundary | Includes |
|---|---|
| A | Cached query vectors → Top 100 retrieval → Prepared28 → scaler → ListNet → sorted candidates |
| B | Real-time SBERT+CLIP query encoding, A2 retrieval/ranking and product field construction |
| C | Client HTTP start through complete response body, including queueing and serialization |

A, B and C are measured separately. Their means are not added, and C−B is not a pure network measurement. Product display images load independently.

The final local record contains four repeat observations, not a completed five-repeat attempt. The mean of the four round means was 102.159 ms for A, 164.026 ms for B and 184.525 ms for C. Each boundary had 4,000 requests. Pooled P95 values were 204.994 ms, 337.103 ms and 370.082 ms respectively; these do not establish a stable 200 ms P95 for live HTTP search. Full conditions, repeat variation and interrupted scenarios are in the technical report.

Concurrency tests distinguish Hot64 repeated-query workloads from Broad1000 independent dispatcher keys, Burst64 from 600-second sustained closed-loop loads, and empty from workload-conditioned product LRUs. Warmed cache means clear the product LRU, run the fixed query list twice, record hit/miss counts, and then begin timing. Hot and Broad do not inherit one another's cache state. No full-query answer or query-vector cache is enabled.

Burst uses 20 gated waves of 64 requests, with drain and a 2-second gap between waves. Sustained uses 64 clients, one outstanding HTTP request per client, zero think time, no retry, a 600-second issue window and full drain. Some historical Broad sustained windows ended early; they are recorded as incomplete. HTTP 504/429 responses and Hot coalescing are retained. Successful HTTP responses are not automatically independent compute jobs.

The dispatcher keeps one CPU computation active. A client timeout or disconnect does not prove that computation has ended. Shutdown first stops admission and drains the dispatcher, then closes and waits for the owned encoder worker. Source cleanup does not change these semantics or the original queue/deadline/LRU values.

This repository does not automatically replay the large performance matrix. Its correctness tests and 20-query refactor check are not new performance samples. Use the original raw experiment archive for independent historical statistical reproduction.
