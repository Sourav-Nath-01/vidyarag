"""
One-shot script: adds "type" field to annotations.jsonl.
Types: conceptual | theoretical | procedural | code

Run: python data/eval/add_types.py
"""
import json
from pathlib import Path

# Hand-labeled types for every selected query
# Keys = annotation id
TYPES = {
    # ── DSA ──────────────────────────────────────────────────
    "dsa_001": "conceptual",    # definition of algorithm
    "dsa_002": "theoretical",   # Big-O notation
    "dsa_003": "procedural",    # merge sort step by step
    "dsa_004": "theoretical",   # recurrence / master theorem (skipped)
    "dsa_005": "procedural",    # BST insertion algorithm
    "dsa_006": "conceptual",    # heap data structure
    "dsa_007": "procedural",    # heapify in heap sort
    "dsa_008": "conceptual",    # dynamic programming / memoization
    "dsa_009": "procedural",    # Dijkstra algorithm
    "dsa_010": "theoretical",   # amortized analysis (skipped)
    "dsa_011": "conceptual",    # red black tree properties
    "dsa_012": "procedural",    # quicksort partition
    "dsa_013": "conceptual",    # hash function (skipped)
    "dsa_014": "procedural",    # DFS algorithm
    "dsa_015": "conceptual",    # connected components
    # ── Deep Learning ─────────────────────────────────────────
    "dl_001": "theoretical",    # backpropagation gradients
    "dl_002": "theoretical",    # vanishing gradient problem
    "dl_003": "conceptual",     # CNN how does it work
    "dl_004": "conceptual",     # dropout regularisation
    "dl_005": "conceptual",     # batch normalisation
    "dl_006": "conceptual",     # LSTM vanishing gradients
    "dl_007": "procedural",     # SGD how it works
    "dl_008": "conceptual",     # attention mechanism
    "dl_009": "conceptual",     # transfer learning (skipped)
    "dl_010": "conceptual",     # GAN training (skipped)
    "dl_011": "procedural",     # max pooling computed
    "dl_012": "conceptual",     # word2vec embeddings
    # ── OS ───────────────────────────────────────────────────
    "os_001": "conceptual",     # virtual memory / paging
    "os_002": "procedural",     # round robin scheduling
    "os_003": "conceptual",     # deadlock conditions
    "os_004": "procedural",     # banker algorithm
    "os_005": "conceptual",     # semaphores and mutex
    "os_006": "conceptual",     # virtual address space
    "os_007": "conceptual",     # page fault (skipped)
    "os_008": "procedural",     # LRU page replacement
    "os_009": "procedural",     # fork system call
    "os_010": "conceptual",     # thrashing (skipped)
    "os_011": "conceptual",     # segmentation vs paging
    "os_012": "procedural",     # compile gcc executable
    # ── DBMS ─────────────────────────────────────────────────
    "dbms_001": "conceptual",   # data model / abstraction
    "dbms_002": "procedural",   # ER diagram modelling
    "dbms_003": "code",         # SQL SELECT FROM WHERE
    "dbms_004": "procedural",   # normalisation 1NF
    "dbms_005": "conceptual",   # ACID properties
    "dbms_006": "procedural",   # join operations relational algebra
    "dbms_007": "conceptual",   # B+ tree index
    "dbms_008": "procedural",   # two phase locking
    "dbms_009": "procedural",   # write ahead logging recovery
    "dbms_010": "code",         # SQL GROUP BY HAVING aggregates
    "dbms_011": "theoretical",  # functional dependency Armstrong axioms
    "dbms_012": "procedural",   # ER to relational schema
    # ── ML ───────────────────────────────────────────────────
    "ml_001": "theoretical",    # bias variance tradeoff
    "ml_002": "theoretical",    # SVM margin maximisation
    "ml_003": "theoretical",    # perceptron convergence
    "ml_004": "theoretical",    # naive bayes conditional independence
    "ml_005": "conceptual",     # k-NN classification
    "ml_006": "conceptual",     # decision tree information gain
    "ml_007": "conceptual",     # cross validation overfitting
    "ml_008": "procedural",     # logistic regression gradient descent
    "ml_009": "procedural",     # PCA dimensionality reduction
    "ml_010": "procedural",     # EM algorithm clustering
    "ml_011": "conceptual",     # ensemble / random forest / boosting
    "ml_012": "procedural",     # k-means clustering
    # ── CN ───────────────────────────────────────────────────
    "cn_001": "conceptual",     # OSI model 7 layers
    "cn_002": "procedural",     # TCP 3-way handshake
    "cn_003": "conceptual",     # IP addressing / CIDR
    "cn_004": "conceptual",     # congestion control TCP
    "cn_005": "conceptual",     # distance vector routing
    "cn_006": "procedural",     # ARP resolve IP to MAC
    "cn_007": "conceptual",     # TCP vs UDP
    "cn_008": "conceptual",     # 802.11 CSMA/CA
    "cn_009": "conceptual",     # NAT network address translation
    "cn_010": "procedural",     # DNS resolution
    "cn_011": "conceptual",     # Ethernet frame / CSMA/CD
    "cn_012": "conceptual",     # digital to analog modem
    # ── COA ──────────────────────────────────────────────────
    "coa_001": "conceptual",    # pipelining in processor
    "coa_002": "conceptual",    # cache memory direct mapped
    "coa_003": "conceptual",    # RISC vs CISC ISA
    "coa_004": "conceptual",    # pipeline hazards
    "coa_005": "conceptual",    # cache coherence multiprocessor
    "coa_006": "procedural",    # two's complement
    "coa_007": "conceptual",    # datapath and control unit
    "coa_008": "conceptual",    # memory hierarchy locality
    "coa_009": "conceptual",    # branch prediction
    # ── CV ───────────────────────────────────────────────────
    "cv_001": "procedural",     # edge detection Sobel Canny
    "cv_002": "conceptual",     # image convolution kernel
    "cv_003": "procedural",     # SIFT feature detection
    "cv_004": "procedural",     # optical flow estimation
    "cv_005": "conceptual",     # image segmentation
    "cv_006": "procedural",     # camera calibration intrinsic
    "cv_007": "procedural",     # HOG features
    "cv_008": "conceptual",     # stereo vision depth
    # ── DAA ──────────────────────────────────────────────────
    "daa_001": "procedural",    # Kruskal MST
    "daa_002": "theoretical",   # NP completeness polynomial reduction
    "daa_003": "procedural",    # Bellman-Ford negative edges
    "daa_004": "theoretical",   # greedy exchange argument proof
    "daa_005": "procedural",    # activity selection greedy
    "daa_006": "procedural",    # knapsack DP
    "daa_007": "procedural",    # topological sort
    "daa_008": "theoretical",   # TSP approximation
}

ann_path = Path(__file__).parent / "annotations.jsonl"
out_path = ann_path  # overwrite in place

lines = ann_path.read_text(encoding="utf-8").splitlines()
updated = []
changed = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    qid = obj.get("id", "")
    if qid in TYPES and "type" not in obj:
        obj["type"] = TYPES[qid]
        changed += 1
    updated.append(json.dumps(obj, ensure_ascii=False))

out_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
print(f"Done — added 'type' to {changed} annotations → {out_path}")
