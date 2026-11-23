"""Embed a small retrieval corpus and pass the actual vectors to the Rust example."""

import argparse
import json
import math
import os
import sys
import urllib.request
from array import array
from pathlib import Path


class StayOnEndpoint(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, status, reason, headers, new_url):
        return None


TAGS = [
    "Rust 工具链",
    "GitHub Actions 持续集成",
    "代码格式与静态检查",
    "图传播",
    "时间门控",
    "CSR 稀疏矩阵",
    "文献引用",
    "来源证明",
    "独立证据根",
    "非负最小二乘",
    "秩亏矩阵",
    "浮点精度",
    "上下文质量",
    "半衰期",
    "分片去重",
    "CPU 采样",
    "性能分析",
    "线程调度",
    "模型网关",
    "API 密钥",
    "向量模型",
    "软件许可证",
    "可选依赖",
    "Apache-2.0",
]
DOCUMENTS = [
    (
        "Rust 构建检查",
        [0, 1, 2],
        "项目固定 Rust 编译器版本，日常提交前用 cargo fmt 检查格式，用 Clippy 检查警告。构建服务器在 Linux、Windows、macOS 均运行工作区测试。",
    ),
    (
        "时间门控与传播",
        [3, 4, 5],
        "宿主冻结可见拓扑后交付 CSR 边权，查询可以逐边附加时间系数。某条边的系数为零时，脉冲和双场的质量都无法穿过该边。",
    ),
    (
        "独立证据计算",
        [6, 7, 8],
        "文献衍生条目可以依赖多个来源。计算候选支持时，排除候选自身的全部依赖根。联合前提只要包含一个被排除来源，该整组权重就不算独立。",
    ),
    (
        "非负锥求解",
        [9, 10, 11],
        "对近线性相关的向量求非负系数，活动集求解器使用 Householder QR 并在秩亏时回落到 SVD。输入为零时直接得到零系数。",
    ),
    (
        "带半衰期的上下文",
        [12, 13, 14],
        "上下文随片段距离指数衰减，按父文档和原文区间计算覆盖。把同一文档拆分成二十份不应增加上下文质量，重叠区间只分配一次。",
    ),
    (
        "CPU 热点排查",
        [15, 16, 17],
        "用采样器查看长时间运行下的调用栈，先将线程等待与应用 CPU 开销分开。定位热点后做单项修改，以相同数据和线程池重跑基准。",
    ),
    (
        "网关凭据与模型配置",
        [18, 19, 20],
        "实例通过配置引用模型网关端口和密钥文件。向量服务使用独立资源项声明名称和维度，凭据保存在本地，推理请求从宿主发起。",
    ),
    (
        "计算模块许可证",
        [21, 22, 23],
        "Apache-2.0 的 DTSC 模块可单独启用。包含 VCPToolBox 派生算法的模块使用 CC BY-NC-SA 4.0，通过 Cargo feature 控制可选依赖。",
    ),
    (
        "任务取消与复用",
        [3, 15, 0],
        "数值操作接受协作取消令牌。传播停止后释放当前调用的逻辑状态，保留工作区和返回缓冲容量供同一个调用方下一次复用。",
    ),
    (
        "候选标签残差",
        [9, 3, 20],
        "回调按本层残差提供候选标签，活动集计算正系数并去除已解释的向量成分。下一层查询处理尚未覆盖的方向。",
    ),
]
QUERIES = [
    ("怎样让一条过期的连接在检索时彻底不传能量？", 1),
    ("候选文章引用了两篇底稿，什么情况下要把证据整组扣掉？", 2),
    ("两个基向量几乎平行时，怎样稳健地拟合非负系数？", 3),
    ("为什么把一篇材料拆成很多块不能让它有更大的上下文权重？", 4),
    ("怎么用调用栈看出 CPU 真正在执行什么？", 5),
    ("如何设置远端嵌入接口的 token 和向量维数？", 6),
    ("想单独使用 DTSC，Cargo 应该怎样避开非商业许可依赖？", 7),
]


def gateway(endpoint, model, strings, dimensions):
    key = os.environ.get("NEXUS_EMBEDDING_KEY")
    if not key:
        raise SystemExit("Set NEXUS_EMBEDDING_KEY for the requested gateway.")
    opener = urllib.request.build_opener(StayOnEndpoint)
    vectors = []
    served_model = None
    for start in range(0, len(strings), 8):
        batch = strings[start : start + 8]
        payload = {"model": model, "input": batch}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/embeddings",
            body,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with opener.open(request, timeout=45) as response:
            payload = json.load(response)
        returned_model = payload.get("model") or model
        if served_model is not None and returned_model != served_model:
            raise ValueError("gateway model changed between embedding batches")
        served_model = returned_model
        rows = sorted(payload["data"], key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(batch))):
            raise ValueError(
                "gateway embedding indices differ from the submitted inputs"
            )
        vectors.extend(row["embedding"] for row in rows)
    return vectors, served_model


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--endpoint", required=True, help="OpenAI-compatible /v1 root")
    cli.add_argument("--model", required=True)
    cli.add_argument(
        "--dimensions", type=int, help="request a dimension supported by the model"
    )
    args = cli.parse_args()
    if args.dimensions is not None and args.dimensions < 1:
        cli.error("--dimensions must be positive")
    passages = TAGS + [document[2] for document in DOCUMENTS]
    queries = [query[0] for query in QUERIES]
    vectors, model = gateway(
        args.endpoint, args.model, passages + queries, args.dimensions
    )
    expected = len(passages) + len(queries)
    if len(vectors) != expected or not vectors or not vectors[0]:
        raise ValueError("embedding provider returned an incomplete matrix")
    dimension = len(vectors[0])
    if args.dimensions is not None and dimension != args.dimensions:
        raise ValueError(
            f"requested {args.dimensions} dimensions, received {dimension}"
        )
    if any(
        len(row) != dimension
        or not all(math.isfinite(value) for value in row)
        or not any(row)
        for row in vectors
    ):
        raise ValueError("embedding provider returned a malformed vector")
    result = {
        "name": "Live query smoke test",
        "producer": model,
        "dimension": dimension,
        "concepts": [
            {"id": i, "label": name, "aliases": []} for i, name in enumerate(TAGS)
        ],
        "sources": [
            {"id": i, "paths": [tags]} for i, (_, tags, _) in enumerate(DOCUMENTS)
        ],
        "relations": [],
        "candidates": [
            {
                "id": i,
                "document": i,
                "sources": [i],
                "paths": [{"source": i, "path": 0}],
            }
            for i in range(len(DOCUMENTS))
        ],
        "queries": [
            {"split": "smoke", "text": text, "relevant_documents": [expected]}
            for text, expected in QUERIES
        ],
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "fixture.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    for name, rows in [
        ("concepts", vectors[: len(TAGS)]),
        ("candidates", vectors[len(TAGS) : len(passages)]),
        ("queries", vectors[len(passages) :]),
    ]:
        values = array("f", (value for row in rows for value in row))
        if sys.byteorder != "little":
            values.byteswap()
        (args.destination / f"{name}.f32").write_bytes(values.tobytes())
    print(
        f"Generated {len(vectors)} real embeddings, {dimension} dimensions, model {model}"
    )


if __name__ == "__main__":
    main()
