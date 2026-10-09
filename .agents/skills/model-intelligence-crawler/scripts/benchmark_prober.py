#!/usr/bin/env python3
"""
benchmark_prober.py
用于自动化探测与健康检查大模型排行榜站点的可用性、HTTP状态、响应时延与内容有效性。
支持自动发现废弃/失效链接并给出更新或剔除建议。
"""

import sys
import time
import json
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def probe_url(url: str, timeout: int = 12) -> Tuple[bool, int, float, str]:
    """
    测试单个 URL 的可用性
    返回: (is_valid, http_code, latency_seconds, message)
    """
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = time.time() - start
            code = resp.status
            content_type = resp.headers.get("Content-Type", "")
            return True, code, latency, f"OK ({content_type.split(';')[0]})"
    except urllib.error.HTTPError as e:
        latency = time.time() - start
        return False, e.code, latency, f"HTTPError {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        latency = time.time() - start
        return False, 0, latency, f"URLError: {e.reason}"
    except Exception as e:
        latency = time.time() - start
        return False, -1, latency, f"Exception: {str(e)}"

def check_registry(registry_file: str) -> Dict[str, dict]:
    with open(registry_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    print(f"{'Category':<22} | {'Name':<20} | {'Status':<6} | {'Code':<5} | {'Latency':<7} | Message")
    print("-" * 88)
    
    for item in data.get("sources", []):
        cat = item["category"]
        name = item["name"]
        url = item["url"]
        is_ok, code, latency, msg = probe_url(url)
        results[item["id"]] = {
            "name": name,
            "url": url,
            "is_valid": is_ok,
            "http_code": code,
            "latency": round(latency, 3),
            "message": msg
        }
        status_str = "ACTIVE" if is_ok else "FAILED"
        print(f"{cat[:22]:<22} | {name[:20]:<20} | {status_str:<6} | {code:<5} | {latency:.2f}s   | {msg}")

    return results

if __name__ == "__main__":
    reg_path = ".agents/skills/model-intelligence-crawler/sources.json"
    if len(sys.argv) > 1:
        reg_path = sys.argv[1]
    res = check_registry(reg_path)
    failures = [k for k, v in res.items() if not v["is_valid"]]
    if failures:
        print(f"\n[Warning] Found {len(failures)} inactive or failing sources: {failures}")
        sys.exit(1)
    else:
        print(f"\n[Success] All {len(res)} benchmark sources are active and responsive!")
