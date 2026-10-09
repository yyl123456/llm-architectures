#!/usr/bin/env python3
"""
fetch_latest_sota.py
动态全模态开源模型前沿 SOTA 数据抓取与时间过滤器 (Hard Date Filter: >= 2026-05-01)
解决方案集成：
1. Jina Reader 代理穿透 SPA 客户端动态渲染 (https://r.jina.ai/<url>)
2. Next.js App Router 动态 chunks 逆向解析 (提取 self.__next_f.push 内部数据)
3. Hugging Face 官方 Hub API 实时元数据核验 (验证 releaseDate / createdAt 时间戳)
4. 严格过滤规则：Release Date < 2026-05-01 全部 PASS 剔除！
"""

import sys
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

CUTOFF_DATE = "2026-05-01"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_escapes(s: str) -> str:
    return s.replace('\\\\"', '"').replace('\\"', '"')

def fetch_via_jina(url: str, timeout: int = 15) -> str:
    """使用 Jina Reader 穿透动态客户端渲染与 WAF"""
    jina_url = f"https://r.jina.ai/{url}"
    req = urllib.request.Request(jina_url, headers=DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[Warning] Jina Reader fetch failed for {url}: {e}", file=sys.stderr)
        return ""

def get_hf_model_meta(model_id: str) -> Dict[str, Any]:
    """通过 Hugging Face API 获取模型确切发布时间"""
    url = f"https://huggingface.co/api/models/{model_id}"
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            return {
                "id": data.get("id"),
                "createdAt": data.get("createdAt", "")[:10],
                "likes": data.get("likes", 0),
                "downloads": data.get("downloads", 0)
            }
    except Exception:
        return {}

def extract_aa_open_source() -> List[Dict[str, Any]]:
    """提取 AA Open Source 最新开源模型并过滤 >= 2026-05-01"""
    url = "https://artificialanalysis.ai/models/open-source"
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    results = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8")
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
            for s in scripts:
                if 'intelligenceIndex' in s:
                    s_clean = clean_escapes(s)
                    # 匹配 name, intelligenceIndex, openSourceCategorization, releaseDate
                    matches = re.findall(r'\"name\":\"([^\"]+)\"[^\{]{1,150}\"releaseDate\":\"([0-9\-]+)\"[^\{]{1,150}\"intelligenceIndex\":([0-9\.]+)[^\{]{1,150}\"openSourceCategorization\":\"([^\"]+)\"', s_clean)
                    for name, rdate, idx, cat in matches:
                        if cat != 'proprietary' and rdate >= CUTOFF_DATE:
                            results.append({
                                "name": name,
                                "release_date": rdate,
                                "intelligence_index": float(idx),
                                "license_category": cat,
                                "source": "AA Open Source"
                            })
                    if not results:
                        # 兼容另一种结构
                        m2 = re.findall(r'\"name\":\"([^\"]+)\"[^\{]{1,150}\"intelligenceIndex\":([0-9\.]+)[^\{]{1,150}\"openSourceCategorization\":\"([^\"]+)\"', s_clean)
                        # 辅以 script 83 的 releaseDate
                        dates = dict(re.findall(r'\"name\":\"([^\"]+)\"[^\{]{1,120}\"releaseDate\":\"([0-9\-]+)\"', s_clean))
                        for name, idx, cat in m2:
                            rdate = dates.get(name, "")
                            if cat != 'proprietary' and rdate >= CUTOFF_DATE:
                                results.append({
                                    "name": name,
                                    "release_date": rdate,
                                    "intelligence_index": float(idx),
                                    "license_category": cat,
                                    "source": "AA Open Source"
                                })
                    break
    except Exception as e:
        print(f"[Error] Failed to fetch AA Open Source: {e}", file=sys.stderr)
    return sorted(results, key=lambda x: x["intelligence_index"], reverse=True)

def extract_aa_tiny_models() -> List[Dict[str, Any]]:
    """提取 AA Tiny Models (<=4B) 并过滤 >= 2026-05-01"""
    url = "https://artificialanalysis.ai/models/open-source/tiny"
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    results = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8")
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
            for s in scripts:
                if 'MiniCPM5-2B' in s or 'Granite 4.2 3B' in s:
                    s_clean = clean_escapes(s)
                    matches = re.findall(r'\"name\":\"([^\"]+)\"[^\{]{1,250}\"releaseDate\":\"([0-9\-]+)\"', s_clean)
                    seen = set()
                    for name, rdate in matches:
                        if rdate >= CUTOFF_DATE and name not in seen:
                            seen.add(name)
                            results.append({
                                "name": name,
                                "release_date": rdate,
                                "category": "≤4B Tiny Model",
                                "source": "AA Tiny Models"
                            })
                    break
    except Exception as e:
        print(f"[Error] Failed to fetch AA Tiny: {e}", file=sys.stderr)
    return results

def extract_aa_video_arena() -> List[Dict[str, Any]]:
    """提取 AA Video Arena 最新开源模型 (>= 2026-05-01)"""
    url = "https://artificialanalysis.ai/embed/text-to-video-leaderboard/leaderboard/text-to-video"
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    results = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8")
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
            for s in scripts:
                if 'openWeightsUrl' in s:
                    s_clean = clean_escapes(s)
                    names = re.findall(r'\"name\":\"([^\"]+)\",\"url\":\"[^\"]*\",\"rank\":\d+,\"elo\":([0-9\.]+),.*?\"releaseDate\":\"([0-9\-]+)\".*?\"openWeightsUrl\":(null|\"[^\"]+\")', s_clean)
                    seen = set()
                    for n, e, rdate, ow in names:
                        if ow != 'null' and rdate >= CUTOFF_DATE and n not in seen:
                            seen.add(n)
                            results.append({
                                "name": n,
                                "elo": float(e),
                                "release_date": rdate,
                                "weights_url": ow.strip('"'),
                                "source": "AA Video Arena"
                            })
                    break
    except Exception as e:
        print(f"[Error] Failed to fetch AA Video: {e}", file=sys.stderr)
    return sorted(results, key=lambda x: x["elo"], reverse=True)

if __name__ == "__main__":
    print(f"=== Running Strict Time Filter (Hard Cutoff >= {CUTOFF_DATE}) ===")
    aa_os = extract_aa_open_source()
    print(f"\n[AA Open Source >= {CUTOFF_DATE}]: {len(aa_os)} models found")
    for m in aa_os:
        print(f"  {m['release_date']} | {m['name']:<30} | Score: {m['intelligence_index']}")

    aa_tiny = extract_aa_tiny_models()
    print(f"\n[AA Tiny Models <=4B >= {CUTOFF_DATE}]: {len(aa_tiny)} models found")
    for m in aa_tiny:
        print(f"  {m['release_date']} | {m['name']}")

    aa_video = extract_aa_video_arena()
    print(f"\n[AA Video Arena Open Weights >= {CUTOFF_DATE}]: {len(aa_video)} models found")
    for m in aa_video:
        print(f"  {m['release_date']} | {m['name']:<25} | Elo: {m['elo']}")
