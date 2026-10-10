#!/usr/bin/env python3
"""
verify_index.py - 全局索引与相对链接有效性检测脚本
扫描所有 Markdown 文档中的相对文件/目录链接，验证目标是否存在，防止破损死链。
用法: python3 scripts/verify_index.py
"""

import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def find_markdown_files(root: Path):
    md_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # 忽略 .git
        if "/.git" in dirpath or dirpath.endswith("/.git"):
            continue
        for f in filenames:
            if f.endswith(".md"):
                md_files.append(Path(dirpath) / f)
    return md_files

def check_links_in_file(md_path: Path):
    errors = []
    content = md_path.read_text(encoding="utf-8", errors="ignore")
    # 匹配 Markdown 链接 [text](target)
    # 忽略纯 http(s) 链接、锚点 #、mailto
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    for match in pattern.finditer(content):
        link_text = match.group(1)
        target = match.group(2).strip()
        
        # 忽略外链与纯锚点
        if target.startswith("http://") or target.startswith("https://") or target.startswith("#") or target.startswith("mailto:"):
            continue
        
        # 去除链接中的锚点后缀，如 path/to/file.md#section
        clean_target = target.split("#")[0].strip()
        if not clean_target:
            continue
        
        # 移除角度括号 <...> 包装
        if clean_target.startswith("<") and clean_target.endswith(">"):
            clean_target = clean_target[1:-1]
        
        # 解析相对路径
        if clean_target.startswith("/"):
            resolved = ROOT_DIR / clean_target.lstrip("/")
        else:
            resolved = (md_path.parent / clean_target).resolve()
        
        if not resolved.exists():
            errors.append((clean_target, str(resolved)))
            
    return errors

def main():
    print(f"=== Running Index & Link Verification from Root: {ROOT_DIR} ===")
    md_files = find_markdown_files(ROOT_DIR)
    total_errors = 0
    checked_files = 0

    for md in md_files:
        checked_files += 1
        errors = check_links_in_file(md)
        rel_path = md.relative_to(ROOT_DIR)
        if errors:
            print(f"\n[FAIL] Broken links in {rel_path}:")
            for target, resolved in errors:
                print(f"  - Link target: '{target}' -> Does not exist: {resolved}")
            total_errors += len(errors)

    if total_errors == 0:
        print(f"\n[PASS] Verified {checked_files} Markdown files. All relative links and index entries are valid!")
        return 0
    else:
        print(f"\n[ERROR] Found {total_errors} broken link(s) across {checked_files} files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
