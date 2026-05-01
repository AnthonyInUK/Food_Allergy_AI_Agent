#!/usr/bin/env python3
"""
Food Allergy AI Agent - CLI 版本
主要 UI 已迁移到 Next.js 前端 (frontend/)
"""

import argparse
from dotenv import load_dotenv

load_dotenv()

from langsmith_config import configure_langsmith_tracing_compat

configure_langsmith_tracing_compat()

from graph_logic import query_with_graph, get_cache_stats, clear_all_caches


def main():
    parser = argparse.ArgumentParser(description="Food Allergy AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    # 查询命令
    query_cmd = subparsers.add_parser("query", help="查询问题")
    query_cmd.add_argument("question", help="要查询的问题")

    # 交互模式
    subparsers.add_parser("interactive", help="交互模式")

    # 缓存统计
    subparsers.add_parser("stats", help="显示缓存统计")

    # 清空缓存
    subparsers.add_parser("clear", help="清空所有缓存")

    args = parser.parse_args()

    if args.command == "query":
        print(f"查询: {args.question}\n")
        for result in query_with_graph(args.question):
            if result.get("node") == "end":
                print(f"答案:\n{result.get('generation')}\n")

    elif args.command == "interactive":
        while True:
            try:
                q = input(">> ").strip()
                if q.lower() in ["exit", "quit"]:
                    break
                for result in query_with_graph(q):
                    if result.get("node") == "end":
                        print(result.get('generation'))
            except KeyboardInterrupt:
                break

    elif args.command == "stats":
        stats = get_cache_stats()
        print(f"Hit Rate: {stats['hit_rate']:.1f}%")
        print(f"Total: {stats['total_queries']}")

    elif args.command == "clear":
        clear_all_caches()
        print("Cache cleared")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
