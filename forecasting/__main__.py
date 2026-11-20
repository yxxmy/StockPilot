"""独立时序CLI：数据快照、离线检查和8C滚动实验。"""

import argparse
import json

from forecasting.data import ForecastDataError, fetch_snapshot, load_snapshot, save_snapshot
from forecasting.targets import prepare_targets


def main():
    parser = argparse.ArgumentParser(description="StockPilot 时序数据及独立滚动实验；不调用DeepSeek")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="联网获取同源前复权/不复权对照并保存新快照")
    fetch.add_argument("symbol")
    fetch.add_argument("--start-date", required=True)
    fetch.add_argument("--end-date", required=True)
    fetch.add_argument("--source", choices=["tencent", "eastmoney"], default="tencent")
    fetch.add_argument("--output-dir", default="data/forecasting")
    inspect = sub.add_parser("inspect", help="离线验证快照哈希和日期目标，不请求网络")
    inspect.add_argument("path")
    evaluate = sub.add_parser("evaluate",help="离线运行验证/测试滚动实验，保存独立JSON及图表")
    evaluate.add_argument("path")
    evaluate.add_argument("--output-dir",default="data/forecasting/experiments")
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            from forecasting.evaluation import evaluate_snapshot,save_experiment
            experiment=evaluate_snapshot(load_snapshot(args.path))
            path=save_experiment(experiment,args.output_dir)
            print(json.dumps({"status":experiment["status"],"run_id":experiment["run_id"],"path":str(path),"summary":experiment["summary"]},ensure_ascii=False,indent=2))
            return 1 if experiment["status"]=="failed" else 0
        snapshot = fetch_snapshot(args.symbol, args.start_date, args.end_date, source=args.source) if args.command == "fetch" else load_snapshot(args.path)
        result = prepare_targets(snapshot)
        path = str(save_snapshot(snapshot, args.output_dir)) if args.command == "fetch" else args.path
        print(json.dumps({"path": path, "snapshot_id": snapshot["snapshot_id"], "metadata": snapshot["body"]["metadata"],
            "returns": len(result["returns"]), "evaluation_targets": len(result["targets"]),
            "initial_train_range": result["initial_train_range"], "target_limitation": result["limitation"]}, ensure_ascii=False, indent=2))
        return 0
    except ForecastDataError as exc:
        print(f"数据准备失败：{exc}；未训练模型。")
        return 1
    except (ValueError, OSError):
        print("实验执行或保存失败，未发布完整实验。" if args.command=="evaluate" else "数据准备失败：日期参数或文件访问错误；未训练模型。")
        return 1
    except (ImportError, RuntimeError):
        print("实验依赖或运行环境不可用，请检查requirements-forecast.txt；未发布完整实验。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
