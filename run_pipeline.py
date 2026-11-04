"""CLI: python run_pipeline.py [collect|mlflow|report|all]"""
import sys

from src import analysis, config, registry


def main(stage: str) -> None:
    print(f"Model={config.MODEL} dry_run={config.DRY_RUN} outputs={config.OUTPUT_DIR}")
    if stage in ("collect", "all"):
        analysis.collect_calls()
    if stage in ("mlflow", "all"):
        df = analysis.load_calls()
        table, best = registry.run_task3(df)
        print(table.to_string(index=False))
        print(f"Promoted {best} to '{config.PRODUCTION_ALIAS}'")
    if stage in ("report", "all"):
        from src import report
        print("Report:", report.build())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
