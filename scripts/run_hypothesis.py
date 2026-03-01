from pathlib import Path
import yaml

from nyc_taxi_mlops.hypothesis_modeling import build_parser, run


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "modeling.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parser = build_parser()
    args = parser.parse_args([])

    args.input_xlsx = str(project_root / cfg["input_xlsx"])
    zone_lookup = cfg.get("zone_lookup")
    args.zone_lookup = str(project_root / zone_lookup) if zone_lookup else None
    args.output_dir = str(project_root / cfg["output_dir"])
    args.seed = int(cfg.get("seed", 42))
    args.min_zone_train_rows = int(cfg.get("min_zone_train_rows", 120))
    args.max_zones = int(cfg.get("max_zones", 80))
    args.min_relative_gain = float(cfg.get("min_relative_gain", 0.05))
    args.fill_full_grid = bool(cfg.get("fill_full_grid", False))

    run(args)


if __name__ == "__main__":
    main()
