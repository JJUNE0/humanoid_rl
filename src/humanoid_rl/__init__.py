def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("list", "setup", "launch", "train"):
        from humanoid_rl.cli import main as cli_main

        raise SystemExit(cli_main())
    from humanoid_rl.train import main as train_main

    train_main()
