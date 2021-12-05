from amypet.cli import main


def test_cmd(nvml):
    main(["cuinfo", "-n"])
