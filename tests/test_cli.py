from amypad.cli import main


def test_run():
    main(["hasext", "foo.bar", "bar"])


def test_cmd(nvml):
    main(["cuinfo", "-n"])
