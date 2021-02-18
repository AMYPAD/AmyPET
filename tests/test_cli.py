from amypad.cli import main


def test_main():
    main(["hasext", "foo.bar", "bar"])
