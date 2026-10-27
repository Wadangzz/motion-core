def test_import_needs_nothing_optional():
    """Importing the package must work on a bare numpy + scipy install."""
    import motion_core
    from motion_core.loads import LoadModel  # noqa: F401

    assert motion_core.__version__
