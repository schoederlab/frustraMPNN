"""Basic package tests."""



def test_import():
    """Test that package can be imported."""
    import frustrampnn

    assert hasattr(frustrampnn, "__version__")


def test_version():
    """Test version string format."""
    import frustrampnn

    version = frustrampnn.__version__
    parts = version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_submodules_exist():
    """Test that submodules can be imported."""
    from frustrampnn import cli, data, inference, model, validation, visualization

    # Verify modules are not None
    assert model is not None
    assert inference is not None
    assert data is not None
    assert visualization is not None
    assert validation is not None
    assert cli is not None



