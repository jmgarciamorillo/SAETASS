import pytest


def pytest_addoption(parser):
    """Adds command-line options to pytest."""
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="show plots for visual inspection",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="update/generate golden dataset on disk",
    )


@pytest.fixture
def plot_results(request):
    """Fixture that returns True if --plot was passed, False otherwise."""
    return request.config.getoption("--plot")


@pytest.fixture
def update_golden(request):
    """Fixture that returns True if --update-golden was passed, False otherwise."""
    return request.config.getoption("--update-golden")
