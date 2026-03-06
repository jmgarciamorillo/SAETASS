import pytest


def pytest_addoption(parser):
    """Adds the --plot command-line option to pytest."""
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="show plots for visual inspection",
    )


@pytest.fixture
def plot_results(request):
    """Fixture that returns True if --plot was passed, False otherwise."""
    return request.config.getoption("--plot")
