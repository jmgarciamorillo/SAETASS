import numpy as np
import pytest

from saetass.state import State


class TestState:
    def test_initialization(self):
        f = np.ones((10, 20))
        state = State(f)
        assert state.n_p == 10
        assert state.n_r == 20
        assert state.grid_shape == (10, 20)
        assert state.t == 0.0

        # 1D initialization
        f_1d = np.ones((20,))
        state_1d = State(f_1d)
        assert state_1d.grid_shape == (1, 20)

    def test_clone(self):
        f = np.ones((10, 20))
        state = State(f, t=1.0)
        state.record_substep("first")

        # Clone without history
        clone1 = state.clone(copy_history=False)
        assert clone1.t == 1.0
        assert len(clone1.history) == 0

        # Clone with history
        clone2 = state.clone(copy_history=True)
        assert len(clone2.history) == 1

    def test_get_and_update_f(self):
        f_1d = np.ones((20,))
        state = State(f_1d)
        assert state.get_f().shape == (20,)  # Returns natural dimensionality

        # update with 1D
        new_f = np.zeros((20,))
        state.update_f(new_f)
        assert np.all(state.get_f() == 0)

        # invalid shape update
        with pytest.raises(ValueError):
            state.update_f(np.zeros((10,)))

        # 2D update
        state2 = State(np.ones((10, 20)))
        state2.update_f(np.zeros((10, 20)))
        assert np.all(state2.get_f() == 0)

    def test_time_and_stage(self):
        state = State(np.ones((10, 20)))
        state.set_time(5.0)
        assert state.t == 5.0
        assert state.dt == 5.0

        state.step_stage("split1")
        assert state.stage == 1
        assert state.stage_name == "split1"

    def test_history_management(self):
        state = State(np.ones((10, 20)), t=0.0)
        state.record_substep("init")

        state.set_time(1.0)
        state.update_f(np.zeros((10, 20)))
        state.record_substep("step1")

        assert len(state.history) == 2

        sub = state.get_substep(0)
        assert sub["t"] == 0.0
        assert np.all(sub["f"] == 1)

        with pytest.raises(IndexError):
            state.get_substep(5)

        # Restore by name
        state.restore_substep("init")
        assert state.t == 0.0
        assert np.all(state.f == 1)

        # Restore by index
        state.restore_substep(1)
        assert state.t == 1.0
        assert np.all(state.f == 0)

        with pytest.raises(ValueError):
            state.restore_substep("invalid")

        with pytest.raises(IndexError):
            state.restore_substep(5)

        state.clear_history()
        assert len(state.history) == 0

    def test_str_repr(self):
        state = State(np.ones((10, 20)))
        assert "State" in repr(state)


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    print("Running tests...")
    pytest.main([__file__])
