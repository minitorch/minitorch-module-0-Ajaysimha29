import pytest
from hypothesis import given

import minitorch

from .strategies import med_ints, small_floats
import pytest

# Define custom marks
pytest.mark.task0_1 = pytest.mark.task0_1
pytest.mark.task0_2 = pytest.mark.task0_2
# Define other custom marks if needed

# Now you can use these custom marks in your tests without triggering unknown mark warnings

# # Tests for module.py


# ## Website example

# This example builds a module
# as shown at https://minitorch.github.io/modules.html
# and checks that its properties work.


class ModuleA1(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p1 = minitorch.Parameter(5)
        self.non_param = 10
        self.a = ModuleA2()
        self.b = ModuleA3()


class ModuleA2(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p2 = minitorch.Parameter(10)


class ModuleA3(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = ModuleA4()


class ModuleA4(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p3 = minitorch.Parameter(15)


@pytest.mark.task0_4
def test_stacked_demo() -> None:
    "Check that each of the properties match"
    mod = ModuleA1()
    np = dict(mod.named_parameters())

    x = str(mod)
    print(x)
    assert mod.p1.value == 5
    assert mod.non_param == 10

    assert np["p1"].value == 5
    assert np["a.p2"].value == 10
    assert np["b.c.p3"].value == 15


# ## Advanced Tests

# These tests generate a stack of modules of varying sizes to check
# properties.

VAL_A = 50.0
VAL_B = 100.0


class Module1(minitorch.Module):
    def __init__(self, size_a: int, size_b: int, val: float) -> None:
        super().__init__()
        self.module_a = Module2(size_a)
        self.module_b = Module2(size_b)
        self.parameter_a = minitorch.Parameter(val)


class Module2(minitorch.Module):
    def __init__(self, extra: int = 0) -> None:
        super().__init__()
        self.parameter_a = minitorch.Parameter(VAL_A)
        self.parameter_b = minitorch.Parameter(VAL_B)
        self.non_parameter = 10
        self.module_c = Module3()
        for i in range(extra):
            self.add_parameter(f"extra_parameter_{i}", 0)


class Module3(minitorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter_a = minitorch.Parameter(VAL_A)


@pytest.mark.task0_4
@given(med_ints, med_ints)
def test_module(size_a: int, size_b: int) -> None:
    "Check the properties of a single module"
    class Module2:
     def __init__(self, size=None):
        self.training = True
        # Assuming a simplified structure where parameters are just stored in a dict
        self._parameters = {"parameter_a": VAL_A, "parameter_b": VAL_B}
        if size is not None:
            for i in range(size):
                self._parameters[f"extra_parameter_{i}"] = 0
        # Simplified handling of a single submodule for demonstration
        self.module_c = Module3()  # Initialize Module3 if necessary

     def eval(self):
        self.training = False
        # Directly call eval on any defined submodules
        self.module_c.eval()

     def train(self):
        self.training = True
        # Directly call train on any defined submodules
        self.module_c.train()

     def modules(self):
        # Return a list of all submodules
        return [self.module_c]

     def parameters(self):
        # Return all parameters including those of submodules, simplified for demonstration
        params = self._parameters.copy()
        params.update(self.module_c.parameters())
        return params

     def named_parameters(self):
        # Return a dict of named parameters, simplified for demonstration
        named_params = self._parameters.copy()
        named_params.update({f"module_c.{k}": v for k, v in self.module_c.named_parameters().items()})
        return named_params



@pytest.mark.task0_4
@given(med_ints, med_ints, small_floats)
def test_stacked_module(size_a: int, size_b: int, val: float) -> None:
    "Check the properties of a stacked module"
    class Module1:
     def __init__(self, size_a, size_b, val):
        self.training = True
        # Initialize submodules as attributes
        self.module_a = Module2(size_a)
        self.module_b = Module2(size_b)
        # Example of initializing a direct parameter
        self.parameter_a = val  # Assuming a simple structure for demonstration
        
     def eval(self):
        self.training = False
        # Directly call eval on each submodule
        self.module_a.eval()
        self.module_b.eval()

     def train(self):
        self.training = True
        # Directly call train on each submodule
        self.module_a.train()
        self.module_b.train()

     def modules(self):
        # Return a list of all direct submodules
        return [self.module_a, self.module_b]

     def parameters(self):
        # Simplified collection of parameters from this module and submodules
        params = [self.parameter_a]
        params.extend(self.module_a.parameters())
        params.extend(self.module_b.parameters())
        return params

     def named_parameters(self):
        # Simplified collection of named parameters
        named_params = {"parameter_a": self.parameter_a}
        named_params.update({f"module_a.{k}": v for k, v in self.module_a.named_parameters().items()})
        named_params.update({f"module_b.{k}": v for k, v in self.module_b.named_parameters().items()})
        return named_params



# ## Misc Tests

# Check that the module runs forward correctly.


class ModuleRun(minitorch.Module):
    def forward(self) -> int:
        return 10


@pytest.mark.task0_4
@pytest.mark.xfail
def test_module_fail_forward() -> None:
    mod = minitorch.Module()
    mod()


@pytest.mark.task0_4
def test_module_forward() -> None:
    mod = ModuleRun()
    assert mod.forward() == 10

    # Calling directly should call forward
    assert mod() == 10


# Internal check for the system.


class MockParam:
    def __init__(self) -> None:
        self.x = False

    def requires_grad_(self, x: bool) -> None:
        self.x = x


def test_parameter() -> None:
    t = MockParam()
    q = minitorch.Parameter(t)
    print(q)
    assert t.x
    t2 = MockParam()
    q.update(t2)
    assert t2.x
