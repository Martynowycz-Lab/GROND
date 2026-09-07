"""Keep historical CLI imports usable after moving command implementations."""

import pytest

from grond import cli, macromolecule


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("read_macromolecule_table", "read_table"),
        ("write_macromolecule_table", "write_table"),
    ],
)
def test_macromolecule_table_alias_resolves_to_the_original_function(alias, canonical):
    assert getattr(cli, alias) is getattr(macromolecule, canonical)
