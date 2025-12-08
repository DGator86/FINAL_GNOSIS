from datetime import datetime

from gnosis.utils.option_utils import OptionUtils


def test_option_symbol_builder_rounding_and_padding():
    symbol = OptionUtils.generate_occ_symbol("aapl", datetime(2024, 1, 19), "call", 150)
    assert symbol == "AAPL240119C00150000"

    # Ensure three-decimal rounding is deterministic
    symbol2 = OptionUtils.generate_occ_symbol("msft", datetime(2024, 1, 26), "put", 176.005)
    assert symbol2 == "MSFT240126P00176005"


def test_parse_roundtrip_matches_builder():
    expiration = datetime(2024, 2, 16)
    built = OptionUtils.generate_occ_symbol("SPY", expiration, "put", 420.5)
    parsed = OptionUtils.parse_occ_symbol(built)

    assert parsed["symbol"] == "SPY"
    assert parsed["expiration"] == expiration
    assert parsed["option_type"] == "put"
    assert parsed["strike"] == 420.5
