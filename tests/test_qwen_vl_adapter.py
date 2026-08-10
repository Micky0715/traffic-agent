from src.vision.qwen_vl_adapter import _coerce_parameter_value


def test_coerce_parameter_value_leaves_valid_numbers_alone():
    raw = {"name": "power", "raw_name": "功率", "raw_text": "45kW", "value": 45, "unit": "kW"}
    assert _coerce_parameter_value(raw)["value"] == 45


def test_coerce_parameter_value_nulls_out_non_numeric_string():
    """Regression test for a real crash found this session: the VLM put a
    device ID string in a numeric field, which crashed DeviceEntity
    validation with no handling anywhere in the call chain."""
    raw = {"name": "motor_id", "raw_name": "电机编号", "raw_text": "M-13", "value": "M-13", "unit": None}
    coerced = _coerce_parameter_value(raw)
    assert coerced["value"] is None
    assert coerced["raw_text"] == "M-13"  # the actual reading is preserved, only the bad numeric coercion is dropped


def test_coerce_parameter_value_handles_missing_value_field():
    raw = {"name": "power", "raw_name": "功率", "raw_text": "45kW", "unit": "kW"}
    assert _coerce_parameter_value(raw).get("value") is None


def test_coerce_parameter_value_accepts_numeric_string():
    raw = {"name": "power", "raw_name": "功率", "raw_text": "45", "value": "45", "unit": None}
    assert _coerce_parameter_value(raw)["value"] == "45"  # left as-is; Pydantic itself coerces numeric strings
