from src.vision.config import load_config
from src.vision.validator import VisualParseValidator
from src.vision.schemas import DrawingParseResult, DrawingMetadata, DeviceEntity, DeviceParameter

CFG = load_config()
VALIDATOR = VisualParseValidator(CFG)


def _result(device_id: str, parameters=None) -> DrawingParseResult:
    return DrawingParseResult(
        document_id="d", page_number=1, drawing_metadata=DrawingMetadata(),
        devices=[DeviceEntity(device_id=device_id, parameters=parameters or [])],
    )


def test_ocr_vlm_ledger_all_agree_is_confirmed():
    out = VALIDATOR.validate(_result("A12风机"), ocr_device_id_hint="A12风机")
    assert out.validation.status == "confirmed"
    assert out.validation.warnings == []


def test_ocr_disagrees_but_vlm_matches_ledger_is_suspected_and_keeps_ocr():
    out = VALIDATOR.validate(_result("A12风机"), ocr_device_id_hint="A1Z风机")
    assert out.validation.status == "suspected"
    assert any("A1Z" in w for w in out.validation.warnings)


def test_ocr_and_vlm_disagree_and_no_ledger_match_is_conflict():
    out = VALIDATOR.validate(_result("Z99"), ocr_device_id_hint="A1Z")
    assert out.validation.status == "conflict"


def test_no_ocr_hint_but_ledger_match_is_confirmed():
    out = VALIDATOR.validate(_result("A12风机"), ocr_device_id_hint=None)
    assert out.validation.status == "confirmed"


def test_no_ocr_hint_and_no_ledger_match_is_suspected_not_conflict():
    # Single unverified source with no contradiction shouldn't be treated the
    # same as an active disagreement.
    out = VALIDATOR.validate(_result("Z99"), ocr_device_id_hint=None)
    assert out.validation.status == "suspected"


def test_unit_outside_whitelist_is_conflict_not_silently_fixed():
    param = DeviceParameter(name="power", raw_name="功率", raw_text="45KVV", value=45, unit="KVV")
    out = VALIDATOR.validate(_result("A12风机", [param]))
    assert out.validation.status == "conflict"
    # the value must NOT be silently rewritten to a "corrected" unit
    assert out.devices[0].parameters[0].unit == "KVV"


def test_ocr_and_vlm_parameter_value_conflict_is_recorded_not_overwritten():
    param = DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW")
    out = VALIDATOR.validate(
        _result("A12风机", [param]),
        ocr_parameter_hints={"A12风机": {"power": "450kW"}},
    )
    assert out.validation.status == "conflict"
    assert out.devices[0].parameters[0].raw_text == "45kW"  # VLM's own reading untouched


def test_relation_corroborated_by_ledger_boosts_confidence():
    from src.vision.schemas import VisualRelation
    result = DrawingParseResult(
        document_id="d", page_number=1,
        relations=[VisualRelation(source_id="A12风机", relation="connected_to", target_id="B07屏蔽门", confidence=0.6)],
    )
    out = VALIDATOR.validate(result)
    assert out.relations[0].source == "corroborated"
    assert out.relations[0].confidence > 0.6


def test_relation_not_in_ledger_stays_vlm_only():
    from src.vision.schemas import VisualRelation
    result = DrawingParseResult(
        document_id="d", page_number=1,
        relations=[VisualRelation(source_id="X1", relation="connected_to", target_id="X2", confidence=0.6)],
    )
    out = VALIDATOR.validate(result)
    assert out.relations[0].source == "vlm_only"
    assert out.relations[0].confidence == 0.6


# --- 4-way (OCR original / OCR processed / VLM / ledger) cross-check ---

def test_both_ocr_readings_agree_with_vlm_and_ledger_is_confirmed():
    out = VALIDATOR.validate(
        _result("A12风机"),
        ocr_device_id_hint="A12风机",
        ocr_processed_device_id_hint="A12风机",
    )
    assert out.validation.status == "confirmed"


def test_original_ocr_wrong_but_processed_ocr_matches_vlm_is_suspected_not_conflict():
    """Preprocessing fixed the OCR reading — worth surfacing as a suspected
    case with the original OCR's mistake preserved, not silently resolved."""
    out = VALIDATOR.validate(
        _result("A12风机"),
        ocr_device_id_hint="A1Z风机",
        ocr_processed_device_id_hint="A12风机",
    )
    assert out.validation.status == "suspected"
    assert any("A1Z" in w for w in out.validation.warnings)


def test_original_and_processed_ocr_disagree_with_each_other_note_is_included():
    out = VALIDATOR.validate(
        _result("A12风机"),
        ocr_device_id_hint="A1Z风机",
        ocr_processed_device_id_hint="A17风机",
    )
    assert out.validation.status == "suspected"
    assert any("原图 OCR 与预处理图 OCR 本身也不一致" in w for w in out.validation.warnings)


# --- table completeness (`incomplete` status) ---

def test_missing_expected_parameter_is_incomplete_not_silently_passing():
    param = DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW")
    out = VALIDATOR.validate(
        _result("A12风机", [param]),
        expected_parameter_names=["power", "air_volume"],
    )
    assert out.validation.status == "incomplete"
    assert any("air_volume" in w for w in out.validation.warnings)


def test_all_expected_parameters_present_is_not_incomplete():
    params = [
        DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW"),
        DeviceParameter(name="air_volume", raw_name="风量", raw_text="28000m3/h", value=28000, unit="m3/h"),
    ]
    out = VALIDATOR.validate(
        _result("A12风机", params),
        expected_parameter_names=["power", "air_volume"],
    )
    assert out.validation.status == "confirmed"


def test_no_expected_parameter_names_never_triggers_incomplete():
    """No generic per-device-type schema exists in this repo — completeness
    is only checked when the caller explicitly supplies what 'complete'
    means, never guessed."""
    param = DeviceParameter(name="power", raw_name="功率", raw_text="45kW", value=45, unit="kW")
    out = VALIDATOR.validate(_result("A12风机", [param]))
    assert out.validation.status == "confirmed"


# --- duplicate device detection ---

def test_duplicate_device_id_is_flagged():
    result = DrawingParseResult(
        document_id="d", page_number=1,
        devices=[DeviceEntity(device_id="A12风机"), DeviceEntity(device_id="A12风机")],
    )
    out = VALIDATOR.validate(result)
    assert out.validation.status == "suspected"
    assert any("出现了 2 次" in w for w in out.validation.warnings)
