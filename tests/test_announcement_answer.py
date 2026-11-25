from agents.announcement_answer import extract_announcement_answer


def hit(text, ref_id="R1", *, page=1, document_id="a" * 64):
    return {"text": text, "ref_id": ref_id, "page": page, "document_id": document_id}


def test_extracts_a_share_registration_date_from_the_correct_table_column():
    records = [hit(
        "董事会于2024年6月12日披露公告。股份类别 股权登记日 最后交易日 除权（息）日 现金红利发放日 "
        "Ａ股 2024/6/18 － 2024/6/19 2024/6/19"
    )]
    answers = extract_announcement_answer("2023 年度分红的股权登记日是哪天？", records)
    assert answers == [("股权登记日", "2024-06-18", "R1")]


def test_per_ten_shares_conversion_retains_tax_and_citation():
    records = [hit("每10股派发现金红利 3.0876 元（含税）。税后每股派发现金红利 0.277884 元。", "R2")]
    assert extract_announcement_answer("每股分多少现金红利？", records) == [
        ("每股现金分红", "0.30876 元（含税）", "R2")
    ]
    assert extract_announcement_answer("每 10 股派多少？", records) == [
        ("每 10 股现金分红", "3.0876 元（含税）", "R2")
    ]
    assert extract_announcement_answer("税后每股分红是多少？", records) == []


def test_conflicting_missing_and_cross_document_evidence_cannot_supply_one_answer():
    question = "股权登记日是哪一天？"
    assert extract_announcement_answer(question, [hit("股权登记日为2024年6月31日，红利发放日为2024年7月2日。")]) == []
    assert extract_announcement_answer(question, [hit("股权登记日为2024年6月18日。原方案的股权登记日为2024年6月17日。")]) == []
    assert extract_announcement_answer(question, [hit("股权登记日为2024年6月18日。"), hit("另一份公告", "R2", document_id="b" * 64)]) == []
    assert extract_announcement_answer("分红什么时候发放？", [hit("每股现金红利 3 元（含税）。")]) == []
    assert extract_announcement_answer("股权登记日是什么意思？", [hit("股权登记日为2024年6月18日。")]) == []
