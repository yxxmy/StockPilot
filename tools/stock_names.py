"""从巨潮资讯公开名录读取 A 股简称，保留证券代码的前导零。"""

import re

import requests


def load_a_share_names():
    try:
        response = requests.get("https://www.cninfo.com.cn/new/data/szse_stock.json", timeout=(5, 10))
        response.raise_for_status()
        records = response.json()["stockList"]
        if not isinstance(records, list):
            raise ValueError("Invalid directory")
        pairs = set()
        for row in records:
            if not isinstance(row, dict) or row.get("category") != "A股":
                continue
            name, code = row.get("zwjc"), row.get("code")
            if isinstance(name, str) and name.strip() and isinstance(code, str) and re.fullmatch(r"[0-9]{6}", code):
                pairs.add((name.strip(), code))
        if not pairs:
            raise ValueError("Empty directory")
    except (requests.RequestException, ValueError, KeyError, TypeError):
        raise ValueError("暂时无法读取巨潮资讯的股票简称名录，请稍后重试，或直接输入六位股票代码。") from None
    return sorted(pairs)
