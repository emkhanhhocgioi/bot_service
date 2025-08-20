#!/usr/bin/env python3
"""Utility functions for location detection and API calls."""

import re
import unicodedata
import requests
from typing import List


def _strip_diacritics(s: str) -> str:
    """Return ASCII-only version of s for simple matching (remove diacritics)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def find_vietnam_provinces_in_text(text: str) -> List[str]:
    """Tách query và tìm các tỉnh/thành Việt Nam.

    Trả về danh sách (không trùng) tên tỉnh/thành ở dạng có dấu như trong danh sách.
    Phương pháp: chuẩn hoá (lowercase, bỏ dấu), sau đó tìm từng tên (cụm từ) trong chuỗi.
    """
    if not text:
        return []

    provinces = [
        "Hà Nội",
        "Hồ Chí Minh",
        "Hải Phòng",
        "Đà Nẵng",
        "Cần Thơ",
        "An Giang",
        "Bà Rịa - Vũng Tàu",
        "Bắc Giang",
        "Bắc Kạn",
        "Bạc Liêu",
        "Bắc Ninh",
        "Bến Tre",
        "Bình Định",
        "Bình Dương",
        "Bình Phước",
        "Bình Thuận",
        "Cà Mau",
        "Cao Bằng",
        "Đắk Lắk",
        "Đắk Nông",
        "Điện Biên",
        "Đồng Nai",
        "Đồng Tháp",
        "Gia Lai",
        "Hà Giang",
        "Hà Nam",
        "Hà Tĩnh",
        "Hải Dương",
        "Hậu Giang",
        "Hòa Bình",
        "Hưng Yên",
        "Khánh Hòa",
        "Kiên Giang",
        "Kon Tum",
        "Lai Châu",
        "Lâm Đồng",
        "Lạng Sơn",
        "Lào Cai",
        "Long An",
        "Nam Định",
        "Nghệ An",
        "Ninh Bình",
        "Ninh Thuận",
        "Phú Thọ",
        "Phú Yên",
        "Quảng Bình",
        "Quảng Nam",
        "Quảng Ngãi",
        "Quảng Ninh",
        "Quảng Trị",
        "Sóc Trăng",
        "Sơn La",
        "Tây Ninh",
        "Thái Bình",
        "Thái Nguyên",
        "Thanh Hóa",
        "Thừa Thiên - Huế",
        "Tiền Giang",
        "Trà Vinh",
        "Tuyên Quang",
        "Vĩnh Long",
        "Vĩnh Phúc",
        "Yên Bái",
    ]

    # Normalize input: lowercase, remove punctuation except spaces, strip diacritics
    text_norm = text.lower()
    text_norm = re.sub(r"[\-–—_/\\]+", " ", text_norm)  # treat various separators as spaces
    text_norm = re.sub(r"[^a-z0-9\s]+", " ", _strip_diacritics(text_norm))
    # Collapse spaces
    text_norm = re.sub(r"\s+", " ", text_norm).strip()

    found: List[str] = []
    for name in provinces:
        name_norm = name.lower()
        name_norm = re.sub(r"[\-–—_/\\]+", " ", name_norm)
        name_norm = re.sub(r"[^a-z0-9\s]+", " ", _strip_diacritics(name_norm))
        name_norm = re.sub(r"\s+", " ", name_norm).strip()

        # word-boundary search on normalized strings
        if name_norm and re.search(rf"\b{re.escape(name_norm)}\b", text_norm):
            found.append(name)

    # Remove duplicates while preserving order
    seen = set()
    result: List[str] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            result.append(f)

    return result


def call_local_api(payload: dict = None, method: str = "POST", url: str = "http://localhost:3001/api", timeout: int = 5) -> dict:
    """Gọi API cục bộ tại `url` (mặc định http://localhost:3001/api).

    - payload: dict để gửi (JSON cho POST hoặc params cho GET)
    - method: 'POST' hoặc 'GET'
    - timeout: giây chờ

    Trả về dict nếu response JSON, hoặc {'text': <raw text>} nếu response không phải JSON.
    Ném `requests.RequestException` nếu có lỗi kết nối/HTTP.
    """
    payload = payload or {}
    method = (method or "POST").upper()

    try:
        if method == "GET":
            resp = requests.get(url, params=payload, timeout=timeout)
        else:
            resp = requests.post(url, json=payload, timeout=timeout)

        resp.raise_for_status()
        # try parse JSON, fallback to text
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}

    except requests.RequestException:
        # re-raise so caller can handle/log
        raise


def search_trips_with_provinces(query: str, intent: str) -> dict:
  
    # Tách các tỉnh từ query
    provinces = find_vietnam_provinces_in_text(query)
    
    if len(provinces) < 2:
        return {
            "error": "Không tìm thấy đủ thông tin điểm đi và điểm đến",
            "found_provinces": provinces,
            "suggestion": "Vui lòng cung cấp cả điểm đi và điểm đến trong query"
        }
    
    # Sử dụng tỉnh đầu tiên làm from, tỉnh thứ hai làm to
    from_location = provinces[0]
    to_location = provinces[1]
    
    # Tạo params cho API call
    params = {
        "from": from_location,
        "to": to_location,
        **kwargs  # Thêm các tham số khác như departureTime, busType, etc.
    }
    
    # Loại bỏ các params None
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        # Gọi API trips/search
        response = call_local_api(
            payload=params,
            method="GET", 
            url="http://localhost:3001/api/trips/search",
            timeout=10
        )
        
        # Thêm thông tin về việc tách tỉnh để debug
        response["_debug"] = {
            "original_query": query,
            "detected_provinces": provinces,
            "from": from_location,
            "to": to_location,
            "search_params": params
        }
        
        return response
        
    except requests.RequestException as e:
        return {
            "error": f"Lỗi khi gọi API: {str(e)}",
            "detected_provinces": provinces,
            "from": from_location,
            "to": to_location
        }
