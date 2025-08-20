#!/usr/bin/env python3
"""Lightweight intent loader and predictor.

This pared-down module keeps only model loading and intent prediction.
"""
from pathlib import Path
import joblib
from typing import Tuple, List
import re
import unicodedata
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MongoSearchBot:
  

    def __init__(self, model_path: Path = None, resources_path: Path = None):
        self.base_dir = Path(__file__).parent
        self.model_path = model_path or (self.base_dir / "intent_transformer.h5")
        self.resources_path = resources_path or (self.base_dir / "intent_transformer_resources.joblib")

        # Model and preprocessing objects
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 50  # should match what the model expects

        self._load_model()

    def _load_model(self):
        """Load transformer model and preprocessing resources."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model không tìm thấy: {self.model_path}")
        if not self.resources_path.exists():
            raise FileNotFoundError(f"Resources không tìm thấy: {self.resources_path}")

        self.model = load_model(self.model_path)
        resources = joblib.load(self.resources_path)
        self.tokenizer = resources.get("tokenizer")
        self.label_encoder = resources.get("label_encoder")
        print(f"Đã tải model từ {self.model_path}")

    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent using the loaded transformer model.

        Returns (intent_label, confidence).
        """
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            raise ValueError("Model chưa được tải")

        sequences = self.tokenizer.texts_to_sequences([text])
        X = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

        probs = self.model.predict(X, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        intent = self.label_encoder.inverse_transform([idx])[0]

        return intent, confidence

def predict_intent(text: str) -> Tuple[str, float]:

    return MongoSearchBot().predict_intent(text)
   
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

    
    text_norm = text.lower()
    text_norm = re.sub(r"[\-–—_/\\]+", " ", text_norm)  
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



def call_local_api(payload: dict = None, method: str = "POST", url: str = "https://trip-service-z00x.onrender.com/api", timeout: int = 5) -> dict:

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
def find_prices_after_gia(text: str) -> List[str]:
    """Tìm các số/giá xuất hiện ngay sau từ 'giá' trong văn bản.
    
    Trả về danh sách các chuỗi giá như '1.200.000', '1.2 triệu', '500k', '1,000,000 VND', '1000₫', v.v.
    """
    if not text:
        return []

  
    pattern = re.compile(
        r'\b(?:gia|giá)\b\s*[:\-–—]?\s*'                       # từ "giá" và dấu phân cách
        r'([0-9]+(?:[.,\s][0-9]{3})*(?:[.,][0-9]+)?'           # số với dấu phân cách (1.000.000, 1,000, 1.5)
        r'(?:\s*(?:k|K|k\b|vnđ|vnd|đ|d|đồng|dong|usd|\$|₫))?)', # optional unit/currency
        re.IGNORECASE
    )

    matches = pattern.findall(text)
    # dọn token: trim không cần xử lý sâu, trả về nguyên chuỗi như thấy (người dùng có thể parse tiếp nếu cần)
    results = [m.strip() for m in matches if m and m.strip()]

    return results

def search_trips_with_provinces(query: str, intent: str) -> dict:

    # Tách các tỉnh từ query
    provinces = find_vietnam_provinces_in_text(query)
    
    if len(provinces) < 2:
        return {
            "error": "Không tìm thấy đủ thông tin điểm đi và điểm đến",
            "found_provinces": provinces,
            "suggestion": "Vui lòng cung cấp cả điểm đi và điểm đến trong query"
        }
    

    from_location = provinces[0]
    to_location = provinces[1]
    
    if (intent == "ask_route" or intent == "ask_destination"):
        params = {
            "from": from_location,
            "to": to_location,
        }
        
        # Loại bỏ các params None
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            # Gọi API trips/search
            response = call_local_api(
                payload=params,
                method="GET", 
                url="https://trip-service-z00x.onrender.com/api/trips/search",
                timeout=10
            )
          
            return response
            
        except requests.RequestException as e:
            return {
                "error": f"Lỗi khi gọi API: {str(e)}",
                "detected_provinces": provinces,
                "from": from_location,
                "to": to_location
            }
    if (intent == "ask_price" or intent == "ask_destination"):
            price = find_prices_after_gia(query)
            print(f"Found prices: {price}")
            print(f"From: {from_location}, To: {to_location}")
            params = {
                "maxPrice": price,
                "from": from_location,
                "to": to_location,
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
                print(f"API response: {response}")
                return response
                
            except requests.RequestException as e:
                return {
                    "error": f"Lỗi khi gọi API: {str(e)}",
                    "detected_provinces": provinces,
                    "from": from_location,
                    "to": to_location
                }