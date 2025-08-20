#!/usr/bin/env python3
"""Demo script để test hàm tìm kiếm chuyến xe với tách tỉnh tự động."""

import sys
import json
from pathlib import Path

# Thêm thư mục hiện tại vào Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import các hàm cần thiết từ utils (tách ra để tránh lỗi tensorflow)
try:
    from utils import find_vietnam_provinces_in_text, search_trips_with_provinces
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    print("Tip: Đảm bảo requests đã được cài đặt: pip install requests")
    sys.exit(1)

def test_province_extraction():
    """Test hàm tách tỉnh."""
    print("\n=== TEST TÁCH TỈNH ===")
    
    test_queries = [
        "Tôi muốn đi từ Quảng Bình đến Đà Nẵng",
        "Có chuyến xe từ hcm đến ha noi không?",
        "Đi Sài Gòn về Huế",
        "Từ Cần Thơ tới Hải Phòng",
        "Tìm vé xe Quảng Ninh - Lâm Đồng"
    ]
    
    for query in test_queries:
        provinces = find_vietnam_provinces_in_text(query)
        print(f"Query: '{query}'")
        print(f"Tỉnh tìm được: {provinces}")
        print()

def test_trip_search():
    """Test hàm tìm kiếm chuyến xe."""
    print("\n=== TEST TÌM KIẾM CHUYẾN XE ===")
    
    test_cases = [
        {
            "query": "Tôi muốn đi từ Quảng Bình đến Đà Nẵng ngày mai",
           
        },
        {
            "query": "Có chuyến xe từ HCM đến Hà Nội không?",
          
        },
        {
            "query": "Tìm xe từ Cần Thơ đến Hải Phòng giá rẻ",
           
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"--- Test Case {i} ---")
        print(f"Query: '{case['query']}'")
        print(f"Params: {case['params']}")
        
        try:
            result = search_trips_with_provinces(case["query"])
            print("Kết quả:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        print()

def main():
    print("🚌 DEMO TÌM KIẾM CHUYẾN XE VỚI TÁCH TỈNH TỰ ĐỘNG")
    print("=" * 50)
    
    # Test tách tỉnh
    test_province_extraction()
    
    # Test tìm kiếm chuyến xe
    print("⚠️  Lưu ý: API localhost:3000 cần phải chạy để test thành công")
    test_trip_search()
    
    print("\n✨ Demo hoàn thành!")
    print("\nCách sử dụng trong code:")
    print("""
from utils import search_trips_with_provinces

# Tìm chuyến xe đơn giản
result = search_trips_with_provinces("Đi từ Hà Nội đến Đà Nẵng")

# Với thêm tham số
result = search_trips_with_provinces(
    "Tìm xe từ HCM đến Hà Nội", 
    departureTime="2025-08-25",
    busType="giường nằm",
    maxPrice=800000
)
""")

if __name__ == "__main__":
    main()
