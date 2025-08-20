#!/usr/bin/env python3
"""
Demo script showing complete usage of transformer intent classification 
with MongoDB search for bus route information.
"""
from pathlib import Path
import json
from mongo_search_bot import MongoSearchBot


def demo_transformer_training():
    """Demonstrate transformer model training."""
    print("=== BƯỚC 1: HUẤN LUYỆN TRANSFORMER MODEL ===")
    print("Chạy lệnh sau để huấn luyện:")
    print("python train_transformer.py --epochs 30 --batch-size 16 --embed-dim 128")
    print()


def demo_mongodb_setup():
    """Show MongoDB collection structure and sample data."""
    print("=== BƯỚC 2: THIẾT LẬP MONGODB ===")
    print("Cấu trúc collection 'routes' trong database 'bus_routes':")
    
    sample_document = {
        "_id": "ObjectId('668f57fb28a50e0749c36')",
        "routeCode": "QB-HN-02",
        "partnerId": "ObjectId('687fe3869ae80b3c1f6f5d3')",
        "from": "Đà Nẵng",
        "to": "Quảng Bình",
        "departureTime": "23:00",
        "duration": "5 giờ",
        "price": 200000,
        "totalSeats": 22,
        "availableSeats": 19,
        "busType": "Xe limousine 22 chỗ",
        "licensePlate": "29A-1234",
        "rating": 0,
        "tags": [],
        "description": "xe có wifi free",
        "isActive": True,
        "createdAt": "2025-08-03T12:37:15.396+00:00",
        "bookedSeats": [3],
        "images": [1]
    }
    
    print(json.dumps(sample_document, indent=2, ensure_ascii=False))
    print()
    print("Để import dữ liệu mẫu vào MongoDB:")
    print("mongoimport --db bus_routes --collection routes --file sample_routes.json --jsonArray")
    print()


def demo_search_queries():
    """Demonstrate various search queries."""
    print("=== BƯỚC 3: THỬ NGHIỆM TÌM KIẾM ===")
    
    try:
        bot = MongoSearchBot()
        
        test_queries = [
            "Giá vé từ Hà Nội đi Đà Nẵng bao nhiêu",
            "Có tuyến nào từ TP Hồ Chí Minh đi Cần Thơ không", 
            "Giờ khởi hành từ Đà Nẵng đến Quảng Bình là mấy giờ",
            "Từ Đà Nẵng có thể đi những đâu",
            "Mô tả về tuyến Hà Nội - Hải Phòng",
            "Giá vé trung bình từ Sài Gòn đi Cần Thơ",
            "Giờ khởi hành mặc định từ Hà Nội"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            result = bot.search_routes(query)
            
            print(f"Intent: {result['predicted_intent']} (tin cậy: {result['confidence']:.3f})")
            print(f"Message: {result['message']}")
            
            if result['results']:
                print("Top results:")
                for j, res in enumerate(result['results'][:2], 1):
                    route = res['route_info']
                    highlighted = res['highlighted_fields']
                    print(f"  {j}. Route: {route.get('from', 'N/A')} -> {route.get('to', 'N/A')}")
                    for field in highlighted:
                        if field in route:
                            print(f"     {field}: {route[field]}")
            print()
            
    except Exception as e:
        print(f"Lỗi demo: {e}")
        print("Đảm bảo đã:")
        print("1. Cài đặt MongoDB và chạy service")
        print("2. Huấn luyện transformer model")
        print("3. Cài đặt các dependencies: pip install -r requirements.txt")


def demo_usage_examples():
    """Show code usage examples."""
    print("=== BƯỚC 4: SỬ DỤNG TRONG CODE ===")
    
    usage_code = '''
# Sử dụng cơ bản
from mongo_search_bot import MongoSearchBot

bot = MongoSearchBot(
    mongo_uri="mongodb://localhost:27017",
    db_name="bus_routes", 
    collection_name="routes"
)

# Tìm kiếm với intent classification
result = bot.search_routes("Giá vé từ Hà Nội đi Đà Nẵng bao nhiêu")

print(f"Intent: {result['predicted_intent']}")
print(f"Confidence: {result['confidence']}")
print(f"Results found: {len(result['results'])}")

for route_result in result['results']:
    route = route_result['route_info']
    print(f"Route: {route['from']} -> {route['to']}")
    print(f"Price: {route['price']} VND")
    print(f"Departure: {route['departureTime']}")

bot.close()

# Sử dụng chỉ intent classification
from intent_bot import predict_intent

intent, confidence = predict_intent("Giá vé đi Đà Lạt bao nhiêu", use_transformer=True)
print(f"Intent: {intent}, Confidence: {confidence}")
'''
    
    print(usage_code)


def main():
    print("🚌 DEMO: TRANSFORMER INTENT CLASSIFICATION + MONGODB SEARCH")
    print("=" * 60)
    
    demo_transformer_training()
    demo_mongodb_setup()
    demo_search_queries()
    demo_usage_examples()
    
    print("=== KẾT THÚC DEMO ===")
    print("Để bắt đầu sử dụng:")
    print("1. Cài dependencies: pip install -r requirements.txt")
    print("2. Chạy MongoDB service")
    print("3. Huấn luyện model: python train_transformer.py")
    print("4. Import dữ liệu bus routes vào MongoDB")
    print("5. Sử dụng MongoSearchBot để tìm kiếm")


if __name__ == "__main__":
    main()
