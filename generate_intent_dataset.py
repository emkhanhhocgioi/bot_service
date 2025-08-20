import random
import json
from pathlib import Path

origins = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre", "Bình Định",
    "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau", "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông",
    "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng",
    "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên",
    "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
    "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long",
    "Vĩnh Phúc", "Yên Bái", "Sài Gòn", "Đà Lạt", "Nha Trang", "Vinh", "Huế"
]
destinations = [d for d in origins if d != ""]

# Only templates related to price, route and time
price_templates = [
    "Giá vé từ {o} đi {d} bao nhiêu",
    "Bao nhiêu tiền một vé {o} - {d}",
    "Xe từ {o} đến {d} giá thế nào",
    "Đi {o} {d} hết bao nhiêu tiền",
    "Vé {o} đi {d} bao nhiêu tiền"
]

route_templates = [
    "Có tuyến nào từ {o} đi {d} không",
    "Xe từ {o} đến {d} có không",
    "Tuyến {o} {d} có xe chạy không",
    "Có chuyến nào đi từ {o} tới {d} không",
    "Từ {o} đi {d} có xe không"
]

time_templates = [
    "Giờ khởi hành từ {o} đến {d} là mấy giờ",
    "Chuyến đi {o} - {d} khởi hành lúc mấy",
    "Thời gian xuất phát {o} tới {d} là khi nào",
    "Tuyến {o}-{d} có giờ chạy như thế nào",
    "Mất bao lâu từ {o} đến {d}"
]

default_time_templates = [
    "Giờ khởi hành mặc định từ {o} là mấy giờ",
    "Thời gian xuất phát thông thường ở {o} là khi nào",
    "Xe thường chạy từ {o} lúc mấy giờ",
    "Khởi hành tiêu chuẩn từ {o} là lúc nào"
]

description_templates = [
    "Bạn có thể mô tả về tuyến {o} - {d} không",
    "Tuyến {o} đến {d} có gì đặc biệt",
    "Giới thiệu về tuyến xe {o} {d}",
    "Tuyến {o} {d} có điểm gì nổi bật"
]

destination_templates = [
    "Từ {o} có thể đi những đâu",
    "Các điểm đến từ {o} là gì",
    "Xe từ {o} đi được những tỉnh nào",
    "Từ {o} có tuyến đi tỉnh nào"
]

average_price_templates = [
    "Giá vé trung bình từ {o} đến {d} là bao nhiêu",
    "Trung bình đi {o} {d} hết bao nhiêu tiền",
    "Vé {o} đi {d} thường giá bao nhiêu",
    "Giá thông thường tuyến {o} {d} là bao nhiêu"
]

dataset = []

# configurable counts
count_each = 120

# Generate price intents
for _ in range(count_each):
    o, d = random.sample(origins, 2)
    dataset.append({"text": random.choice(price_templates).format(o=o, d=d), "intent": "ask_price"})

# Generate route intents
for _ in range(count_each):
    o, d = random.sample(origins, 2)
    dataset.append({"text": random.choice(route_templates).format(o=o, d=d), "intent": "ask_route"})

# Generate time intents
for _ in range(count_each):
    o, d = random.sample(origins, 2)
    dataset.append({"text": random.choice(time_templates).format(o=o, d=d), "intent": "ask_time"})

# Generate default time intents
for _ in range(count_each):
    o = random.choice(origins)
    dataset.append({"text": random.choice(default_time_templates).format(o=o), "intent": "ask_default_time"})

# Generate description intents
for _ in range(count_each):
    o, d = random.sample(origins, 2)
    dataset.append({"text": random.choice(description_templates).format(o=o, d=d), "intent": "ask_description"})

# Generate destination intents
for _ in range(count_each):
    o = random.choice(origins)
    dataset.append({"text": random.choice(destination_templates).format(o=o), "intent": "ask_destination"})

# Generate average price intents
for _ in range(count_each):
    o, d = random.sample(origins, 2)
    dataset.append({"text": random.choice(average_price_templates).format(o=o, d=d), "intent": "ask_average_price"})

# Save to JSON (overwrites existing intents file)
output_path = Path(__file__).parent / "intents_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Tổng số câu: {len(dataset)}")
print(f"Lưu dataset vào: {output_path}")