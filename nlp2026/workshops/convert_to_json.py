# 1. กำหนดรูปแบบคำศัพท์
import re

VIOLATION_KEYWORD = ["ละเมิด","ปลอมแปลง","เลียนแบบ","ทำซ้ำ","ดัดแปลง"]
PATENT_KEYWORD = ["สิทธิบัตร","การประดิษฐ์","ผังภูมิวงจร"]
COPYRIGHT_KEYWORD = ["ลิขสิทธิ์","วรรณกรรม","ศิลปกรรม","ดนตรีกรรม"]

def detect_category(text):
    # ใช้ Regex
    is_patent = any(re.search(k,text)  for k in PATENT_KEYWORD)
    is_violation = any(re.search(k,text)  for k in VIOLATION_KEYWORD)
    is_copyright = any(re.search(k,text)  for k in COPYRIGHT_KEYWORD)
    if is_violation:
        if is_patent: return 1
        if is_copyright: return 2
    return 0 
# ทดสอบรัน
sample = "พบการทำซ้ำวรรณกรรมโดยไม่ได้รับอนุญาต"
# print(f"Text : {sample}")
# print(f"Predicted Class : {detect_category(sample)}")

# 2. Context-aware Confidence
def cal_confidence(text, predicted_class):
    base_conf = 0.70
    signals = []
    if "มาตรา" in text or "พ.ร.บ." in text:
        base_conf += 0.15
        signals.append("statotury_ref")
    if "คำพิพากษา" in text :
        base_conf += 0.10
        signals.append("precedent_ref")
    return min(base_conf,0.99), signals
# ทดสอบรัน
text_with_context = "ละเมิดลิขสิทธิ์ตามมาตรา 27 แห่ง พ.ร.บ. ลิขสิทธิ์ "
conf, sig = cal_confidence(text_with_context, 2)
print(f"Confidence: {conf: 2f}")
print(f"Signals: {sig}")

#3. ลำดับศักดิ์ของข้อมูล เพื่อหาค่าน้ำหนักข้อมูล เชื่อมต่อไปที่ 17
def get_physic_gate_preview(predicted_class, text):
    # 0:None, 1: Patent (Hight Complexity) , 2: Copyright (medium Complexity)
    weights = {0:1, 1:8.5, 2:6.5}
    base_weight = weights.get(predicted_class, 1.0)
    # ปรับน้ำหนักตามความรุนแรงที่อยู่ในข้อความ
    if "ร้ายแรง" in text or "จำนวนมาก" in text:
        base_weight = min(base_weight+1.0,10)
    return base_weight
#Run Program
text = "การละเมิดสิทธิบัตรรายใหญ่"
weight = get_physic_gate_preview(1,"text")
print(f"Physics Gate Weight review: {weight}/10.0")

# 4. Creation  JSON
import json
from datetime import datetime
def create_json_entry(doc_id, text):
    label = detect_category(text)
    conf, signals = cal_confidence(text,label)
    weight = get_physic_gate_preview(label,text)
    entry = {
        "id" : f"LAW-{doc_id:04d}",
        "text": text,
        "label": label,
        "metadata":{
            "confidence": conf,
            "context-signals": signals,
            "physic_gate_weight": weight,
            "processed_at": datetime.now().isoformat(),
            "requires_expert_review": conf <0.85
        }
    }
    return entry
# รัน Code  แสดงตัวอย่าง JSON
sample_entry = create_json_entry(1,"ละเมิดสิทธิบัตรการประดิษฐ์")
print(json.dumps(sample_entry, indent=4,ensure_ascii=False))



