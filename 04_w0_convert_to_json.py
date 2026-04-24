import re
import json
from datetime import datetime

# 1. กำหนดรูปแบบคำศัพท์ (ใช้ Compiled Regex เพื่อความเร็ว)
RE_VIOLATION = re.compile(r"ละเมิด|ปลอมแปลง|เลียนแบบ|ทำซ้ำ|ดัดแปลง")
RE_PATENT = re.compile(r"สิทธิบัตร|การประดิษฐ์|ผังภูมิวงจร")
RE_COPYRIGHT = re.compile(r"ลิขสิทธิ์|วรรณกรรม|ศิลปกรรม|ดนตรีกรรม")

def detect_category(text):
    # ตรวจสอบการละเมิดก่อน
    if not RE_VIOLATION.search(text):
        return 0
    
    if RE_PATENT.search(text): return 1
    if RE_COPYRIGHT.search(text): return 2
    return 0

# 2. Context-aware Confidence
def cal_confidence(text, predicted_class):
    base_conf = 0.70
    signals = []
    
    # เพิ่มคำค้นหาให้ครอบคลุมบริบทกฎหมายไทย
    if any(k in text for k in ["มาตรา", "พ.ร.บ.", "พระราชบัญญัติ"]):
        base_conf += 0.15
        signals.append("statutory_ref")
    if any(k in text for k in ["คำพิพากษา", "ฎีกา", "ศาล"]):
        base_conf += 0.10
        signals.append("precedent_ref")
        
    return round(min(base_conf, 0.99), 2), signals

# 3. ลำดับศักดิ์ของข้อมูล (Physics Gate)
def get_physic_gate_preview(predicted_class, text):
    weights = {0: 1.0, 1: 8.5, 2: 6.5}
    base_weight = weights.get(predicted_class, 1.0)
    
    # ตรวจสอบความรุนแรง
    if any(k in text for k in ["ร้ายแรง", "จำนวนมาก", "รายใหญ่", "มูลค่าสูง"]):
        base_weight = min(base_weight + 1.5, 10.0)
        
    return base_weight

# 4. Creation JSON
def create_json_entry(doc_id, text):
    label = detect_category(text)
    conf, signals = cal_confidence(text, label)
    weight = get_physic_gate_preview(label, text)
    
    return {
        "id": f"LAW-{doc_id:04d}",
        "text": text,
        "label": label,
        "metadata": {
            "confidence": conf,
            "context_signals": signals,
            "physic_gate_weight": weight,
            "processed_at": datetime.now().isoformat(timespec='seconds'),
            "requires_expert_review": conf < 0.85
        }
    }

# --- ทดสอบการรันระบบ ---
sample_text = "ละเมิดสิทธิบัตรการประดิษฐ์รายใหญ่ ตามมาตรา 20 แห่ง พ.ร.บ. สิทธิบัตร"
entry = create_json_entry(1, sample_text)

print(json.dumps(entry, indent=4, ensure_ascii=False))