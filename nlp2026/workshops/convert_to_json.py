"""
convert_to_json.py (v3 — Professional Edition)
------------------
อ่าน data/raw/thai_ip_corpus.txt → สร้าง workshops/data/processed/thai_ip_corpus.json
พร้อม Silver Standard metadata, Context-aware Confidence, และ Ambiguity Flagging

Changelog v3 (15 April 2026):
    [v3-1] เพิ่ม Context-aware Confidence Scoring
    [v3-2] เพิ่ม Legal Hierarchy Classification
    [v3-3] เพิ่ม Ambiguity Flagging
    [v3-4] เพิ่ม Physics Gate Weight Preview
"""
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'thai_ip_corpus.txt')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'thai_ip_corpus.json')

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# =============================================================================
# Keyword Definitions (Expanded)
# =============================================================================

VIOLATION_KEYWORDS = [
    "ละเมิด", "ปลอมแปลง", "เลียนแบบ", "ทำซ้ำ", "ดัดแปลง",
    "เผยแพร่", "จำหน่าย", "ละเมิดสิทธิ", "ละเมิดลิขสิทธิ์",
    "ละเมิดสิทธิบัตร", "การละเมิด", "กระทำผิด", "ฝ่าฝืน",
]

PATENT_KEYWORDS = [
    "สิทธิบัตร", "การประดิษฐ์", "เครื่องหมายการค้า",
    "สิทธิบัตรการประดิษฐ์", "แบบอรรถประโยชน์", "อนุสิทธิบัตร",
    "จดทะเบียนสิทธิบัตร", "คำขอรับสิทธิบัตร", "ผู้ทรงสิทธิ",
]

COPYRIGHT_KEYWORDS = [
    "ลิขสิทธิ์", "วรรณกรรม", "ดนตรีกรรม", "ภาพยนตร์",
    "สิ่งบันทึกเสียง", "งานสร้างสรรค์", "ผู้สร้างสรรค์",
    "เจ้าของลิขสิทธิ์", "ซอฟต์แวร์", "โปรแกรมคอมพิวเตอร์",
]

# Negation patterns
NEGATION_PATTERNS = [
    r"ไม่(?:ได้)?\s*(?:ละเมิด|กระทำผิด|ฝ่าฝืน)",
    r"มิได้\s*(?:ละเมิด|กระทำผิด|ฝ่าฝืน)",
    r"ได้รับอนุญาต",
    r"ถูกต้องตามกฎหมาย",
    r"ไม่เข้าข่าย",
    r"ไม่อยู่ในขอบเขต",
    r"ไม่ถือว่า",
]

# =============================================================================
# Context-aware Confidence (NEW in v3)
# =============================================================================

CONTEXT_BOOST_KEYWORDS = {
    "PATENT": ["สิทธิบัตร", "การประดิษฐ์", "แบบอรรถประโยชน์"],
    "COPYRIGHT": ["ลิขสิทธิ์", "งานสร้างสรรค์", "วรรณกรรม"],
    "PENALTY": ["จำคุก", "ปรับ", "โทษ", "ระวางโทษ"],
}


def compute_context_confidence(text: str, label: int, base_confidence: float) -> float:
    """คำนวณ confidence โดยพิจารณาบริบท"""
    confidence = base_confidence

    if label == 1:  # PATENT_VIOLATION
        context_boost = any(
            kw in text for kw in CONTEXT_BOOST_KEYWORDS["PATENT"])
        if context_boost:
            confidence *= 1.15
    elif label == 2:  # COPYRIGHT_VIOLATION
        context_boost = any(
            kw in text for kw in CONTEXT_BOOST_KEYWORDS["COPYRIGHT"])
        if context_boost:
            confidence *= 1.15

    # Penalty context เพิ่มความน่าเชื่อถือ
    if any(kw in text for kw in CONTEXT_BOOST_KEYWORDS["PENALTY"]):
        confidence *= 1.05

    return min(1.0, confidence)

# =============================================================================
# Legal Hierarchy Classification (NEW in v3)
# =============================================================================


LEGAL_HIERARCHY = {
    "CONSTITUTION": {"level": 1, "name": "รัฐธรรมนูญ"},
    "ACT": {"level": 2, "name": "พระราชบัญญัติ"},
    "ROYAL_DECREE": {"level": 3, "name": "พระราชกฤษฎีกา"},
    "MINISTERIAL_REGULATION": {"level": 4, "name": "กฎกระทรวง"},
}

OFFENSE_CLASSIFICATION = {
    "PATENT_VIOLATION": {
        "compoundable": False,  # ความผิดอาญาแผ่นดิน
        "severity": 7,
        "can_settle": False,
    },
    "COPYRIGHT_VIOLATION": {
        "compoundable": True,   # ความผิดอันยอมความได้
        "severity": 6,
        "can_settle": True,
    },
}


def classify_offense(label: int, text: str) -> Dict:
    """จำแนกประเภทความผิด"""
    if label == 1:
        return OFFENSE_CLASSIFICATION["PATENT_VIOLATION"]
    elif label == 2:
        return OFFENSE_CLASSIFICATION["COPYRIGHT_VIOLATION"]
    else:
        return {"compoundable": None, "severity": 0, "can_settle": None}

# =============================================================================
# Ambiguity Detection (NEW in v3)
# =============================================================================


AMBIGUOUS_PATTERNS = [
    ("สิทธิ์", ["สิทธิ์", "สิท-ธิ์"]),
    ("บัตร", ["บัตร", "บั-ตร"]),
    ("ประดิษฐ์", ["ประดิษฐ์", "ประ-ดิษฐ์"]),
]


def detect_ambiguity(text: str) -> Tuple[bool, List[str]]:
    """ตรวจจับความกำกวมในการตัดคำ"""
    ambiguous_found = []
    for word, variants in AMBIGUOUS_PATTERNS:
        if word in text:
            ambiguous_found.append(f"{word}({','.join(variants)})")
    return len(ambiguous_found) > 0, ambiguous_found

# =============================================================================
# Physics Gate Weight Preview (NEW in v3)
# =============================================================================


def compute_physics_gate_weight(label: int, severity: int, confidence: float) -> float:
    """คำนวณน้ำหนักเบื้องต้นสำหรับ Physics Gate (W17)"""
    base_weight = severity * confidence
    if label == 1:  # PATENT_VIOLATION
        base_weight *= 1.1  # patent cases often have higher stakes
    return min(10.0, base_weight)

# =============================================================================
# Main Functions
# =============================================================================


def has_negation(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in NEGATION_PATTERNS)


def get_matching_rules(text: str) -> Tuple[List[str], float]:
    rules_matched = []
    negated = has_negation(text)

    has_violation = any(kw in text for kw in VIOLATION_KEYWORDS)
    if has_violation:
        rules_matched.append("violation")

    has_patent = any(kw in text for kw in PATENT_KEYWORDS)
    has_copyright = any(kw in text for kw in COPYRIGHT_KEYWORDS)

    if has_patent:
        rules_matched.append("patent")
    if has_copyright:
        rules_matched.append("copyright")

    type_match = 1.0 if (has_patent or has_copyright) else 0.0
    max_score = 2.0
    confidence = (1.0 if has_violation else 0.0 + type_match) / max_score

    if negated and has_violation:
        confidence *= 0.5
        rules_matched.append("negation_detected")

    return rules_matched, min(confidence, 1.0)


def label_text_with_confidence(text: str) -> Tuple[int, float, List[str], float]:
    """
    คืนค่า (label, base_confidence, rules_matched, context_confidence)
    """
    rules_matched, base_confidence = get_matching_rules(text)

    has_violation = "violation" in rules_matched
    has_patent = "patent" in rules_matched
    has_copyright = "copyright" in rules_matched
    negated = "negation_detected" in rules_matched

    if negated:
        label = 0
    elif not has_violation:
        label = 0
    elif has_copyright:
        label = 2
    elif has_patent:
        label = 1
    else:
        label = 0
        base_confidence *= 0.6

    # Apply context-aware confidence
    context_confidence = compute_context_confidence(
        text, label, base_confidence)

    return label, base_confidence, rules_matched, context_confidence


def get_review_priority(confidence: float) -> Tuple[str, bool]:
    if confidence >= 0.8:
        return "low", False
    elif confidence >= 0.5:
        return "medium", True
    else:
        return "high", True


def convert():
    """Main conversion function with v3 enhancements"""

    print(f"\n{'='*60}")
    print("  convert_to_json.py v3 — Professional Edition")
    print("  Silver Standard + Context Confidence + Legal Hierarchy")
    print(f"{'='*60}")

    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        raw = f.read()

    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]

    samples = []
    total_confidence = 0.0
    label_counts = {0: 0, 1: 0, 2: 0}
    label_names = {0: "NO_INFRINGEMENT",
                   1: "PATENT_VIOLATION", 2: "COPYRIGHT_VIOLATION"}
    total_ambiguous = 0

    for i, text in enumerate(lines):
        label, base_conf, rules, context_conf = label_text_with_confidence(
            text)
        review_priority, needs_review = get_review_priority(context_conf)

        # Legal hierarchy classification
        offense_class = classify_offense(label, text)

        # Ambiguity detection
        is_ambiguous, ambiguous_terms = detect_ambiguity(text)
        if is_ambiguous:
            total_ambiguous += 1

        # Physics Gate weight preview
        physics_weight = compute_physics_gate_weight(
            label, offense_class["severity"], context_conf)

        samples.append({
            "id": i,
            "text": text,
            "silver_label": label,
            "label_name": label_names[label],
            "base_confidence": round(base_conf, 3),
            "context_confidence": round(context_conf, 3),
            "final_confidence": round(context_conf, 3),
            "heuristic_rules_applied": rules,
            "requires_expert_review": needs_review,
            "review_priority": review_priority,
            "source": "thai_ip_corpus.txt",
            # v3 enhancements
            "legal_hierarchy": {
                "offense_type": "non_compoundable" if offense_class.get("compoundable") is False else "compoundable" if offense_class.get("compoundable") else None,
                "severity": offense_class.get("severity", 0),
                "can_settle": offense_class.get("can_settle"),
            },
            "ambiguity_flags": {
                "has_ambiguity": is_ambiguous,
                "ambiguous_terms": ambiguous_terms,
            },
            "physics_gate": {
                "preview_weight": round(physics_weight, 2),
                "note": "Preview only — full calculation in W17",
            },
        })

        total_confidence += context_conf
        label_counts[label] += 1

    avg_confidence = total_confidence / len(samples) if samples else 0.0
    ambiguity_rate = total_ambiguous / len(samples) if samples else 0.0

    # Create output with metadata
    output = {
        "metadata": {
            "version": "3.0",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "labeling_method": "weak_supervision_heuristic",
            "labeling_confidence_level": "silver_standard",
            "requires_legal_review": True,
            "gold_standard_required": True,
            "disclaimer": (
                "Labels are heuristic-based, not legal advice. "
                "All samples must be verified by a legal expert (นิติกร) "
                "before use in production or legal proceedings."
            ),
            "inter_annotator_agreement": None,
            "heuristic_version": "3.0_professional",
            "v3_enhancements": [
                "context_aware_confidence",
                "legal_hierarchy_classification",
                "ambiguity_detection",
                "physics_gate_weight_preview",
            ],
            "dataset_stats": {
                "total_samples": len(samples),
                "class_distribution": {
                    label_names[0]: label_counts[0],
                    label_names[1]: label_counts[1],
                    label_names[2]: label_counts[2],
                },
                "average_confidence": round(avg_confidence, 3),
                "samples_requiring_review": sum(1 for s in samples if s["requires_expert_review"]),
                "ambiguity_rate": round(ambiguity_rate, 3),
            }
        },
        "samples": samples
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n  ✅ Converted {len(samples)} lines → {OUT_PATH}")
    print(f"\n  📊 Silver Standard Summary (v3 Professional):")
    print(f"     Labeling method    : weak_supervision_heuristic")
    print(f"     Avg base confidence: {avg_confidence:.1%}")
    print(f"     Ambiguity rate     : {ambiguity_rate:.1%}")
    print(
        f"     Requires review    : {sum(1 for s in samples if s['requires_expert_review'])} samples")
    print(f"\n  📈 Class Distribution:")

    max_count = max(label_counts.values()) if label_counts.values() else 1
    for label_id in [0, 1, 2]:
        name = label_names[label_id]
        count = label_counts[label_id]
        bar = "█" * int(count / max_count * 20)
        print(f"     {name:<25} {count:>3}  {bar}")

    print(f"\n  🆕 v3 Professional Features Added:")
    print(f"     ✓ Context-aware Confidence Scoring")
    print(f"     ✓ Legal Hierarchy Classification")
    print(f"     ✓ Ambiguity Detection")
    print(f"     ✓ Physics Gate Weight Preview (W17)")

    print(f"\n  ⚠️  IMPORTANT:")
    print(f"     These are SILVER STANDARD labels (Weak Supervision)")
    print(f"     ต้องให้นิติกรตรวจสอบก่อนใช้งานจริง")

    # Show high priority samples
    high_priority = [s for s in samples if s["review_priority"] == "high"][:3]
    if high_priority:
        print(f"\n  🔍 High Priority Review Samples:")
        for s in high_priority:
            print(
                f"     [{s['id']}] {s['text'][:50]}... (conf={s['final_confidence']:.0%})")
            if s['ambiguity_flags']['has_ambiguity']:
                print(
                    f"          ⚠️  Ambiguous: {', '.join(s['ambiguity_flags']['ambiguous_terms'])}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    convert()
