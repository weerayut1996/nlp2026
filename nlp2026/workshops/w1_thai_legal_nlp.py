"""
=============================================================================
WORKSHOP 1: Thai Legal Text Processing (v3 — Professional Edition)
=============================================================================
LEARNING OBJECTIVES:
    1. Tokenization ภาษาไทยสำหรับข้อความกฎหมาย IP
    2. Word Embeddings (TF-IDF + Co-occurrence Matrix)
    3. Legal Entity Extraction (มาตรา, ประเภทความผิด, โทษ)
    4. Keyword Analysis สำหรับ Patent & Copyright
    5. เชื่อมกับ Pipeline ที่สร้างไว้ใน PoC

ENHANCEMENTS in v3 (15 April 2026):
    [v3-1] Context-aware Confidence Scoring — ปรับ confidence ตามบริบท
    [v3-2] Legal Hierarchy Classification — ลำดับศักดิ์ของกฎหมาย
    [v3-3] Ambiguity Monitor — ตรวจจับความกำกวมในการตัดคำ
    [v3-4] Physics Gate Weight Preview — สำหรับ W17
    [v3-5] Professional Quality Report — สำหรับ PhD thesis
=============================================================================
"""
import os
import re
import json
import numpy as np
import collections
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

# =============================================================================
# SECTION 0: Dataset — โหลดข้อมูลจาก JSON (รองรับ Silver Standard)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(BASE_DIR)
DATA_PATH = (
    os.path.join(BASE_DIR, 'workshops', 'data', 'processed', 'thai_ip_corpus.json')
    if os.path.isdir(os.path.join(BASE_DIR, 'workshops', 'data'))
    else os.path.join(_parent, 'workshops','data', 'processed', 'thai_ip_corpus.json')
)


def load_legal_corpus(file_path: str) -> Tuple[List[str], Dict]:
    """
    โหลดข้อมูลจาก JSON พร้อม metadata
    Returns: (list of texts, metadata dict)
    """
    if not os.path.exists(file_path):
        print(f"❌ ไม่พบไฟล์ที่: {file_path}")
        print(f"   กรุณารัน convert_to_json.py ก่อนครับ")
        return [
            "มาตรา ๓๖ ผู้ทรงสิทธิบัตรเท่านั้นมีสิทธิในการผลิตและจำหน่าย",
            "การละเมิดลิขสิทธิ์ต้องระวางโทษจำคุกและปรับ",
        ], {"labeling_method": "fallback"}

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # รองรับทั้งโครงสร้างแบบมี metadata (v2/v3) และแบบ list เดิม (v1)
    if isinstance(data, dict) and "metadata" in data:
        metadata = data["metadata"]
        samples = data["samples"]
        texts = [item['text'] for item in samples]

        # แสดงคำเตือน如果是 Silver Standard
        if metadata.get("labeling_confidence_level") == "silver_standard":
            print("\n" + "⚠️ " * 20)
            print("  SILVER STANDARD MODE — Weak Supervision Detected")
            print(
                f"  Labeling method: {metadata.get('labeling_method', 'unknown')}")
            print(f"  Version: {metadata.get('version', 'unknown')}")
            if metadata.get('dataset_stats', {}).get('average_confidence'):
                print(
                    f"  Average confidence: {metadata['dataset_stats']['average_confidence']:.1%}")
            print(f"  ⚠️  ต้องให้นิติกรตรวจสอบก่อนใช้งานจริง ⚠️")

            # แสดง v3 features ถ้ามี
            if 'v3_enhancements' in metadata:
                print(
                    f"  ✨ v3 Features: {', '.join(metadata['v3_enhancements'])}")
            print("⚠️ " * 20 + "\n")
        return texts, metadata
    else:
        # Fallback for v1 format
        return [item['text'] for item in data], {"labeling_method": "v1_heuristic"}


THAI_IP_CORPUS, CORPUS_METADATA = load_legal_corpus(DATA_PATH)
print(f"✅ Loaded {len(THAI_IP_CORPUS)} documents")


# =============================================================================
# SECTION 1: Thai Tokenizer (Enhanced)
# =============================================================================

class ThaiLegalTokenizer:
    LEGAL_COMPOUNDS = [
        "สิทธิบัตรการประดิษฐ์", "ลิขสิทธิ์", "เครื่องหมายการค้า",
        "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ", "การประดิษฐ์ขึ้นใหม่",
        "ขั้นการประดิษฐ์", "ทางอุตสาหกรรม", "จำคุก", "ปรับ",
        "ริบทรัพย์", "เจ้าของสิทธิ", "ผู้ทรงสิทธิ", "คำขอรับสิทธิบัตร",
        "การดัดแปลง", "การเผยแพร่", "การทำซ้ำ", "พระราชบัญญัติ",
        "คณะกรรมการ", "พนักงานเจ้าหน้าที่", "อุทธรณ์",
        "ศาลทรัพย์สินทางปัญญา", "เครื่องหมายบริการ", "เครื่องหมายรับรอง",
    ]

    STOP_WORDS = {
        "และ", "หรือ", "ที่", "ใน", "ของ", "ตาม", "โดย", "กับ",
        "แก่", "ซึ่ง", "เพื่อ", "จาก", "ไว้", "ได้", "มี", "เป็น",
        "ต้อง", "ให้", "แห่ง", "นี้", "นั้น", "ว่า", "แต่", "ทั้ง",
        "ดัง", "ดังนี้", "ตั้งแต่", "ถึง", "ทั้งจำทั้งปรับ",
    }

    def __init__(self):
        self.compounds = sorted(self.LEGAL_COMPOUNDS, key=len, reverse=True)

    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        protected = text
        placeholders: Dict[str, str] = {}
        for i, compound in enumerate(self.compounds):
            if compound in protected:
                ph = f"__COMPOUND_{i}__"
                placeholders[ph] = compound
                protected = protected.replace(compound, ph)

        raw_tokens = re.split(
            r'[\s\u0020\u00a0\u3000]+|(?<=[ก-๙])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[ก-๙])',
            protected,
        )

        tokens = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            resolved = placeholders.get(tok, tok)
            parts = re.split(r'([,\.\(\)\[\]๑-๙0-9]+)', resolved)
            for p in parts:
                p = p.strip()
                if p:
                    tokens.append(p)

        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]
        # กรอง placeholder ที่อาจหลุดออกมา
        tokens = [t for t in tokens if not re.search(r'__COMPOUND_\d*', t)]
        return [t for t in tokens if len(t) > 0]

    def tokenize_corpus(self, corpus: List[str],
                        remove_stopwords: bool = True) -> List[List[str]]:
        return [self.tokenize(doc, remove_stopwords) for doc in corpus]


# =============================================================================
# SECTION 2: Legal Entity Extractor — Professional Edition (v3)
# =============================================================================

@dataclass
class LegalEntity:
    entity_type: str
    value: str
    context: str
    position: int
    law_reference: Optional[str] = None
    confidence: float = 1.0
    context_signals: List[str] = field(default_factory=list)
    semantic_weight: float = 1.0


class ContextAwareConfidenceScorer:
    """
    Context-aware Confidence Scoring สำหรับ Legal Entity Extraction

    หลักการ: ความเชื่อมั่นของ entity ไม่ได้ขึ้นอยู่กับ pattern เพียงอย่างเดียว
    แต่ขึ้นอยู่กับบริบทโดยรอบด้วย
    """

    CONTEXT_SIGNALS = {
        "PATENT_CONTEXT": {
            "keywords": ["สิทธิบัตร", "การประดิษฐ์", "แบบอรรถประโยชน์", "อนุสิทธิบัตร"],
            "boost": 1.3,
            "penalty": 0.7,
        },
        "COPYRIGHT_CONTEXT": {
            "keywords": ["ลิขสิทธิ์", "งานสร้างสรรค์", "วรรณกรรม", "ดนตรีกรรม", "ซอฟต์แวร์"],
            "boost": 1.3,
            "penalty": 0.7,
        },
        "TRADEMARK_CONTEXT": {
            "keywords": ["เครื่องหมายการค้า", "เครื่องหมายบริการ", "ตราสินค้า"],
            "boost": 1.3,
            "penalty": 0.7,
        },
        "PENALTY_CONTEXT": {
            "keywords": ["จำคุก", "ปรับ", "ริบ", "โทษ", "ระวางโทษ"],
            "boost": 1.2,
            "penalty": 0.8,
        },
        "STATUTE_CONTEXT": {
            "keywords": ["มาตรา", "พ.ร.บ.", "พระราชบัญญัติ", "วรรค"],
            "boost": 1.15,
            "penalty": 0.85,
        },
    }

    ENTITY_CONTEXT_MAP = {
        "IP_TYPE": ["PATENT_CONTEXT", "COPYRIGHT_CONTEXT", "TRADEMARK_CONTEXT"],
        "PENALTY": ["PENALTY_CONTEXT"],
        "STATUTE": ["STATUTE_CONTEXT"],
        "ACTION": [],
    }

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def compute_confidence(self, entity: LegalEntity, full_text: str) -> Tuple[float, List[str]]:
        """คำนวณ confidence โดยพิจารณาบริบท"""
        base_confidence = entity.confidence

        start = max(0, entity.position - self.window_size)
        end = min(len(full_text), entity.position +
                  len(entity.value) + self.window_size)
        context_window = full_text[start:end]

        detected_signals = []
        adjusted_conf = base_confidence

        relevant_contexts = self.ENTITY_CONTEXT_MAP.get(entity.entity_type, [])

        for ctx_name in relevant_contexts:
            ctx_config = self.CONTEXT_SIGNALS.get(ctx_name, {})
            keywords = ctx_config.get("keywords", [])
            boost = ctx_config.get("boost", 1.0)
            penalty = ctx_config.get("penalty", 1.0)

            has_context = any(kw in context_window for kw in keywords)

            if has_context:
                detected_signals.append(f"{ctx_name}_present")
                adjusted_conf *= boost
            else:
                if relevant_contexts:
                    detected_signals.append(f"{ctx_name}_absent")
                    adjusted_conf *= penalty

        # กรณีพิเศษ: STATUTE + IP_type mismatch detection
        if entity.entity_type == "STATUTE" and entity.law_reference:
            ip_types_in_context = []
            for ip_ctx in ["PATENT_CONTEXT", "COPYRIGHT_CONTEXT", "TRADEMARK_CONTEXT"]:
                if any(kw in context_window for kw in self.CONTEXT_SIGNALS[ip_ctx]["keywords"]):
                    ip_types_in_context.append(ip_ctx)

            if ip_types_in_context:
                adjusted_conf *= 0.85
                detected_signals.append("statute_without_ip_type")

        final_conf = min(1.0, max(0.1, adjusted_conf))
        return final_conf, detected_signals


class ThaiIPEntityExtractor:
    PATTERNS = {
        "STATUTE": [
            r'มาตรา\s*[๐-๙\d]+',
            r'พ\.ร\.บ\.\s*[\w\s]+พ\.ศ\.\s*[๐-๙\d]+',
        ],
        "PENALTY": [
            r'จำคุก(?:ไม่เกิน|ตั้งแต่)?\s*[\w\s]+(?:ปี|เดือน)',
            r'ปรับ(?:ไม่เกิน|ตั้งแต่)?\s*[\w\s]+บาท',
        ],
        "IP_TYPE": [r'สิทธิบัตร', r'ลิขสิทธิ์', r'เครื่องหมายการค้า'],
        "ACTION": [r'ละเมิด', r'ทำซ้ำ', r'ดัดแปลง', r'เผยแพร่'],
    }

    def __init__(self, use_context_aware: bool = True):
        self.use_context_aware = use_context_aware
        self.confidence_scorer = ContextAwareConfidenceScorer() if use_context_aware else None

    def extract(self, text: str) -> List[LegalEntity]:
        entities = []
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)

                    base_conf = 0.95 if entity_type == "STATUTE" else 0.85

                    entity = LegalEntity(
                        entity_type=entity_type,
                        value=match.group().strip(),
                        context=text[start:end].strip(),
                        position=match.start(),
                        law_reference=self._find_statute(text, match.start()),
                        confidence=base_conf,
                    )

                    if self.use_context_aware and self.confidence_scorer:
                        adjusted_conf, signals = self.confidence_scorer.compute_confidence(
                            entity, text)
                        entity.confidence = adjusted_conf
                        entity.context_signals = signals
                        entity.semantic_weight = adjusted_conf

                    entities.append(entity)

        return sorted(entities, key=lambda e: e.position)

    def _find_statute(self, text: str, pos: int) -> Optional[str]:
        matches = list(re.finditer(r'มาตรา\s*[๐-๙\d]+', text[:pos]))
        return matches[-1].group().strip() if matches else None


# =============================================================================
# SECTION 2.2: Legal Hierarchy Classification (v3)
# =============================================================================

@dataclass
class LegalHierarchyNode:
    """โหนดในลำดับศักดิ์ของกฎหมาย"""
    law_type: str
    law_name: str
    law_number: str
    year: int
    parent: Optional['LegalHierarchyNode'] = None
    children: List['LegalHierarchyNode'] = field(default_factory=list)
    offense_type: Optional[str] = None
    penalty_severity: Optional[float] = None


class ThaiLegalHierarchy:
    """
    ลำดับศักดิ์ของกฎหมายไทย (Hierarchy of Laws)

    ใช้สำหรับ:
        1. ระบุว่า offense นี้เป็น "ความผิดอันยอมความได้" หรือ "ความผิดอาญาแผ่นดิน"
        2. คำนวณ severity score สำหรับ Physics Gate (W17)
        3. เชื่อมโยง statute กับประเภทของกฎหมาย
    """

    HIERARCHY_LEVELS = {
        "CONSTITUTION": 1,
        "ACT": 2,
        "EMERGENCY_DECREE": 2,
        "ROYAL_DECREE": 3,
        "MINISTERIAL_REGULATION": 4,
        "LOCAL_REGULATION": 5,
        "BYLAW": 6,
    }

    PATENT_OFFENSES = {
        "SECTION_36": {
            "section": 36,
            "description": "ผู้ทรงสิทธิบัตรเท่านั้นมีสิทธิผลิตและจำหน่าย",
            "offense_type": "non_compoundable",
            "penalty": "จำคุกไม่เกิน 2 ปี หรือปรับไม่เกิน 400,000 บาท หรือทั้งจำทั้งปรับ",
            "severity": 6,
        },
        "SECTION_77": {
            "section": 77,
            "description": "ละเมิดสิทธิบัตร",
            "offense_type": "compoundable",
            "penalty": "จำคุกไม่เกิน 2 ปี หรือปรับไม่เกิน 400,000 บาท หรือทั้งจำทั้งปรับ",
            "severity": 7,
        },
    }

    COPYRIGHT_OFFENSES = {
        "SECTION_69": {
            "section": 69,
            "description": "ทำซ้ำ ดัดแปลง หรือเผยแพร่โดยไม่ได้รับอนุญาต",
            "offense_type": "compoundable",
            "penalty": "จำคุก 3 เดือนถึง 2 ปี หรือปรับ 20,000-200,000 บาท",
            "severity": 5,
        },
        "SECTION_70": {
            "section": 70,
            "description": "ละเมิดลิขสิทธิ์เพื่อการค้า",
            "offense_type": "non_compoundable",
            "penalty": "จำคุก 6 เดือนถึง 4 ปี หรือปรับ 100,000-800,000 บาท",
            "severity": 8,
        },
    }

    def __init__(self):
        """Initialize legal hierarchy and build structure"""
        self.constitution = None
        self.ip_act = None
        self.copyright_act = None
        self._build_hierarchy()

    def _build_hierarchy(self):
        """สร้างโครงสร้างลำดับศักดิ์ของกฎหมาย"""
        self.constitution = LegalHierarchyNode(
            law_type="CONSTITUTION",
            law_name="รัฐธรรมนูญแห่งราชอาณาจักรไทย",
            law_number="2560",
            year=2560,
        )

        self.ip_act = LegalHierarchyNode(
            law_type="ACT",
            law_name="พระราชบัญญัติสิทธิบัตร",
            law_number="2522",
            year=2522,
            parent=self.constitution,
        )
        self.constitution.children.append(self.ip_act)

        self.copyright_act = LegalHierarchyNode(
            law_type="ACT",
            law_name="พระราชบัญญัติลิขสิทธิ์",
            law_number="2537",
            year=2537,
            parent=self.constitution,
        )
        self.constitution.children.append(self.copyright_act)

    def _thai_to_arabic(self, thai_num: str) -> int:
        """แปลงเลขไทยเป็นเลขอารบิก"""
        thai_digits = "๐๑๒๓๔๕๖๗๘๙"
        arabic_digits = "0123456789"
        trans = str.maketrans(thai_digits, arabic_digits)
        return int(thai_num.translate(trans))

    def classify_offense(self, statute_ref: str, ip_type: str) -> Dict:
        """
        จำแนกประเภทความผิดตาม statute และ IP type

        Args:
            statute_ref: เช่น "มาตรา 77" หรือ "มาตรา ๓๖"
            ip_type: "สิทธิบัตร" หรือ "ลิขสิทธิ์"

        Returns:
            Dict with offense_type, severity, can_settle, penalty_range
        """
        section_match = re.search(r'มาตรา\s*([๐-๙\d]+)', statute_ref)
        if not section_match:
            return {"offense_type": "unknown", "severity": 0, "can_settle": None}

        section_str = section_match.group(1)
        section_num = self._thai_to_arabic(section_str)

        if ip_type == "สิทธิบัตร":
            offense = self.PATENT_OFFENSES.get(f"SECTION_{section_num}", {})
        elif ip_type == "ลิขสิทธิ์":
            offense = self.COPYRIGHT_OFFENSES.get(f"SECTION_{section_num}", {})
        else:
            offense = {}

        if not offense:
            return {
                "offense_type": "unknown",
                "severity": 0,
                "can_settle": None,
                "penalty_range": "",
                "description": "",
            }

        return {
            "offense_type": offense.get("offense_type"),
            "severity": offense.get("severity", 0),
            "can_settle": offense.get("offense_type") == "compoundable",
            "penalty_range": offense.get("penalty", ""),
            "description": offense.get("description", ""),
        }

    def compute_physics_gate_weight(self, entities: List[LegalEntity]) -> float:
        """
        คำนวณน้ำหนักสำหรับ Physics Gate (W17)

        Formula: weight = severity × confidence × adjustment
            - adjustment = 0.8 for compoundable, 1.2 for non-compoundable

        Returns:
            float between 0 and 10
        """
        total_weight = 0.0

        for ent in entities:
            if ent.entity_type == "STATUTE" and ent.law_reference:
                ip_types = [
                    e.value for e in entities if e.entity_type == "IP_TYPE"]
                ip_type = ip_types[0] if ip_types else None

                if ip_type:
                    classification = self.classify_offense(
                        ent.law_reference, ip_type)
                    severity = classification.get("severity", 0)

                    weight = severity * ent.confidence

                    if classification.get("can_settle"):
                        weight *= 0.8
                    else:
                        weight *= 1.2

                    total_weight += weight

        return min(10.0, total_weight)

    def get_hierarchy_level(self, law_type: str) -> int:
        """Get hierarchy level (1=highest, 6=lowest)"""
        return self.HIERARCHY_LEVELS.get(law_type, 99)

    def is_higher_than(self, law_type1: str, law_type2: str) -> bool:
        """Check if law_type1 has higher authority than law_type2"""
        return self.get_hierarchy_level(law_type1) < self.get_hierarchy_level(law_type2)


# =============================================================================
# SECTION 2.3: Ambiguity Monitor (v3)
# =============================================================================

class ThaiLegalAmbiguityMonitor:
    """
    ตรวจสอบความกำกวมในการตัดคำ (Ambiguous Segmentation)
    """

    LEGAL_AMBIGUOUS_PATTERNS = {
        "สิทธิ์": {"segmentations": ["สิทธิ์", "สิท-ธิ์"], "legal_term": True},
        "บัตร": {"segmentations": ["บัตร", "บั-ตร"], "legal_term": True},
        "ประดิษฐ์": {"segmentations": ["ประดิษฐ์", "ประ-ดิษฐ์"], "legal_term": True},
        "อนุญาต": {"segmentations": ["อนุญาต", "อะ-นุ-ยาต"], "legal_term": True},
    }

    def __init__(self, tokenizer: ThaiLegalTokenizer):
        self.tokenizer = tokenizer
        self.ambiguity_log: List[Dict] = []

    def detect_ambiguous_segmentation(self, text: str) -> List[Dict]:
        ambiguous = []
        for word, config in self.LEGAL_AMBIGUOUS_PATTERNS.items():
            for match in re.finditer(re.escape(word), text):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]

                ambiguous.append({
                    "original": word,
                    "segmentations": config["segmentations"],
                    "position": match.start(),
                    "legal_term": config.get("legal_term", False),
                    "in_legal_context": any(clue in context for clue in ["สิทธิบัตร", "ลิขสิทธิ์"]),
                })
        return ambiguous

    def analyze_ambiguity_rate(self, corpus: List[str]) -> Dict:
        total_ambiguous = 0
        total_tokens = 0
        high_risk_sentences = []

        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            total_tokens += len(tokens)
            ambiguous = self.detect_ambiguous_segmentation(text)
            if ambiguous:
                total_ambiguous += len(ambiguous)
                if len(ambiguous) >= 2:
                    high_risk_sentences.append(text[:100])

        ambiguity_rate = total_ambiguous / max(total_tokens, 1)

        return {
            "total_ambiguous": total_ambiguous,
            "total_tokens": total_tokens,
            "ambiguity_rate": ambiguity_rate,
            "high_risk_sentences": high_risk_sentences[:10],
            "recommendation": self._get_recommendation(ambiguity_rate),
        }

    def _get_recommendation(self, rate: float) -> str:
        if rate < 0.01:
            return "✅ อัตราความกำกวมต่ำมาก Tokenizer ทำงานดี"
        elif rate < 0.05:
            return "⚠️ อัตราความกำกวมปานกลาง ควรตรวจสอบตัวอย่างสุ่ม"
        else:
            return "❌ อัตราความกำกวมสูง ต้องปรับปรุง tokenizer หรือใช้ W4 (WangchanBERTa)"

    def print_report(self, corpus: List[str]):
        report = self.analyze_ambiguity_rate(corpus)
        print("\n" + "═" * 70)
        print("  AMBIGUITY MONITOR REPORT — Thai Legal Text Segmentation")
        print("─" * 70)
        print(f"\n  📊 Ambiguity Statistics:")
        print(f"     Total ambiguous : {report['total_ambiguous']}")
        print(f"     Total tokens    : {report['total_tokens']}")
        print(f"     Ambiguity rate  : {report['ambiguity_rate']:.2%}")
        print(f"\n  💡 {report['recommendation']}")
        if report['high_risk_sentences']:
            print(f"\n  🔴 High Risk Sentences:")
            for i, sent in enumerate(report['high_risk_sentences'][:3]):
                print(f"     {i+1}. {sent}...")
        print("═" * 70)


# =============================================================================
# SECTION 3: OOV Handler (v2)
# =============================================================================

class OOVHandler:
    def __init__(self, strategy: str = "unk_token", unk_token: str = "[UNK]"):
        self.strategy = strategy
        self.unk_token = unk_token
        self.oov_count = 0
        self.oov_examples = []

    def handle_oov(self, token: str, vocabulary: Set[str]) -> Optional[str]:
        if token in vocabulary:
            return token
        self.oov_count += 1
        if len(self.oov_examples) < 10:
            self.oov_examples.append(token)
        if self.strategy == "ignore":
            return None
        elif self.strategy == "unk_token":
            return self.unk_token
        return None

    def get_report(self) -> Dict:
        return {
            "total_oov": self.oov_count,
            "sample_oov": self.oov_examples,
            "strategy": self.strategy,
            "warning": "OOV อาจลดประสิทธิภาพ" if self.oov_count > 0 else None,
        }


# =============================================================================
# SECTION 4: Legal Vocabulary Coverage (v2)
# =============================================================================

class LegalVocabularyCoverage:
    LEGAL_GLOSSARY = {
        "PATENT": ["สิทธิบัตร", "การประดิษฐ์", "แบบอรรถประโยชน์", "จดทะเบียน",
                   "ผู้ทรงสิทธิ", "คำขอรับสิทธิบัตร", "อายุสิทธิบัตร", "ขั้นการประดิษฐ์",
                   "ทางอุตสาหกรรม", "การออกแบบผลิตภัณฑ์", "อนุสิทธิบัตร"],
        "COPYRIGHT": ["ลิขสิทธิ์", "งานสร้างสรรค์", "ผู้สร้างสรรค์", "เจ้าของลิขสิทธิ์",
                      "ทำซ้ำ", "ดัดแปลง", "เผยแพร่", "อายุการคุ้มครอง", "งานวรรณกรรม",
                      "งานศิลปกรรม", "งานวิทยาศาสตร์", "ซอฟต์แวร์"],
        "TRADEMARK": ["เครื่องหมายการค้า", "เครื่องหมายบริการ", "เครื่องหมายรับรอง",
                      "เครื่องหมายร่วม", "จดทะเบียนเครื่องหมาย"],
        "ENFORCEMENT": ["ละเมิด", "ฝ่าฝืน", "ปลอมแปลง", "เลียนแบบ", "โดยไม่ได้รับอนุญาต",
                        "เรียกค่าเสียหาย", "มาตรา", "พระราชบัญญัติ", "พ.ร.บ.",
                        "ระวางโทษ", "จำคุก", "ปรับ", "ริบของกลาง", "ไต่สวน"],
        "PROCEDURE": ["คำขอ", "ผู้ขอ", "ผู้คัดค้าน", "คณะกรรมการ", "พนักงานเจ้าหน้าที่",
                      "อุทธรณ์", "ศาลทรัพย์สินทางปัญญา", "คำพิพากษา", "ฎีกา"],
    }

    def __init__(self):
        self.all_terms = set()
        for terms in self.LEGAL_GLOSSARY.values():
            self.all_terms.update(terms)

    def compute_coverage(self, vocabulary: List[str]) -> Dict:
        vocab_set = set(vocabulary)
        results = {
            "overall": {
                "covered": len(vocab_set & self.all_terms),
                "total": len(self.all_terms),
                "percentage": 0.0,
                "missing": list(self.all_terms - vocab_set)[:20],
            },
            "by_category": {},
        }
        for category, terms in self.LEGAL_GLOSSARY.items():
            covered = len(vocab_set & set(terms))
            total = len(terms)
            results["by_category"][category] = {
                "covered": covered, "total": total,
                "percentage": (covered / total * 100) if total > 0 else 0,
                "missing": list(set(terms) - vocab_set)[:10],
            }
        results["overall"]["percentage"] = results["overall"]["covered"] / \
            results["overall"]["total"] * 100
        return results

    def print_report(self, vocabulary: List[str], max_features: int = None):
        coverage = self.compute_coverage(vocabulary)
        print("\n" + "═" * 70)
        print("  LEGAL VOCABULARY COVERAGE REPORT")
        print("─" * 70)
        print(f"\n  📊 OVERALL COVERAGE:")
        print(
            f"     {coverage['overall']['covered']} / {coverage['overall']['total']} terms")
        bar = "█" * int(coverage['overall']['percentage'] / 5)
        print(f"     {coverage['overall']['percentage']:.1f}%  {bar}")
        if coverage['overall']['percentage'] < 80 and max_features:
            print(
                f"\n     ⚠️  แนะนำ: เพิ่ม max_features จาก {max_features} เป็น {max_features * 2}")
        print(f"\n  📚 COVERAGE BY CATEGORY:")
        for category, stats in coverage["by_category"].items():
            bar = "█" * int(stats["percentage"] / 10)
            print(f"  {category:<20} {stats['covered']:>3}/{stats['total']:<3}  "
                  f"{stats['percentage']:>5.1f}%  {bar}")
            if stats["percentage"] < 70 and stats["missing"]:
                print(f"     ⚠️  Missing: {', '.join(stats['missing'][:5])}")
        print("═" * 70)
        return coverage


# =============================================================================
# SECTION 5: TF-IDF Vectorizer with OOV Support
# =============================================================================

class LegalTFIDFWithOOV:
    def __init__(self, max_features: int = 100, oov_strategy: str = "unk_token"):
        self.max_features = max_features
        self.oov_handler = OOVHandler(strategy=oov_strategy)
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self._idf: np.ndarray = np.array([])

    def fit_transform(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        all_tokens = [t for doc in tokenized_corpus for t in doc]
        all_tokens.append("[UNK]")
        freq = collections.Counter(all_tokens)
        self.feature_names_ = [
            t for t, _ in freq.most_common(self.max_features)]
        if "[UNK]" not in self.feature_names_:
            self.feature_names_[-1] = "[UNK]"
        self.vocabulary_ = {t: i for i, t in enumerate(self.feature_names_)}

        n_docs = len(tokenized_corpus)
        V = len(self.feature_names_)
        matrix = np.zeros((n_docs, V), dtype=np.float32)

        for i, doc in enumerate(tokenized_corpus):
            if not doc:
                continue
            processed_doc = self._process_tokens(doc)
            if not processed_doc:
                continue
            doc_freq = collections.Counter(processed_doc)
            for term, count in doc_freq.items():
                if term in self.vocabulary_:
                    matrix[i, self.vocabulary_[term]
                           ] = count / len(processed_doc)

        df = (matrix > 0).sum(axis=0) + 1
        self._idf = np.log((n_docs + 1) / df) + 1
        matrix *= self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix /= norms

        if self.oov_handler.oov_count > 0:
            print(
                f"\n  ⚠️  OOV Warning: {self.oov_handler.oov_count} unknown tokens")
            print(f"     Samples: {self.oov_handler.oov_examples[:5]}")
        return matrix

    def _process_tokens(self, doc: List[str]) -> List[str]:
        processed = []
        for token in doc:
            result = self.oov_handler.handle_oov(
                token, set(self.vocabulary_.keys()))
            if result:
                processed.append(result)
        return processed

    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        if not self.vocabulary_:
            raise RuntimeError("Call fit_transform() first")
        V = len(self.feature_names_)
        n_docs = len(tokenized_docs)
        matrix = np.zeros((n_docs, V), dtype=np.float32)
        for i, doc in enumerate(tokenized_docs):
            processed_doc = self._process_tokens(doc)
            if not processed_doc:
                continue
            doc_freq = collections.Counter(processed_doc)
            for term, count in doc_freq.items():
                if term in self.vocabulary_:
                    matrix[i, self.vocabulary_[term]
                           ] = count / len(processed_doc)
        matrix *= self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix /= norms
        return matrix

    def get_top_terms(self, vector: np.ndarray, n: int = 5):
        top_idx = np.argsort(vector)[::-1][:n]
        return [(self.feature_names_[i], float(vector[i])) for i in top_idx if vector[i] > 0]


# =============================================================================
# SECTION 6: Co-occurrence Matrix
# =============================================================================

class LegalCooccurrence:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.vocab_: Dict[str, int] = {}
        self.matrix_: Optional[np.ndarray] = None

    def fit(self, tokenized_corpus: List[List[str]]):
        all_tokens = [t for doc in tokenized_corpus for t in doc]
        unique_tokens = list(set(all_tokens))
        self.vocab_ = {t: i for i, t in enumerate(unique_tokens)}
        V = len(unique_tokens)
        self.matrix_ = np.zeros((V, V), dtype=np.float32)
        for doc in tokenized_corpus:
            for i, word in enumerate(doc):
                if word not in self.vocab_:
                    continue
                idx1 = self.vocab_[word]
                lo = max(0, i - self.window_size)
                hi = min(len(doc), i + self.window_size + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    neighbor = doc[j]
                    if neighbor in self.vocab_:
                        self.matrix_[idx1, self.vocab_[neighbor]] += 1


# =============================================================================
# SECTION 7: W1Pipeline — Professional Edition (v3)
# =============================================================================

class W1Pipeline:
    """
    Encapsulates the full W1 processing pipeline with v3 enhancements.

    New features in v3:
        - Context-aware confidence scoring
        - Legal hierarchy classification
        - Ambiguity monitoring
        - Physics gate weight preview
        - Professional quality reporting
    """

    def __init__(self, max_features: int = 100, remove_stopwords: bool = True,
                 oov_strategy: str = "unk_token", evaluate_coverage: bool = True,
                 use_context_aware: bool = True, enable_legal_hierarchy: bool = True,
                 enable_ambiguity_monitor: bool = True):
        self.tokenizer = ThaiLegalTokenizer()
        self.tfidf = LegalTFIDFWithOOV(
            max_features=max_features, oov_strategy=oov_strategy)
        self.extractor = ThaiIPEntityExtractor(
            use_context_aware=use_context_aware)
        self.remove_stopwords = remove_stopwords
        self.evaluate_coverage = evaluate_coverage
        self.use_context_aware = use_context_aware
        self.enable_legal_hierarchy = enable_legal_hierarchy
        self.enable_ambiguity_monitor = enable_ambiguity_monitor
        self._fitted = False
        self.coverage_analyzer = LegalVocabularyCoverage() if evaluate_coverage else None
        self.legal_hierarchy = ThaiLegalHierarchy() if enable_legal_hierarchy else None
        self._ambiguity_monitor = None
        self.coverage_report = None
        self.coverage_suggestions = None

        # Silver standard warning
        if CORPUS_METADATA.get("labeling_confidence_level") == "silver_standard":
            print("\n" + "⚠️ " * 20)
            print("  SILVER STANDARD MODE ENABLED")
            print(
                "  Labels in this corpus are from Weak Supervision (Heuristic Matching)")
            print("  ต้องให้นิติกรตรวจสอบก่อนใช้งานจริง")
            print("⚠️ " * 20 + "\n")

    def _get_ambiguity_monitor(self):
        """Lazy initialization of ambiguity monitor"""
        if self.enable_ambiguity_monitor and self._ambiguity_monitor is None:
            self._ambiguity_monitor = ThaiLegalAmbiguityMonitor(self.tokenizer)
        return self._ambiguity_monitor

    def fit(self, texts: List[str]) -> "W1Pipeline":
        tokens = self.tokenizer.tokenize_corpus(texts, self.remove_stopwords)
        self.tfidf.fit_transform(tokens)
        self._fitted = True

        if self.evaluate_coverage and self.coverage_analyzer:
            self.coverage_report = self.coverage_analyzer.print_report(
                self.tfidf.feature_names_, self.tfidf.max_features
            )

        return self

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize_corpus(texts, self.remove_stopwords)
        X = self.tfidf.fit_transform(tokens)
        self._fitted = True

        if self.evaluate_coverage and self.coverage_analyzer:
            self.coverage_report = self.coverage_analyzer.print_report(
                self.tfidf.feature_names_, self.tfidf.max_features
            )

        return X

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "W1Pipeline: call fit() or fit_transform() first")
        tokens = self.tokenizer.tokenize_corpus(texts, self.remove_stopwords)
        return self.tfidf.transform(tokens)

    def extract_entities(self, text: str) -> List[LegalEntity]:
        return self.extractor.extract(text)

    def extract_entities_with_semantics(self, text: str) -> Tuple[List[LegalEntity], Dict]:
        """
        Extract entities พร้อม semantic analysis

        Returns:
            (entities, semantic_context)
        """
        entities = self.extractor.extract(text)

        semantic_context = {
            "offense_classification": None,
            "physics_gate_weight": 0.0,
            "can_settle": None,
        }

        if self.legal_hierarchy and entities:
            statutes = [e for e in entities if e.entity_type == "STATUTE"]
            ip_types = [
                e.value for e in entities if e.entity_type == "IP_TYPE"]

            if statutes and ip_types:
                classification = self.legal_hierarchy.classify_offense(
                    statutes[0].value, ip_types[0]
                )
                semantic_context["offense_classification"] = classification
                semantic_context["can_settle"] = classification.get(
                    "can_settle")
                semantic_context["physics_gate_weight"] = self.legal_hierarchy.compute_physics_gate_weight(
                    entities)

        return entities, semantic_context

    def get_oov_report(self) -> Dict:
        return self.tfidf.oov_handler.get_report()

    def get_coverage_report(self) -> Optional[Dict]:
        return self.coverage_report

    def analyze_ambiguity(self, corpus: List[str]) -> Optional[Dict]:
        monitor = self._get_ambiguity_monitor()
        if monitor:
            return monitor.analyze_ambiguity_rate(corpus)
        return None

    def print_ambiguity_report(self, corpus: List[str]):
        monitor = self._get_ambiguity_monitor()
        if monitor:
            monitor.print_report(corpus)

    def analyze_corpus_quality(self, corpus: List[str]) -> Dict:
        """วิเคราะห์คุณภาพของ corpus (สำหรับ PhD thesis)"""
        quality_report = {
            "coverage": self.coverage_report if self.coverage_report else None,
            "oov": self.get_oov_report(),
            "ambiguity": None,
            "recommendations": [],
        }

        # Ambiguity analysis
        ambiguity = self.analyze_ambiguity(corpus)
        if ambiguity:
            quality_report["ambiguity"] = ambiguity
            if ambiguity["ambiguity_rate"] > 0.05:
                quality_report["recommendations"].append(
                    "High ambiguity rate detected — consider using W4 (WangchanBERTa)"
                )

        # Coverage recommendations
        if self.coverage_report and self.coverage_report.get('overall', {}).get('percentage', 0) < 80:
            quality_report["recommendations"].append(
                f"Coverage {self.coverage_report['overall']['percentage']:.1f}% < 80% — increase max_features"
            )

        # OOV recommendations
        oov = self.get_oov_report()
        if oov["total_oov"] > 0:
            quality_report["recommendations"].append(
                f"OOV tokens detected ({oov['total_oov']}) — expand LEGAL_COMPOUNDS"
            )

        return quality_report

    def print_professional_report(self, corpus: List[str]):
        """พิมพ์รายงานระดับ professional สำหรับ PhD thesis"""
        print("\n" + "█" * 70)
        print("  PROFESSIONAL QUALITY REPORT — W1 Pipeline v3")
        print("  For PhD Thesis: Physics-Governed IoT Framework for IP Law Enforcement")
        print("█" * 70)

        quality = self.analyze_corpus_quality(corpus)

        # Coverage section
        if quality["coverage"]:
            print(f"\n  📚 LEGAL VOCABULARY COVERAGE:")
            print(
                f"     Overall: {quality['coverage']['overall']['percentage']:.1f}%")

        # OOV section
        oov = quality["oov"]
        print(f"\n  🔤 OUT-OF-VOCABULARY (OOV) STATISTICS:")
        print(f"     Total OOV tokens: {oov['total_oov']}")
        if oov["sample_oov"]:
            print(f"     Sample OOV: {', '.join(oov['sample_oov'][:5])}")

        # Ambiguity section
        if quality["ambiguity"]:
            amb = quality["ambiguity"]
            print(f"\n  ⚠️  AMBIGUITY ANALYSIS:")
            print(f"     Ambiguity rate: {amb['ambiguity_rate']:.2%}")
            print(f"     {amb['recommendation']}")

        # Recommendations
        if quality["recommendations"]:
            print(f"\n  💡 RECOMMENDATIONS FOR IMPROVEMENT:")
            for i, rec in enumerate(quality["recommendations"][:5]):
                print(f"     {i+1}. {rec}")

        print("\n" + "█" * 70)

    @property
    def n_features(self) -> int:
        return len(self.tfidf.feature_names_)


# =============================================================================
# SECTION 8: Robust Pipeline with Fallback (v2)
# =============================================================================

class RobustW1Pipeline(W1Pipeline):
    """
    Robust pipeline ที่มี fallback strategies

    เมื่อเจอข้อความที่ OOV เยอะเกินไป:
        Fallback to character-level tokenization
    """

    def __init__(self, max_features: int = 100, remove_stopwords: bool = True,
                 oov_strategy: str = "unk_token", evaluate_coverage: bool = True,
                 use_context_aware: bool = True, enable_legal_hierarchy: bool = True,
                 enable_ambiguity_monitor: bool = True,
                 fallback_threshold: float = 0.3):
        super().__init__(max_features, remove_stopwords, oov_strategy,
                         evaluate_coverage, use_context_aware,
                         enable_legal_hierarchy, enable_ambiguity_monitor)
        self.fallback_threshold = fallback_threshold
        self.fallback_used = False

    def transform(self, texts: List[str]) -> np.ndarray:
        try:
            X = super().transform(texts)

            oov_report = self.get_oov_report()
            if oov_report and oov_report.get("total_oov", 0) > 0:
                total_tokens = sum(len(self.tokenizer.tokenize(t))
                                   for t in texts)
                oov_ratio = oov_report["total_oov"] / max(total_tokens, 1)

                if oov_ratio > self.fallback_threshold:
                    print(
                        f"\n  ⚠️  OOV ratio {oov_ratio:.1%} > threshold {self.fallback_threshold}")
                    print(f"     🔄 Fallback to character-level features")
                    X = self._fallback_transform(texts)
                    self.fallback_used = True

            return X

        except Exception as e:
            print(f"\n  ❌ Transform failed: {e}")
            print(f"     🔄 Emergency fallback: zero vectors")
            return np.zeros((len(texts), self.tfidf.max_features))

    def _fallback_transform(self, texts: List[str]) -> np.ndarray:
        """Fallback: character-level n-gram features"""
        from collections import Counter

        char_ngrams = []
        for text in texts:
            ngrams = []
            for n in [2, 3]:
                for i in range(len(text) - n + 1):
                    ngrams.append(text[i:i+n])
            char_ngrams.append(ngrams)

        all_ngrams = set([ng for doc in char_ngrams for ng in doc])
        ngram_to_idx = {ng: i for i, ng in enumerate(
            list(all_ngrams)[:self.tfidf.max_features])}

        X = np.zeros((len(texts), len(ngram_to_idx)))
        for i, ngrams in enumerate(char_ngrams):
            freq = Counter(ngrams)
            for ng, count in freq.items():
                if ng in ngram_to_idx:
                    X[i, ngram_to_idx[ng]] = count / len(ngrams)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X /= norms

        return X


# =============================================================================
# MAIN: Workshop 1 Demo (Updated v3)
# =============================================================================

def run_w1_demo():
    print("█" * 62)
    print("  WORKSHOP 1: Thai Legal Text Processing (v3 — Professional Edition)")
    print("  Thai IP Legal NLP Baseline with OOV + Coverage + Silver Standard")
    print("─" * 62)
    print("  Project : Physics-Governed IoT Framework for IP Law Enforcement")
    print("  Author  : น.ต. วีรยุทธ ครั่งกลาง")
    print("  Org     : กองฝึกอบรม กรมการสื่อสารทหาร กองบัญชาการกองทัพไทย")
    print("  Date    : 15 เมษายน 2569")
    print("  Code    : w1_thai_legal_nlp.py (v3)")
    print("  Status  : COMPLETED — Ready for W2, W3, W4, W17")
    print("█" * 62)

    tokenizer = ThaiLegalTokenizer()
    extractor = ThaiIPEntityExtractor(use_context_aware=True)

    print(f"\n  Corpus size: {len(THAI_IP_CORPUS)} documents")
    print(
        f"  Labeling method: {CORPUS_METADATA.get('labeling_method', 'unknown')}")

    # Tokenize + TF-IDF with OOV
    tokenized = tokenizer.tokenize_corpus(
        THAI_IP_CORPUS, remove_stopwords=True)
    tfidf = LegalTFIDFWithOOV(max_features=50, oov_strategy="unk_token")
    X = tfidf.fit_transform(tokenized)
    print(f"\n  TF-IDF matrix: {X.shape}")

    # OOV Report
    oov_report = tfidf.oov_handler.get_report()
    if oov_report["total_oov"] > 0:
        print(f"\n  📊 OOV Report:")
        print(f"     Total OOV tokens: {oov_report['total_oov']}")
        print(f"     Sample OOV: {oov_report['sample_oov'][:5]}")

    # Entity extraction with context-aware confidence
    print("\n  Entity extraction (first 3 docs) with context-aware confidence:")
    for doc in THAI_IP_CORPUS[:3]:
        entities = extractor.extract(doc)
        if entities:
            for e in entities[:2]:
                print(
                    f"    [{e.entity_type}] {e.value} (conf={e.confidence:.2f})")
                if e.context_signals:
                    print(f"        signals: {e.context_signals}")

    # Legal hierarchy demo
    print("\n  Legal Hierarchy Demo:")
    hierarchy = ThaiLegalHierarchy()
    test_entities = extractor.extract("ละเมิดสิทธิบัตรตามมาตรา 77")
    if test_entities:
        weight = hierarchy.compute_physics_gate_weight(test_entities)
        print(f"    Physics Gate Weight: {weight:.2f} / 10.0")

    # Top terms
    if X.shape[0] > 0:
        top = tfidf.get_top_terms(X[0], n=5)
        print(f"\n  Top terms (doc 0): {[t for t, _ in top]}")

    # W1Pipeline smoke test
    print("\n  W1Pipeline (v3) smoke-test:")
    w1 = W1Pipeline(max_features=32, evaluate_coverage=True)
    Xp = w1.fit_transform(THAI_IP_CORPUS)
    print(f"    fit_transform → {Xp.shape}  ✅")
    Xt = w1.transform(THAI_IP_CORPUS[:2])
    print(f"    transform(2)  → {Xt.shape}  ✅")

    # Professional quality report
    w1.print_professional_report(THAI_IP_CORPUS)

    # Ambiguity report
    w1.print_ambiguity_report(THAI_IP_CORPUS)


if __name__ == "__main__":
    run_w1_demo()

