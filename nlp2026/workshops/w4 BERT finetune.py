import numpy as np

MODEL_REGISTRY = {
    "mbert": {
        "hf_name" : "bert-base-multilingual-cased",
        "params"  : "110M",
        "thai_cov": "~1%",
        "note"    : "ใช้ได้หลายภาษา แต่ Thai coverage น้อย",
    },
    "xlmr": {
        "hf_name" : "xlm-roberta-base",
        "params"  : "270M",
        "thai_cov": "~5%",
        "note"    : "RoBERTa architecture, Thai ดีกว่า mBERT",
    },
    "wangchanberta": {
        "hf_name" : "airesearch/wangchanberta-base-att-spm-uncased",
        "params"  : "110M",
        "thai_cov": "100%",
        "note"    : "ดีที่สุดสำหรับ Thai legal text",
    },
}

def show_model_comparison():
    print("=" * 60)
    print("  4.1 เลือก Pretrained Model สำหรับ Thai Legal Text")
    print("=" * 60)
    print(f"  {'Model':<16} {'Params':<8} {'Thai%':<8} หมายเหตุ")
    print(f"  {'─'*56}")
    for name, m in MODEL_REGISTRY.items():
        print(f"  {name:<16} {m['params']:<8} {m['thai_cov']:<8} {m['note']}")
    print(f"\n  → เลือก WangchanBERTa เพราะ pre-train บน Thai Wikipedia + CCNet")

show_model_comparison()

def get_device():
    """ตรวจสอบ GPU อัตโนมัติ: CUDA → MPS → CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print(f"  ✅ Apple Silicon (MPS)")
            return "mps"
        else:
            print(f"  ⚠️  ไม่มี GPU → ใช้ CPU (ช้ากว่า 10-20x)")
            return "cpu"
    except ImportError:
        print("  ℹ️  ไม่มี PyTorch → ใช้ Mock mode")
        return "mock"

DEVICE = get_device()
print(f"\n  Device ที่ใช้: {DEVICE}")


class MockTokenize:
    """จำลอง BERT Tokenizer"""
    LEGAL_TERMS = ["ภูมิปัญญาท้องถิ่น", "สิทธิการประดิษฐ์", "อนุสิทธิบัตร", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ"]

    def encode(self, texts, max_length=128):
        """แปลงข้อความ -> input_ids + attention_mask
           input_ids : [CLS] + token_ids + [SEP] + [PAD]...
           attention_mask: 1 = real token, 0 = padding
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attention_mask = [], []  # ✅ reset ก่อน loop
        for text in texts:
            # [CLS=1] ... [SEP=2]
            ids = [1] + [ord(c) % 5000 + 100 for c in text[:max_length - 2]] + [2]  # ✅ แก้ typo และ bracket
            pad_len = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids  = ids + [0] * pad_len
            input_ids.append(ids)           # ✅ เพิ่ม append ที่หายไป
            attention_mask.append(mask)     # ✅ เพิ่ม append ที่หายไป

        return {
            "input_ids"      : np.array(input_ids,      dtype=np.int32),
            "attention_mask" : np.array(attention_mask, dtype=np.int32),
        }

    def show(self, text, max_length=32): # ← เพิ่ม max_length ให้ยาวพอ
        enc = self.encode([text], max_length)
        ids =enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        real_len = mask.sum()
        print(f"\n ข้อความ : '{text}'")
        print(f" input_ids : {ids.tolist()}") # ← แสดงทั้ง array รวม 0
        print(f" mask : {mask.tolist()}")
        print(f" real tokens: {real_len} PAD: {max_length - real_len}")
        print(f"\n แยกส่วน:")
        print(f" [CLS] = {ids[0]}")
        print(f" tokens= {ids[1:real_len-1].tolist()}")
        print(f" [SEP] = {ids[real_len-1]}")
        print(f" [PAD] = {ids[real_len:].tolist()}") # ← แสดง 0s


# ── Vocabulary Expansion Demo ──────────────────────────────────
def vocab_expansion_demo():
    tok = MockTokenize()
    print("=" * 58)
    print(" STEP 1: ก่อน expansion — คำใหม่ถูกตัดเป็นชิ้น")
    print("=" * 58)
    tok.show("สิทธิบัตรการประดิษฐ์")
    print("\n" + "=" * 58)
    print(" STEP 2: เพิ่มคำใหม่เข้า vocab (Vocabulary Expansion)")
    print("=" * 58) # vocab ปกติ + เพิ่มคำกฎหมาย
    base_vocab_size = 5000
    new_vocab = {term:base_vocab_size + i
             for i, term in enumerate(MockTokenize.LEGAL_TERMS)}
    print(f"\n vocab เดิม : {base_vocab_size} คำ")
    print(f" เพิ่มคำใหม่: {len(new_vocab)} คำ")
    print(f" vocab ใหม่ : {base_vocab_size +len(new_vocab)} คำ\n")
    for term, idx in new_vocab.items():
        print(f" '{term}' → id {idx} (token ใหม่ weight = random ❗)")
    print("\n" + "=" * 58)
    print(" STEP 3: ทำไมต้อง Warm-up ก่อน Fine-tune")
    print("=" * 58)
    print(""" ปัญหา: คำเดิม → weights ผ่าน pre-train มาแล้ว (มีความหมาย) คำใหม่ → weights = random ❗ (ยังไม่มีความหมาย) ถ้า fine-tune ทุก layer พร้อมกันเลย: gradient จากคำใหม่ (random) จะรบกวน weights เดิม → โมเดลลืมสิ่งที่เรียนมา = Catastrophic Forgetting ❌ วิธีแก้ — Warm-up 3 ขั้นตอน: ขั้น 1 │ Freeze ทุก layer ยกเว้น embedding │ train แค่ embedding 2-3 epochs│ → คำใหม่เริ่มมีความหมาย ขั้น 2 │ Unfreeze ทุก layer │ train ด้วย LR ต่ำ (2e-5) │ → ปรับ weights ทั้งหมดพร้อมกันขั้น 3 │ Fine-tune จนกว่า val_loss นิ่ง │ → โมเดลพร้อมใช้งาน ✅ """)
    print("=" * 58)
    print(" STEP 4: จำลอง embedding weight ก่อน/หลัง warm-up")
    print("=" * 58)
    np.random.seed(42)
    d_model = 8 # embedding ของคำเดิม (pretrained) — มีค่าชัดเจน
    old_emb = np.array([0.82, -0.34, 0.56, 0.91, -0.12, 0.67, -0.45, 0.23]) # embedding ของคำใหม่ก่อน warm-up — random
    new_before = np.random.randn(d_model) * 0.02 # หลัง warm-up — เริ่มมีทิศทาง (จำลอง)
    new_after = old_emb *0.6 + np.random.randn(d_model) * 0.1
    print(f"\n คำเดิม 'ละเมิด' : {old_emb.round(2)}")
    print(f" คำใหม่ ก่อน warm-up : {new_before.round(2)} ← random")
    print(f" คำใหม่ หลัง warm-up : {new_after.round(2)} ← มีทิศทางแล้ว")

    # cosine similarity
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n cosine similarity กับ 'ละเมิด':")
    print(f" ก่อน warm-up : {cosine(old_emb, new_before):.3f} (ไม่เกี่ยวกัน)")
    print(f" หลัง warm-up : {cosine(old_emb, new_after):.3f} (ใกล้เคียงกัน)")
   
    # ── รัน ──────────────────────────────────────────────────────
tok = MockTokenize()
tok.show("ผู้ต้องหาละเมิดสิทธิบัตร")
print()
vocab_expansion_demo()
 
# 4. BERT Encoder
import numpy as np

# --- 1. Mock Tokenizer (สำหรับจำลองการแปลงข้อความ) ---
class MockTokenizer:
    def encode(self, texts, max_length=32):
        batch_size = len(texts)
        # จำลองการสร้าง input_ids และ attention_mask
        input_ids = np.random.randint(100, 10000, size=(batch_size, max_length))
        attention_mask = np.ones((batch_size, max_length))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# --- 2. Mock BERT Encoder (จำลอง Transformer 12 Layers) ---
class MockBERTEncoder:
    """
    จำลอง BERT 12-layer encoder
    Output: h_cls shape (batch, 768)
    """
    def __init__(self, hidden_size=768, seed=42):
        self.hidden_size = hidden_size
        self.rng = np.random.RandomState(seed)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        # จำลองค่า h_cls (Hidden state ของ [CLS] token)
        h_cls = self.rng.randn(batch, self.hidden_size).astype(np.float32) * 0.1
        return h_cls

# --- 3. Classification Head (The Judge) ---
class ClassificationHead:
    """
    Linear layer สำหรับจำแนก class (PATENT, COPYRIGHT, NONE)
    """
    def __init__(self, hidden_size=768, n_classes=3, dropout=0.1, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier Initialization สำหรับน้ำหนัก
        s = np.sqrt(2.0 / (hidden_size + n_classes))
        self.W = rng.randn(n_classes, hidden_size).astype(np.float32) * s
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.dropout = dropout

    def forward(self, h_cls, training=False):
        # Dropout: ใช้เฉพาะช่วง training เพื่อป้องกัน Overfitting
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)

        # คำนวณ Logits: (batch, 768) @ (768, 3) = (batch, 3)
        logits = h_cls @ self.W.T + self.b
        
        # Softmax: แปลงเป็นความน่าจะเป็น
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

# --- 4. BERT For Classification (Main Class) ---
class BERTForClassification:
    CLASS_NAMES = ["ละเมิด_สิทธิบัตร", "ละเมิด_ลิขสิทธิ์", "ไม่ละเมิด"]

    def __init__(self, seed=42):
        self.encoder = MockBERTEncoder(seed=seed)
        self.head = ClassificationHead(seed=seed)
        self.tokenizer = MockTokenizer()

    def predict_proba(self, input_ids, attention_mask):
        h_cls = self.encoder.forward(input_ids, attention_mask)
        return self.head.forward(h_cls)

    def predict(self, input_ids, attention_mask):
        return np.argmax(self.predict_proba(input_ids, attention_mask), axis=1)

    def show_prediction(self, texts):
        # ขั้นตอน Tokenization
        enc = self.tokenizer.encode(texts, max_length=32)
        
        # ขั้นตอนการทำนาย
        prob = self.predict_proba(enc["input_ids"], enc["attention_mask"])
        pred = np.argmax(prob, axis=1)

        # แสดงผลลัพธ์
        print(f"\n{'ข้อความ':<45} {'การทำนาย':<20} {'ความมั่นใจ'}")
        print(f"{'─'*75}")
        for text, p, pr in zip(texts, pred, prob):
            confidence = pr[p] * 100
            print(f"{text[:43]:<45} {self.CLASS_NAMES[p]:<20} {confidence:.1f}%")

# --- 5. Execution Block ---
if __name__ == "__main__":
    # เริ่มต้นโมเดล
    model = BERTForClassification(seed=42)
    
    # ตัวอย่างข้อมูลทดสอบ (Legal Context)
    test_cases = [
        "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตร",
        "จำเลยทำซ้ำงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
        "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรถูกต้องแล้ว",
        "มีการดัดแปลงโปรแกรมคอมพิวเตอร์เพื่อการค้า"
    ]
    
    model.show_prediction(test_cases)
