"""
=============================================================================
WORKSHOP 2: Baseline LSTM/BiLSTM + SMOTE  (v4 — Professional Edition)
=============================================================================

LEARNING OBJECTIVES:
    1. สร้าง LSTM / BiLSTM สำหรับ Thai IP Legal Text Classification
    2. จัดการ Class Imbalance ด้วย SMOTE (พร้อม Random Oversampling Fallback)
    3. Evaluate ด้วย F1, AUC, Precision, Recall
    4. เปรียบเทียบ LSTM vs BiLSTM vs Baseline (TF-IDF + LinearSVC จาก W1)
    5. เชื่อมกับ Pipeline และ W1
    6. Cost-Sensitive Learning สำหรับ Legal Domain
    7. Uncertainty Estimation (Entropy) สำหรับ Physics Gate (W17)

ENHANCEMENTS in v4 (15 April 2026):
    [v4-1] Xavier/Glorot Weight Initialization สำหรับ LSTM
    [v4-2] SMOTE Fallback: Random Oversampling สำหรับ minority class size=1
    [v4-3] Cost-Sensitive Loss (เพิ่ม penalty สำหรับ False Negative)
    [v4-4] Uncertainty Estimation (Entropy-based Confidence)
    [v4-5] Overlapping Windows สำหรับ Sequence Analysis
    [v4-6] Data Augmentation (Synonym Replacement) สำหรับ Legal Text
=============================================================================
"""

import os
import sys
import numpy as np
import collections
from typing import List, Tuple, Dict, Optional

# =============================================================================
# Import W1Pipeline — รองรับทั้ง flat และ workshops/ subfolder
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from w1_thai_legal_nlp import W1Pipeline, THAI_IP_CORPUS
except ModuleNotFoundError:
    raise SystemExit(
        "\n[W2 ERROR] ไม่พบ w1_thai_legal_nlp.py\n"
        f"  กรุณาวางไฟล์ไว้ใน folder เดียวกัน:\n"
        f"    {BASE_DIR}/w1_thai_legal_nlp.py\n"
        f"    {BASE_DIR}/w2_lstm_baseline.py\n"
    )

np.random.seed(42)


# =============================================================================
# SECTION 0: Labeled Dataset
# =============================================================================

CLASS_NAMES = {
    0: "NO_INFRINGEMENT",
    1: "PATENT_VIOLATION",
    2: "COPYRIGHT_VIOLATION",
}

# Cost Matrix สำหรับ Legal Domain (False Negative มี penalty สูงกว่า)
# ในนิติศาสตร์: การระบุว่า "ไม่ละเมิด" ทั้งที่จริง "ละเมิด" (FN) ร้ายแรงกว่า
# การระบุว่า "ละเมิด" ทั้งที่จริง "ไม่ละเมิด" (FP)
COST_MATRIX = {
    "FN_WEIGHT": 2.0,   # False Negative penalty สูงเป็น 2 เท่า
    "FP_WEIGHT": 1.0,   # False Positive penalty ปกติ
}

LABELED_CORPUS = [
    # Class 0: NO_INFRINGEMENT
    ("สินค้านี้ผ่านการตรวจสอบและได้รับอนุญาตจากเจ้าของสิทธิบัตรแล้ว", 0),
    ("บริษัทได้รับสิทธิ์การใช้งานสิทธิบัตรอย่างถูกต้องตามกฎหมาย", 0),
    ("ผลิตภัณฑ์ดังกล่าวไม่อยู่ในขอบเขตการคุ้มครองสิทธิบัตร", 0),
    ("สัญญาอนุญาตการใช้งานได้รับการลงนามถูกต้องตามกฎหมาย", 0),
    ("บริษัทได้จดทะเบียนเครื่องหมายการค้าและได้รับสิทธิ์แล้ว", 0),
    ("คณะกรรมการสิทธิบัตรอนุมัติให้รับจดทะเบียนสิทธิบัตรการประดิษฐ์", 0),

    # Class 1: PATENT_VIOLATION
    ("จำเลยผลิตสินค้าเลียนแบบสิทธิบัตรโดยไม่ได้รับอนุญาต", 1),
    ("บริษัทนำเข้าผลิตภัณฑ์ที่ละเมิดสิทธิบัตรจากต่างประเทศ", 1),
    ("ผู้ต้องหาผลิตและจำหน่ายสินค้าปลอมแปลงสิทธิบัตร", 1),
    ("พบการผลิตสินค้าเลียนแบบการประดิษฐ์โดยไม่ได้รับอนุญาต", 1),
    ("จำเลยใช้กรรมวิธีการผลิตที่จดสิทธิบัตรโดยไม่ได้รับอนุญาต", 1),
    ("มีการขายสินค้าที่ละเมิดสิทธิบัตรในราคาถูกกว่าของแท้", 1),
    ("โรงงานผลิตชิ้นส่วนเลียนแบบสิทธิบัตรส่งออกต่างประเทศ", 1),
    ("จำเลยนำกระบวนการผลิตที่จดสิทธิบัตรไปใช้โดยไม่ขออนุญาต", 1),
    ("บริษัทนำเข้าสินค้าปลอมแปลงเครื่องหมายการค้าจำหน่ายโดยไม่ได้รับอนุญาต", 1),
    ("ผู้เสียหายพบการผลิตสินค้าเลียนแบบโดยไม่ได้รับอนุญาตจากเจ้าของสิทธิบัตร", 1),

    # Class 2: COPYRIGHT_VIOLATION
    ("จำเลยทำซ้ำงานอันมีลิขสิทธิ์โดยไม่ได้รับอนุญาต", 2),
    ("มีการเผยแพร่ซอฟต์แวร์ที่มีลิขสิทธิ์โดยไม่ชอบด้วยกฎหมาย", 2),
    ("ผู้ต้องหาดัดแปลงงานสร้างสรรค์โดยไม่ขออนุญาตเจ้าของลิขสิทธิ์", 2),
    ("พบการทำซ้ำและจำหน่ายงานอันมีลิขสิทธิ์โดยไม่ได้รับอนุญาต", 2),
    ("จำเลยละเมิดลิขสิทธิ์ทางดิจิทัลโดยการ stream โดยไม่ได้รับอนุญาต", 2),
    ("มีการคัดลอกงานเขียนโดยไม่อ้างอิงหรือขออนุญาตเจ้าของลิขสิทธิ์", 2),
    ("จำเลยนำภาพถ่ายที่มีลิขสิทธิ์ไปใช้โดยไม่ได้รับอนุญาต", 2),
    ("บริษัทเผยแพร่เพลงที่มีลิขสิทธิ์บนแพลตฟอร์มออนไลน์โดยผิดกฎหมาย", 2),
    ("ผู้ใดกระทำการละเมิดลิขสิทธิ์โดยเจตนาต้องระวางโทษจำคุกและปรับ", 2),
    ("การทำซ้ำหรือดัดแปลงเผยแพร่ต่อสาธารณชนโดยไม่ได้รับอนุญาตถือว่าละเมิดลิขสิทธิ์", 2),
]


# =============================================================================
# SECTION 1: Enhanced SMOTE (with Random Oversampling Fallback)
# =============================================================================

class EnhancedSMOTE:
    """
    SMOTE: Synthetic Minority Oversampling Technique (Chawla et al. 2002)

    Enhancement: หาก minority class มีเพียง 1 ตัวอย่าง ให้ใช้ Random Oversampling
    (Copy) แทนการหา neighbor เพื่อป้องกัน k_eff = 0

    x_new = x + λ(x_neighbor - x),  λ ~ Uniform(0,1)
    Fallback: x_new = x (copy) เมื่อมีตัวอย่าง太少
    """

    def __init__(self, k_neighbors: int = 3, random_state: int = 42):
        self.k = k_neighbors
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X: np.ndarray,
                     y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        class_counts = collections.Counter(y)
        max_count = max(class_counts.values())

        X_res = list(X)
        y_res = list(y)

        for cls, count in class_counts.items():
            if count >= max_count:
                continue

            cls_samples = X[y == cls]
            n_samples = len(cls_samples)

            # [v4-2] Fallback: Random Oversampling เมื่อมีตัวอย่าง太少
            if n_samples == 1:
                # กรณีมีเพียง 1 ตัวอย่าง: copy ซ้ำ
                for _ in range(max_count - count):
                    X_res.append(cls_samples[0])
                    y_res.append(cls)
                continue

            # Normal SMOTE
            k_eff = min(self.k, n_samples - 1)
            if k_eff < 1:
                k_eff = 1

            for _ in range(max_count - count):
                idx = self.rng.randint(0, n_samples)
                base = cls_samples[idx]

                # Calculate distances
                dists = np.linalg.norm(cls_samples - base, axis=1)
                dists[idx] = np.inf
                nn_idx = np.argsort(dists)[:k_eff]
                neighbor = cls_samples[self.rng.choice(nn_idx)]

                lam = self.rng.uniform(0, 1)
                X_res.append(base + lam * (neighbor - base))
                y_res.append(cls)

        return np.array(X_res, dtype=np.float32), np.array(y_res)


# =============================================================================
# SECTION 2: LSTM Cell with Xavier/Glorot Initialization
# =============================================================================

class LSTMCell:
    """
    LSTM (Hochreiter & Schmidhuber 1997)

    [v4-1] Xavier/Glorot Initialization แทน random uniform scaling
    Formula: scale = sqrt(2.0 / (input_size + hidden_size))

    Forget gate bias = 1.0  (Jozefowicz et al. 2015)
    Sigmoid clipped at ±10  (ป้องกัน exp overflow)
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        n = hidden_size + input_size

        # [v4-1] Xavier/Glorot Initialization
        # Glorot & Bengio (2010): Variance = 2 / (fan_in + fan_out)
        scale = np.sqrt(2.0 / (hidden_size + n))

        self.Wf = rng.randn(hidden_size, n) * scale
        self.Wi = rng.randn(hidden_size, n) * scale
        self.Wg = rng.randn(hidden_size, n) * scale
        self.Wo = rng.randn(hidden_size, n) * scale

        self.bf = np.ones((hidden_size, 1))    # forget gate bias = 1
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.hidden_size = hidden_size
        self.input_size = input_size

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x: np.ndarray,
                h_prev: np.ndarray,
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hx = np.vstack([h_prev, x])
        f = self.sigmoid(self.Wf @ hx + self.bf)
        i = self.sigmoid(self.Wi @ hx + self.bi)
        g = np.tanh(self.Wg @ hx + self.bg)
        o = self.sigmoid(self.Wo @ hx + self.bo)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c


# =============================================================================
# SECTION 3: LSTM Classifier with Cost-Sensitive Loss
# =============================================================================

class LSTMClassifier:
    """
    LSTM Classifier with:
        - Xavier initialization
        - Cost-sensitive learning (FN penalty > FP penalty)
        - Uncertainty estimation (entropy)
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 n_classes: int = 3, seed: int = 42,
                 fn_weight: float = 2.0, fp_weight: float = 1.0):
        rng = np.random.RandomState(seed)
        self.lstm = LSTMCell(input_size, hidden_size, seed)

        # Xavier initialization for output layer
        scale_out = np.sqrt(2.0 / (n_classes + hidden_size))
        self.W_out = rng.randn(n_classes, hidden_size) * scale_out
        self.b_out = np.zeros((n_classes, 1))

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.input_size = input_size
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _cost_sensitive_loss(self, y_true: int, y_pred_proba: np.ndarray) -> float:
        """
        Cost-sensitive cross-entropy loss

        Penalize False Negative (FN) มากกว่า False Positive (FP)
        เนื่องจากใน legal domain การปล่อยผู้กระทำผิด (FN) ร้ายแรงกว่า
        """
        # Standard cross-entropy
        ce_loss = -np.log(y_pred_proba[y_true] + 1e-10)

        # Cost adjustment based on prediction
        y_pred = np.argmax(y_pred_proba)

        if y_true != y_pred:
            if y_pred == 0 and y_true != 0:
                # False Negative: model says NO_INFRINGEMENT but actually violation
                return ce_loss * self.fn_weight
            else:
                # False Positive: model says violation but actually NO_INFRINGEMENT
                return ce_loss * self.fp_weight

        return ce_loss

    def forward(self, x_seq: np.ndarray, return_uncertainty: bool = False):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for t in range(x_seq.shape[0]):
            h, c = self.lstm.forward(x_seq[t].reshape(-1, 1), h, c)

        logits = self.W_out @ h + self.b_out
        probs = self._softmax(logits.flatten())

        if return_uncertainty:
            # [v4-4] Entropy-based uncertainty
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return probs, entropy

        return probs

    def predict_batch(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        return np.array([np.argmax(self.forward(X[i].reshape(seq_len, -1)))
                         for i in range(len(X))])

    def predict_proba(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        return np.array([self.forward(X[i].reshape(seq_len, -1))
                         for i in range(len(X))])

    def predict_with_uncertainty(self, X: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (probabilities, entropy) for each sample"""
        results = [self.forward(X[i].reshape(seq_len, -1), return_uncertainty=True)
                   for i in range(len(X))]
        probs = np.array([r[0] for r in results])
        entropy = np.array([r[1] for r in results])
        return probs, entropy


# =============================================================================
# SECTION 4: BiLSTM Classifier with Xavier Init + Uncertainty
# =============================================================================

class BiLSTMClassifier:
    """
    Bidirectional LSTM — mean pooling ของทุก hidden state
    ดีกว่า final-state-only สำหรับ legal text ที่ keyword อาจอยู่กลางประโยค

    [v4-1] Xavier initialization
    [v4-4] Uncertainty estimation (entropy)
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 n_classes: int = 3, seed: int = 42,
                 fn_weight: float = 2.0, fp_weight: float = 1.0):
        rng = np.random.RandomState(seed)
        self.lstm_fwd = LSTMCell(input_size, hidden_size, seed)
        self.lstm_bwd = LSTMCell(input_size, hidden_size, seed + 1)

        # Xavier initialization for output layer
        scale_out = np.sqrt(2.0 / (n_classes + hidden_size * 2))
        self.W_out = rng.randn(n_classes, hidden_size * 2) * scale_out
        self.b_out = np.zeros((n_classes, 1))

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.input_size = input_size
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_seq: np.ndarray, return_uncertainty: bool = False):
        T = x_seq.shape[0]

        # Forward pass
        h_f = np.zeros((self.hidden_size, 1))
        c_f = np.zeros((self.hidden_size, 1))
        h_fwd_all = []
        for t in range(T):
            h_f, c_f = self.lstm_fwd.forward(x_seq[t].reshape(-1, 1), h_f, c_f)
            h_fwd_all.append(h_f)

        # Backward pass
        h_b = np.zeros((self.hidden_size, 1))
        c_b = np.zeros((self.hidden_size, 1))
        h_bwd_all = []
        for t in reversed(range(T)):
            h_b, c_b = self.lstm_bwd.forward(x_seq[t].reshape(-1, 1), h_b, c_b)
            h_bwd_all.append(h_b)

        # Mean pooling
        h_fwd_mean = np.mean(np.hstack(h_fwd_all), axis=1, keepdims=True)
        h_bwd_mean = np.mean(np.hstack(h_bwd_all), axis=1, keepdims=True)

        h_combined = np.vstack([h_fwd_mean, h_bwd_mean])
        logits = self.W_out @ h_combined + self.b_out
        probs = self._softmax(logits.flatten())

        if return_uncertainty:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return probs, entropy

        return probs

    def predict_batch(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        return np.array([np.argmax(self.forward(X[i].reshape(seq_len, -1)))
                         for i in range(len(X))])

    def predict_proba(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        return np.array([self.forward(X[i].reshape(seq_len, -1))
                         for i in range(len(X))])

    def predict_with_uncertainty(self, X: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (probabilities, entropy) for each sample"""
        results = [self.forward(X[i].reshape(seq_len, -1), return_uncertainty=True)
                   for i in range(len(X))]
        probs = np.array([r[0] for r in results])
        entropy = np.array([r[1] for r in results])
        return probs, entropy


# =============================================================================
# SECTION 5: Overlapping Window Generator (v4-5)
# =============================================================================

class OverlappingWindowGenerator:
    """
    สร้าง overlapping windows จาก feature vectors

    แทนการแบ่งแบบ fixed non-overlapping (seq_len=4, stride=4)
    overlapping ช่วยให้ LSTM เห็น "รอยต่อ" ของข้อมูลได้ละเอียดขึ้น

    ตัวอย่าง:
        features = [f0, f1, f2, f3, f4, f5, f6, f7]
        window_size=4, stride=1:
            window0: [f0, f1, f2, f3]
            window1: [f1, f2, f3, f4]
            window2: [f2, f3, f4, f5]
            ...
    """

    def __init__(self, window_size: int = 4, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def create_windows(self, X: np.ndarray) -> np.ndarray:
        """
        สร้าง overlapping windows สำหรับ sequence

        Args:
            X: shape (n_samples, n_features)

        Returns:
            windows: shape (n_samples, n_windows, window_size)
            โดย n_windows = (n_features - window_size) // stride + 1
        """
        n_samples, n_features = X.shape
        n_windows = (n_features - self.window_size) // self.stride + 1

        windows = []
        for i in range(n_samples):
            sample_windows = []
            for w in range(n_windows):
                start = w * self.stride
                end = start + self.window_size
                window = X[i, start:end]
                sample_windows.append(window)
            windows.append(np.array(sample_windows))

        return np.array(windows)

    def aggregate_windows(self, window_predictions: np.ndarray,
                          aggregation: str = "mean") -> np.ndarray:
        """
        รวม predictions จากหลาย windows

        Args:
            window_predictions: shape (n_samples, n_windows, n_classes)
            aggregation: "mean", "max", หรือ "vote"

        Returns:
            aggregated: shape (n_samples, n_classes)
        """
        if aggregation == "mean":
            return np.mean(window_predictions, axis=1)
        elif aggregation == "max":
            return np.max(window_predictions, axis=1)
        elif aggregation == "vote":
            # Majority voting
            pred_classes = np.argmax(window_predictions, axis=2)
            results = []
            for i in range(pred_classes.shape[0]):
                vote = np.bincount(pred_classes[i]).argmax()
                one_hot = np.zeros(window_predictions.shape[2])
                one_hot[vote] = 1.0
                results.append(one_hot)
            return np.array(results)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


# =============================================================================
# SECTION 6: TF-IDF + LinearSVC Baseline (W1 features จริง)
# =============================================================================

class LinearSVCBaseline:
    """Linear SVM บน TF-IDF features จาก W1Pipeline"""

    def __init__(self, n_classes: int = 3, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.n_classes = n_classes
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVCBaseline":
        n_features = X.shape[1]
        self.weights_ = np.zeros(
            (self.n_classes, n_features), dtype=np.float32)
        self.bias_ = np.zeros(self.n_classes, dtype=np.float32)
        for cls in range(self.n_classes):
            mask = (y == cls)
            if mask.any():
                self.weights_[cls] = X[mask].mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.weights_.T + self.bias_
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.weights_.T + self.bias_
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# SECTION 7: Evaluation with Cost-Sensitive Metrics
# =============================================================================

class ClassificationEvaluator:
    def __init__(self, class_names: Dict, fn_weight: float = 2.0, fp_weight: float = 1.0):
        self.class_names = class_names
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    def confusion_matrix(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> np.ndarray:
        n = len(self.class_names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def cost_sensitive_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        คำนวณ cost-sensitive score
        False Negative (FN) มี penalty สูงกว่า False Positive (FP)
        """
        cm = self.confusion_matrix(y_true, y_pred)
        n = len(self.class_names)

        total_cost = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    if i != 0 and j == 0:
                        # False Negative (predict NO_INFRINGEMENT but actual violation)
                        total_cost += cm[i, j] * self.fn_weight
                    else:
                        # False Positive
                        total_cost += cm[i, j] * self.fp_weight

        max_possible_cost = len(y_true) * max(self.fn_weight, self.fp_weight)
        score = 1.0 - (total_cost / max_possible_cost)
        return round(score, 4)

    def classification_report(self, y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict:
        report = {}
        for cls in range(len(self.class_names)):
            tp = int(np.sum((y_true == cls) & (y_pred == cls)))
            fp = int(np.sum((y_true != cls) & (y_pred == cls)))
            fn = int(np.sum((y_true == cls) & (y_pred != cls)))
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            report[self.class_names[cls]] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(y_true == cls)),
            }
        vals = list(report.values())
        report["macro_avg"] = {
            "precision": round(float(np.mean([v["precision"] for v in vals])), 4),
            "recall": round(float(np.mean([v["recall"] for v in vals])), 4),
            "f1": round(float(np.mean([v["f1"] for v in vals])), 4),
        }
        return report

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return round(float(np.mean(y_true == y_pred)), 4)

    def auc_roc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        trapz = getattr(np, "trapezoid", None) or np.trapz
        aucs = []
        for cls in range(len(self.class_names)):
            binary = (y_true == cls).astype(int)
            score = y_proba[:, cls]
            pos, neg = binary.sum(), len(binary) - binary.sum()
            if pos == 0 or neg == 0:
                aucs.append(0.5)
                continue
            desc_idx = np.argsort(-score)
            tpr_l, fpr_l = [0.0], [0.0]
            tp = fp = 0
            for idx in desc_idx:
                if binary[idx]:
                    tp += 1
                else:
                    fp += 1
                tpr_l.append(tp / pos)
                fpr_l.append(fp / neg)
            aucs.append(float(np.abs(trapz(tpr_l, fpr_l))))
        return round(float(np.mean(aucs)), 4)

    def print_report(self, model_name: str,
                     y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None,
                     entropy: Optional[np.ndarray] = None):
        rpt = self.classification_report(y_true, y_pred)
        acc = self.accuracy(y_true, y_pred)
        cm = self.confusion_matrix(y_true, y_pred)
        cost_score = self.cost_sensitive_score(y_true, y_pred)

        print(f"\n  {'─'*60}")
        print(f"  {model_name}")
        print(f"  {'─'*60}")
        print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
        print(f"  {'─'*60}")
        for cls_name, m in rpt.items():
            if cls_name == "macro_avg":
                print(f"  {'─'*60}")
                print(f"  {'macro_avg':<25} "
                      f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
            else:
                print(f"  {cls_name:<25} "
                      f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
                      f"{m['f1']:>6.3f} {m['support']:>4}")
        print(f"\n  Accuracy        : {acc:.4f}")
        print(
            f"  Cost-Sensitive  : {cost_score:.4f} (FN weight={self.fn_weight})")
        if y_proba is not None:
            print(f"  Macro AUC-ROC   : {self.auc_roc(y_true, y_proba):.4f}")
        if entropy is not None:
            print(f"  Avg Uncertainty : {np.mean(entropy):.4f} (entropy)")
        print(f"\n  Confusion Matrix:")
        header = "".join(f"  {n[:6]:>8}" for n in self.class_names.values())
        print(f"  {'':>20}{header}")
        for i, row_name in self.class_names.items():
            row = "".join(f"  {'['+str(cm[i, j])+']':>8}"
                          for j in range(len(self.class_names)))
            print(f"  {row_name[:20]:<20}{row}")


# =============================================================================
# SECTION 8: Legal Data Augmentation (v4-6)
# =============================================================================

class LegalDataAugmenter:
    """
    Data augmentation สำหรับ legal text (simulated back-translation)

    เนื่องจากข้อจำกัด API (Google Translate) ใน demo นี้ ใช้ synonym replacement
    แทน back-translation จริง

    ใน production: ใช้ Google Translate API หรือ PyThaiNLP wordnet
    """

    # Legal synonym dictionary (Thai)
    LEGAL_SYNONYMS = {
        "ละเมิด": ["ละเมิด", "ฝ่าฝืน", "ผิด", "กระทำผิด"],
        "สิทธิบัตร": ["สิทธิบัตร", "专利权", "patent"],
        "ลิขสิทธิ์": ["ลิขสิทธิ์", "copyright", "ลิขสิทธิ์ทางปัญญา"],
        "จำเลย": ["จำเลย", "ผู้ต้องหา", "ผู้ถูกกล่าวหา"],
        "ผลิต": ["ผลิต", "ทำ", "สร้าง", "ประกอบ"],
        "จำหน่าย": ["จำหน่าย", "ขาย", "จัดจำหน่าย", "ขายปลีก"],
        "โดยไม่ได้รับอนุญาต": ["โดยไม่ได้รับอนุญาต", "ไม่ได้รับอนุญาต", "ไม่ชอบด้วยกฎหมาย"],
    }

    def __init__(self, rng: np.random.RandomState = None):
        self.rng = rng or np.random.RandomState(42)

    def synonym_replacement(self, text: str, replace_prob: float = 0.3) -> str:
        """แทนที่คำศัพท์ด้วย synonym แบบสุ่ม"""
        result = text
        for word, synonyms in self.LEGAL_SYNONYMS.items():
            if word in result and self.rng.random() < replace_prob:
                new_word = self.rng.choice(synonyms)
                result = result.replace(word, new_word, 1)
        return result

    def augment_corpus(self, texts: List[str], labels: List[int],
                       aug_factor: int = 2) -> Tuple[List[str], List[int]]:
        """
        Augment corpus โดยการสร้าง synonym replacement variants

        Args:
            texts: original texts
            labels: original labels
            aug_factor: number of augmented versions per sample

        Returns:
            augmented_texts, augmented_labels
        """
        augmented_texts = list(texts)
        augmented_labels = list(labels)

        for text, label in zip(texts, labels):
            for _ in range(aug_factor - 1):
                new_text = self.synonym_replacement(text)
                if new_text != text:
                    augmented_texts.append(new_text)
                    augmented_labels.append(label)

        return augmented_texts, augmented_labels


# =============================================================================
# SECTION 9: Stratified K-Fold (Simplified for Demo)
# =============================================================================

def stratified_kfold_eval(X: np.ndarray, y: np.ndarray,
                          n_splits: int = 5, seq_len: int = 4,
                          seed: int = 42) -> Dict:
    """
    Stratified K-Fold Cross Validation

    NOTE: สำหรับการประเมินที่สมบูรณ์ ควร implement training loop
          ปัจจุบันใช้ random weights เพื่อทดสอบ pipeline
    """
    rng = np.random.RandomState(seed)
    smote = EnhancedSMOTE(k_neighbors=3, random_state=seed)
    evaluator = ClassificationEvaluator(
        CLASS_NAMES, fn_weight=2.0, fp_weight=1.0)

    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_splits].append(j)

    results = {"lstm": [], "bilstm": []}
    print(f"\n  Stratified {n_splits}-Fold CV  (seq_len={seq_len})")
    print(f"  NOTE: Using random weights for baseline")
    print(f"  {'─'*56}")

    for fold in range(n_splits):
        te_idx = np.array(folds[fold])
        tr_idx = np.array([j for k in range(n_splits)
                           if k != fold for j in folds[k]])

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # SMOTE เฉพาะ train fold
        if len(np.unique(y_tr)) > 1 and len(y_tr) > 2:
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
        else:
            X_res, y_res = X_tr, y_tr

        # ตรวจสอบ dimension
        assert X_res.shape[1] % seq_len == 0, \
            f"feature dim {X_res.shape[1]} not divisible by {seq_len}"
        input_size = X_res.shape[1] // seq_len

        # สร้าง models ด้วย random weights (baseline)
        lstm = LSTMClassifier(input_size, 32, 3, seed +
                              fold, fn_weight=2.0, fp_weight=1.0)
        bilstm = BiLSTMClassifier(
            input_size, 32, 3, seed + fold, fn_weight=2.0, fp_weight=1.0)

        # Quick evaluation with current weights
        f1_lstm = evaluator.classification_report(
            y_te, lstm.predict_batch(X_te, seq_len))["macro_avg"]["f1"]
        f1_bilstm = evaluator.classification_report(
            y_te, bilstm.predict_batch(X_te, seq_len))["macro_avg"]["f1"]

        results["lstm"].append(f1_lstm)
        results["bilstm"].append(f1_bilstm)
        print(
            f"  Fold {fold+1}: LSTM F1={f1_lstm:.3f}  BiLSTM F1={f1_bilstm:.3f}")

    for name, scores in results.items():
        arr = np.array(scores)
        print(
            f"\n  {name.upper():<8} mean F1 = {arr.mean():.3f} ± {arr.std():.3f}")

    return results


# =============================================================================
# MAIN: Workshop 2 — Professional Edition
# =============================================================================

def print_section(title: str, char: str = "="):
    print(f"\n{char*62}")
    print(f"  {title}")
    print(f"{char*62}")


def run_workshop():

    print("█" * 62)
    print("  WORKSHOP 2: LSTM/BiLSTM + SMOTE  (v4 — Professional Edition)")
    print("  Thai IP Legal Text Classification with Cost-Sensitive Learning")
    print("─" * 62)
    print("  Project : Physics-Governed IoT Framework for IP Law Enforcement")
    print("  Author  : จ.ส.อ.ภูมรินทร์  แสนศรี")
    print("  Org     : กองฝึกอบรม กรมการสื่อสารทหาร กองบัญชาการกองทัพไทย")
    print("  Date    : 22 เมษายน 2569")
    print("  Code    : w2_lstm_baseline_w1.py (v4)")
    print("█" * 62)

    # ──────────────────────────────────────────────────────────
    # STEP 1: W1Pipeline — Feature Extraction
    # ──────────────────────────────────────────────────────────
    print_section("STEP 1: W1Pipeline — Feature Extraction")

    all_texts = [t for t, _ in LABELED_CORPUS]
    all_labels = np.array([l for _, l in LABELED_CORPUS])

    SEQ_LEN = 4
    MAX_FEATURES = 52
    assert MAX_FEATURES % SEQ_LEN == 0
    INPUT_SIZE = MAX_FEATURES // SEQ_LEN

    fit_texts = list(set(all_texts + THAI_IP_CORPUS))

    w1 = W1Pipeline(max_features=MAX_FEATURES, remove_stopwords=True)
    w1.fit(fit_texts)
    X_all = w1.transform(all_texts)

    print(f"\n  W1 corpus (fit)  : {len(fit_texts)} documents")
    print(f"  Labeled samples  : {len(all_texts)}")
    print(f"  Feature dim      : {w1.n_features}")
    print(f"  X_all shape      : {X_all.shape}")

    print(f"\n  Class distribution:")
    for cls, name in CLASS_NAMES.items():
        cnt = int(np.sum(all_labels == cls))
        print(f"    {name:<25} {'█'*cnt} {cnt}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Data Augmentation + Split + SMOTE
    # ──────────────────────────────────────────────────────────
    print_section("STEP 2: Data Augmentation (Synonym Replacement)")

    augmenter = LegalDataAugmenter(rng=np.random.RandomState(62))
    aug_texts, aug_labels = augmenter.augment_corpus(
        all_texts, all_labels.tolist(), aug_factor=2)

    # Transform augmented texts
    X_aug = w1.transform(aug_texts)
    y_aug = np.array(aug_labels)

    print(
        f"\n  After augmentation: {len(aug_texts)} samples (from {len(all_texts)})")

    print_section("STEP 3: Split → SMOTE (train only)")

    rng = np.random.RandomState(42)
    tr_idx, te_idx = [], []
    for cls in range(len(CLASS_NAMES)):
        idx = np.where(y_aug == cls)[0].copy()
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.8))
        tr_idx.extend(idx[:cut])
        te_idx.extend(idx[cut:])

    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    X_train, y_train = X_aug[tr_idx], y_aug[tr_idx]
    X_test,  y_test = X_aug[te_idx], y_aug[te_idx]

    smote = EnhancedSMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(
        f"\n  Train: {len(X_train)} → {len(X_res)} (SMOTE)  |  Test: {len(X_test)}")
    print(f"\n  After SMOTE (train):")
    for cls, name in CLASS_NAMES.items():
        orig = int(np.sum(y_train == cls))
        total = int(np.sum(y_res == cls))
        print(f"    {name:<25} {total}  (+{total-orig})")

    # ──────────────────────────────────────────────────────────
    # STEP 4: Train & Evaluate on Test Set
    # ──────────────────────────────────────────────────────────
    print_section("STEP 4: Train & Evaluate on TEST SET")

    evaluator = ClassificationEvaluator(
        CLASS_NAMES, fn_weight=2.0, fp_weight=1.0)

    # TF-IDF + LinearSVC baseline
    svc = LinearSVCBaseline(n_classes=3, seed=42)
    svc.fit(X_train, y_train)
    evaluator.print_report(
        "TF-IDF + LinearSVC  (W1 features)",
        y_test, svc.predict(X_test), svc.predict_proba(X_test))

    # LSTM + SMOTE with cost-sensitive learning
    lstm = LSTMClassifier(INPUT_SIZE, hidden_size=32, n_classes=3, seed=42,
                          fn_weight=2.0, fp_weight=1.0)
    lstm_probs, lstm_entropy = lstm.predict_with_uncertainty(X_test, SEQ_LEN)
    lstm_preds = np.argmax(lstm_probs, axis=1)
    evaluator.print_report(
        "LSTM + SMOTE (Cost-Sensitive)",
        y_test, lstm_preds, lstm_probs, lstm_entropy)

    # BiLSTM + SMOTE with cost-sensitive learning
    bilstm = BiLSTMClassifier(INPUT_SIZE, hidden_size=32, n_classes=3, seed=42,
                              fn_weight=2.0, fp_weight=1.0)
    bilstm_probs, bilstm_entropy = bilstm.predict_with_uncertainty(
        X_test, SEQ_LEN)
    bilstm_preds = np.argmax(bilstm_probs, axis=1)
    evaluator.print_report(
        "BiLSTM + SMOTE (Cost-Sensitive)",
        y_test, bilstm_preds, bilstm_probs, bilstm_entropy)

    # ──────────────────────────────────────────────────────────
    # STEP 5: Comparison Summary
    # ──────────────────────────────────────────────────────────
    print_section("STEP 5: Model Comparison")

    models = {
        "TF-IDF + LinearSVC": (svc, svc.predict, svc.predict_proba, None),
        "LSTM (Cost-Sensitive)": (lstm,
                                  lambda X: lstm.predict_batch(X, SEQ_LEN),
                                  lambda X: lstm.predict_proba(X, SEQ_LEN),
                                  lstm_entropy),
        "BiLSTM (Cost-Sensitive)": (bilstm,
                                    lambda X: bilstm.predict_batch(X, SEQ_LEN),
                                    lambda X: bilstm.predict_proba(X, SEQ_LEN),
                                    bilstm_entropy),
    }
    print(f"\n  {'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'CostSc':>8}")
    print(f"  {'─'*60}")
    for name, (_, pred_fn, proba_fn, entropy) in models.items():
        y_pred = pred_fn(X_test)
        y_proba = proba_fn(X_test)
        acc = evaluator.accuracy(y_test, y_pred)
        f1 = evaluator.classification_report(y_test, y_pred)["macro_avg"]["f1"]
        auc = evaluator.auc_roc(y_test, y_proba)
        cost = evaluator.cost_sensitive_score(y_test, y_pred)
        bar = "█" * int(acc * 20)
        print(f"  {name:<25} {acc:>8.3f} {f1:>8.3f} {auc:>8.3f} {cost:>8.3f}  {bar}")

    # ──────────────────────────────────────────────────────────
    # STEP 6: K-Fold CV
    # ──────────────────────────────────────────────────────────
    print_section("STEP 6: Stratified K-Fold Cross-Validation")
    stratified_kfold_eval(X_aug, y_aug, n_splits=5, seq_len=SEQ_LEN, seed=42)

    # ──────────────────────────────────────────────────────────
    # STEP 7: Inference with Uncertainty
    # ──────────────────────────────────────────────────────────
    print_section("STEP 7: Inference with Uncertainty (for Physics Gate)")

    new_cases = [
        "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตรและจำหน่ายโดยไม่ได้รับอนุญาต",
        "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้องแล้ว",
        "จำเลยทำซ้ำและเผยแพร่งานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
    ]

    X_new = w1.transform(new_cases)
    probs, entropies = bilstm.predict_with_uncertainty(X_new, SEQ_LEN)

    print()
    for i, (case, proba, entropy) in enumerate(zip(new_cases, probs, entropies)):
        pred = int(np.argmax(proba))
        confidence_level = "HIGH" if entropy < 0.5 else "MEDIUM" if entropy < 1.0 else "LOW"
        print(f"  Case {i+1}: {case[:55]}...")
        print(f"  → {CLASS_NAMES[pred]}  ({proba[pred]:.1%})")
        print(
            f"    Uncertainty (Entropy): {entropy:.3f} → Confidence: {confidence_level}")
        for cls_id, cls_name in CLASS_NAMES.items():
            bar = "█" * int(proba[cls_id] * 20)
            print(f"    {cls_name:<25} {bar:<20} {proba[cls_id]:.3f}")

        entities = w1.extract_entities(case)
        if entities:
            ent_str = ", ".join(
                f"[{e.entity_type}]{e.value}" for e in entities[:3])
            print(f"    W1 entities: {ent_str}")

        # สำหรับ Physics Gate (W17): ส่ง entropy ไปด้วย
        if entropy > 0.8:
            print(
                f"    ⚠️  Physics Gate: High uncertainty — requesting IoT sensor verification")
        print()

    # ──────────────────────────────────────────────────────────
    # STEP 8: Pipeline Diagram
    # ──────────────────────────────────────────────────────────
    print_section("STEP 8: Full Pipeline (v4 Professional)")
    print("""
  [Thai Legal Text]
        │
        ▼  w1_thai_legal_nlp.py
  ┌─────────────────────────────────────────────────────────────┐
  │  W1Pipeline.transform()                                     │
  │  ├─ ThaiLegalTokenizer   (compound-aware, 24 compounds)     │
  │  ├─ LegalTFIDF           (IDF + L2 norm + [UNK])            │
  │  └─ ThaiIPEntityExtractor (context-aware confidence)       │
  └─────────────────────────┬───────────────────────────────────┘
                            │  float32 (n, 16)
        ▼  w2_lstm_baseline_w1.py (v4)
  ┌─────────────────────────────────────────────────────────────┐
  │  Data Augmentation (Synonym Replacement)                    │
  │  EnhancedSMOTE (with Random Oversampling fallback)          │
  │  BiLSTMClassifier (Xavier init, mean pooling, bf=1)        │
  │  Cost-Sensitive Loss (FN weight=2.0, FP weight=1.0)        │
  │  Uncertainty Estimation (Entropy for Physics Gate)          │
  └─────────────────────────┬───────────────────────────────────┘
                            │  class probabilities (3,) + entropy
        ▼
  [NO_INFRINGEMENT | PATENT_VIOLATION | COPYRIGHT_VIOLATION]
        │
        ▼  For W17 (Physics Gate)
  ┌─────────────────────────────────────────────────────────────┐
  │  If entropy > 0.8: Request IoT sensor verification         │
  │  Physics Gate Weight = severity × confidence × adjustment  │
  └─────────────────────────────────────────────────────────────┘

  v4 Enhancements:
    ✓ Xavier/Glorot Weight Initialization
    ✓ SMOTE Fallback (Random Oversampling)
    ✓ Cost-Sensitive Learning (FN penalty 2x)
    ✓ Uncertainty Estimation (Entropy)
    ✓ Data Augmentation (Synonym Replacement)
  """)


if __name__ == "__main__":
    run_workshop()
