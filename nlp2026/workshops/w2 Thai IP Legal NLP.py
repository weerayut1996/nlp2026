import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import classification_report, confusion_matrix
import torch 
import torch.nn as nn

#  1. สร้างพจนานุกรมคำพ้องความหมาย และ Data Augmentation
LEGAL_SYNONYMS = {
"ละเมิด":["ฝ่าฝืน","กระทำผิด","ล่วงสิทธิ"],
"จำหน่าย":["ขาย","เผยแพร่","กระจายสินค้า"],
"ปลอมแปลง":["ทำเทียม","เลียนแบบ"]
}

def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    for i , word in enumerate(words):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return " ".join(new_words)
original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original)
# print(f"---Data Augmentation---")
# print(f"Original : {original}")
# print(f"Augmented : {augmented}")

# 2. SMOTE with Fallback
import numpy as np
from imblearn.over_sampling import SMOTE , RandomOverSampler
from collections import Counter
def balance_legal_data(X,y):
   counts = Counter(y)
#    print(f"Original distribution : {counts}")
   # คลาสน้อยสุดมีกี่ตัว
   min_samples = min(counts.values())
   if min_samples > 1 :
       sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=42)
   else:
       sampler = RandomOverSampler(random_state=42)
   X_res , y_res = sampler.fit_resample(X,y)
   print(f"Balanced distribution: {Counter(y_res)}")
   return X_res , y_res

# จำลองข้อมูล Imbalance (class 0 = 10 , class 1 = 2)
X_mock = np.random.randn(12,5)
y_mock = np.array([0]*10 + [1]*2)
X_res, y_res = balance_legal_data(X_mock,y_mock)

# เปลี่ยนข้อมูลนำเข้า เป็น 3  มิติ เพื่อ LSTM
X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1)
y_res_tensor = torch.tensor(y_res, dtype=torch.long)
print(f"Shape หลังจากทำ SMOTE (2D) : {X_res.shape}")
print(f"Shape สำหรับนำเข้า (3D) : {X_res_3d.shape}") # (Batch , Seq_len , Input_size)


    
# 3.1 BiLSTM 
class LegalBiLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3):
        super(LegalBiLSTM,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Mean Pooling
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
model = LegalBiLSTM()
sample_input = torch.randn(1,5,5) # 1 doc 5 token 16 dim
output = model(sample_input)
# print(f"--BiLSTM Output--")
# print(f"Logics: {output.detach().numpy()}")


#3.2 LSTM
class LegalLSTM(nn.Module):
    def __init__(self,input_dim=5, hidden_dim=32,output_dim=3):
        super(LegalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim,output_dim)
        nn.init.xavier_uniform(self.fc.weight) # เลือกค่า weight ที่เหมาะสม ในการเทรนรอบแรก

    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
    
# 3.3 เทรนโมเดล LSTM VS BiLSTM
def train_and_evaluate(model_class , name, X,y,Class_names):
    # print(f"Training : {name} ....")
    model = model_class()
    # Cost-Sensitive Weight (FN = False Negative)
    #0 ไม่ผิด 1 ละเมิดสิทธิบัตร 2 ละเมิดลิขสิทธิ์
    weights = torch.tensor([1.0,2.0,2.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Sample Training Loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs= model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    # Evaluate with CM
        model.eval()
        with torch.no_grad():
            logits = model(X)
            y_pred = torch.argmax(logits, dim=1).numpy()
    cm = confusion_matrix(y,y_pred)

    # Heat Map
    fig,ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, fmt='d',cmap="Greens",
                xticklabels=Class_names,yticklabels=Class_names, ax=ax)
    ax.set_title(f"Confusion Matrix : {name}")
    ax.set_ylabel(f"Actual")
    ax.set_xlabel(f"Predicted")
    plt.tight_layout()
    plt.show()
    return classification_report(y,y_pred ,target_names=Class_names)

# รันเปรียบเทียบ Training & Evaluate
class_list = ['NO-INF', 'PATENT']
# X_train , y_train = ไม่ผ่าน MOTE , X_res, y_res = ผ่านการ SMOTE แล้ว
report_lstm = train_and_evaluate(LegalLSTM, "Unidirectional LSTM", X_res_3d, y_res_tensor, class_list)
# report_bilstm = train_and_evaluate(LegalBiLSTM,"Bidirectional LSTM", X_res_3d,y_res_tensor,class_list)
