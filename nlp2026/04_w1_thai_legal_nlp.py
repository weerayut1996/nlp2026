# การตัดคำ Tokenization + custom_Dict
import re
from pythainlp.tokenize import word_tokenize

LEGAL_KEYWORDS = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด"]

def legal_tokenizer(text):
    # 1.Protect Compound Keywords ด้วย Placeholder
    sorted_kw = sorted(LEGAL_KEYWORDS,key=len,reverse=True)
    placeholders = {}
    protected = text
    for i, kw in enumerate(sorted_kw):
        ph = f"__KW{i}__"
        if kw in protected:
            placeholders[ph] = kw
            protected = protected.replace(kw,ph)
    # 2. tokenize ด้วย pythainlp
    tokens_raw = word_tokenize(protected,engine="newmm",keep_whitespace=False)
    
    # 3. restore placeholder
    return [placeholders.get(t,t) for t in tokens_raw]

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"Input: {test_text}")
print(f"Output: {tokens}")



# ตัดด้วย Deep Learning (Deepcut)
# import deepcut
# print(f"Output Deepcut: {deepcut.tokenize(test_text)}")