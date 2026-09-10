"""5-fold CV LoRA fine-tuning of Qwen3-VL-8B-Instruct on ReXInTheWild.
Trains on 4 folds, evaluates on the held-out fold, repeats for all 5.
Loss is computed on the answer tokens only. Vision tower frozen."""
import os, json, re, gc, sys, time, random
import torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

FOLDS = json.load(open('cv_folds.json'))
K = FOLDS['k']; ROWS = FOLDS['rows']
IMGDIR = 'rexinthewild_images'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
EPOCHS = int(os.environ.get('EPOCHS', 3))
LR = float(os.environ.get('LR', 1e-4))
ACC = int(os.environ.get('ACC', 8))
MAXPX = int(os.environ.get('MAXPX', 768))
SEED = 0

SYS = "You are an expert clinician."
USR = ("Given the following medical image and a multiple-choice question, select the single best answer from the choices.\n"
       "ONLY output your FINAL answer, which must be placed inside curly brackets, e.g. {A}. Provide NO extra words, reasoning or explanation outside the curly brackets.\n"
       "When asked about left/right, answer from the patient's viewpoint unless otherwise stated.")

def load_img(r):
    im = Image.open(f"{IMGDIR}/{r['pmcid']}_{r['image_file_name']}").convert('RGB')
    if max(im.size) > MAXPX:
        s = MAXPX / max(im.size)
        im = im.resize((int(im.width*s), int(im.height*s)), Image.LANCZOS)
    return im

def messages(r):
    return [{'role':'system','content':[{'type':'text','text':SYS}]},
            {'role':'user','content':[{'type':'image'},
                {'type':'text','text':f"{USR}\n\nQuestion: {r['question']}\n\nChoices:\n{r['choices']}"}]}]

class DS(Dataset):
    def __init__(self, rows, proc): self.rows=rows; self.proc=proc
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        prompt = self.proc.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True)
        target = "{" + r['answer'].strip().upper() + "}"
        enc = self.proc(text=[prompt+target], images=[load_img(r)], return_tensors='pt')
        pre = self.proc(text=[prompt], images=[load_img(r)], return_tensors='pt')
        n_pre = pre['input_ids'].shape[1]
        labels = enc['input_ids'].clone()
        labels[:, :n_pre] = -100                       # loss on the answer only
        enc['labels'] = labels
        return {k: v[0] for k, v in enc.items()}

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) if batch[0][k].dim()==0 else
               torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True,
                   padding_value=(-100 if k=='labels' else 0)) if batch[0][k].dim()==1 else
               torch.cat([b[k].unsqueeze(0) for b in batch], 0)
            for k in batch[0]}

def extract(t):
    for pat in [r'\{\s*([A-E])\s*[\}\.\)]', r'\[\s*([A-E])\s*\]', r'\(\s*([A-E])\s*\)', r'^\s*([A-E])\s*\.?\s*$']:
        m = re.search(pat, (t or '').strip(), flags=re.I|re.M)
        if m: return m.group(1).upper()
    return ''

def build(fold):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    proc = AutoProcessor.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, quantization_config=bnb,
            dtype=torch.bfloat16, device_map='cuda:0')
    m = prepare_model_for_kbit_training(m)
    m.config.use_cache = False
    targets = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                    task_type='CAUSAL_LM', target_modules=targets)
    m = get_peft_model(m, lc)
    for n,p in m.named_parameters():
        if 'visual' in n or 'vision' in n: p.requires_grad = False
    m.print_trainable_parameters()
    return m, proc

@torch.no_grad()
def evaluate(m, proc, rows, tag):
    m.eval(); ok=0; preds=[]
    for j,r in enumerate(rows):
        prompt = proc.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True)
        enc = proc(text=[prompt], images=[load_img(r)], return_tensors='pt').to('cuda:0')
        out = m.generate(**enc, max_new_tokens=12, do_sample=False, num_beams=1)
        txt = proc.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        a = extract(txt); preds.append({'idx':r['idx'],'pred':a,'gold':r['answer'].strip().upper(),'raw':txt[:60]})
        if a == r['answer'].strip().upper(): ok += 1
        if (j+1) % 50 == 0: print(f'    [{tag}] eval {j+1}/{len(rows)} acc={ok/(j+1):.3f}', flush=True)
    return ok/len(rows), preds

results = {}
for fold in range(K):
    t0=time.time()
    tr = [r for r in ROWS if r['fold'] != fold]
    te = [r for r in ROWS if r['fold'] == fold]
    print(f'\n===== FOLD {fold}: train {len(tr)}  test {len(te)} =====', flush=True)
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    m, proc = build(fold)
    ds = DS(tr, proc)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    total = EPOCHS*len(dl)//ACC
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(1,total), pct_start=0.06)
    m.train(); step=0
    for ep in range(EPOCHS):
        for i,b in enumerate(dl):
            b = {k:v.to('cuda:0') for k,v in b.items()}
            loss = m(**b).loss / ACC
            loss.backward()
            if (i+1) % ACC == 0:
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(); step+=1
                if step % 20 == 0: print(f'    fold{fold} ep{ep} step{step}/{total} loss={loss.item()*ACC:.4f}', flush=True)
    acc, preds = evaluate(m, proc, te, f'fold{fold}')
    results[fold] = {'n_train':len(tr),'n_test':len(te),'acc':acc,'preds':preds,
                     'minutes':round((time.time()-t0)/60,1)}
    print(f'  FOLD {fold} accuracy = {acc:.4f}  ({results[fold]["minutes"]} min)', flush=True)
    json.dump(results, open('results/cv_qwen8b.json','w'), indent=1)
    del m; gc.collect(); torch.cuda.empty_cache()
accs=[results[f]['acc'] for f in results]
print(f'\n===== 5-fold CV: mean {np.mean(accs)*100:.1f}% (sd {np.std(accs)*100:.1f}) =====')
json.dump(results, open('results/cv_qwen8b.json','w'), indent=1)
