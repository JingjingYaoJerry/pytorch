"""
preprocessing_GUI.py (under development)
by Jingjing YAO (Jerry)

An issue-level, GUI-driven inspector that scans a CSV for many specific 
anomaly types and lets you accept/reject suggested fixes one issue at a time 
(Tkinter GUI).
"""

from __future__ import annotations
import os, csv, re, sys, tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------- 0. robust CSV ----------------------------
def _sniff_delim(fp: str) -> str:
    with open(fp, "r", encoding="utf8", errors="ignore") as f:
        sample = f.readline() + f.readline()
    return csv.Sniffer().sniff(sample).delimiter


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            df = pd.read_csv(path, delimiter=_sniff_delim(path), encoding=enc)
            print(f"[INFO] loaded with {enc}")
            return df
        except (UnicodeDecodeError, csv.Error):
            print(f"[WARN] {enc} failed")
    raise UnicodeDecodeError("cannot decode the file")

# ----------------------- 1. basic wrangling (template) ------------------
def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True))
    # keep ±inf for detection, clone then convert afterwards
    return df

# ------------------------ 2. scan every template rule -------------------
CURRENCY_RE = re.compile(r'^\$[\d,]+(\.\d+)?$')
DIGIT_RE    = re.compile(r'\d')
HTTP_RE     = re.compile(r'^https?://', flags=re.I)

def _scan(df: pd.DataFrame, miss_thr=.5, iqr_k=1.5) -> pd.DataFrame:
    """return DataFrame[ col, row, cause, suggestion_val, suggestion_text ]"""
    issues: List[Dict[str, Any]] = []

    # duplicates ----------------------------------------------------------
    for idx in df[df.duplicated()].index:
        issues.append(dict(col="__ROW__", row=idx,
                           cause="duplicate row",
                           sugg_val="DROP",
                           sugg_text="模板建议: drop_duplicates() 删除重复行"))

    # missing & high-missing ---------------------------------------------
    miss_rate = df.isna().mean()
    for col, rate in miss_rate.items():
        if rate > 0:
            for idx in df[df[col].isna()].index:
                is_high = rate > miss_thr
                cause   = "missing (>50%)" if is_high else "missing"
                if pd.api.types.is_numeric_dtype(df[col]):
                    text = "模板建议: 使用列均值/中位数填补"
                    val  = df[col].median()
                else:
                    text = "模板建议: 使用众数或 'UNK' 填补"
                    mode = df[col].mode()
                    val  = mode.iloc[0] if not mode.empty else "UNK"
                issues.append(dict(col=col, row=idx,
                                   cause=cause,
                                   sugg_val=val,
                                   sugg_text=text))

    # ±inf ---------------------------------------------------------------
    inf_mask = df.replace([np.inf, -np.inf], np.nan)    # create copy
    for col in df.columns:
        for idx, v in df[col].items():
            if v in (np.inf, -np.inf):
                issues.append(dict(col=col, row=idx,
                                   cause="±inf",
                                   sugg_val=np.nan,
                                   sugg_text="模板建议: 将 ±inf 转为 NaN"))

    # negative values ----------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        for idx in df[df[col] < 0].index:
            issues.append(dict(col=col, row=idx,
                               cause="negative value",
                               sugg_val=0,
                               sugg_text="模板建议: 设为 0 或绝对值"))

    # IQR outliers --------------------------------------------------------
    for col in num_cols:
        q1, q3 = df[col].quantile([.25, .75]); iqr = q3-q1
        lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
        mask = (df[col] < lo) | (df[col] > hi)
        for idx in df[mask].index:
            # template 没有直接给 remedy → 用 hf pipeline 生成
            issues.append(dict(col=col, row=idx,
                               cause="outlier",
                               sugg_val=float(df[col].median()),
                               sugg_text=_hf_suggestion("outlier")))

    # currency strings ----------------------------------------------------
    for col in df.select_dtypes(include=['object']).columns:
        for idx, v in df[col].items():
            if isinstance(v, str) and CURRENCY_RE.match(v):
                val = float(v.replace('$', '').replace(',', ''))
                issues.append(dict(col=col, row=idx,
                                   cause="currency string",
                                   sugg_val=val,
                                   sugg_text="模板建议: 去除$与逗号后转数值"))

    # mixed str & num -----------------------------------------------------
    for col in df.select_dtypes(include=['object']).columns:
        for idx, v in df[col].items():
            if isinstance(v, str) and DIGIT_RE.search(v) and not CURRENCY_RE.match(v):
                number = re.findall(r'\d+', v)
                if number:
                    issues.append(dict(col=col, row=idx,
                                       cause="mixed string&number",
                                       sugg_val=int(number[0]),
                                       sugg_text="模板建议: 提取数字部分并转数值"))

    # url prefix ----------------------------------------------------------
    for col in df.select_dtypes(include=['object']).columns:
        for idx, v in df[col].items():
            if isinstance(v, str) and HTTP_RE.match(v):
                stripped = HTTP_RE.sub('', v)
                issues.append(dict(col=col, row=idx,
                                   cause="URL prefix",
                                   sugg_val=stripped,
                                   sugg_text="模板建议: 去除 http(s):// 前缀"))
    return pd.DataFrame(issues)

# ------------------ hf pipeline for not-in-template cases ---------------
def _hf_suggestion(topic: str) -> str:
    try:
        from transformers import pipeline
        gen = pipeline("text2text-generation",
                       model="google/flan-t5-small",
                       max_length=64)
        q = f"Give a concise data-cleaning solution for {topic} in a dataset."
        return "HF建议: " + gen(q, num_return_sequences=1)[0]["generated_text"]
    except Exception:
        return "建议: 替换为列中位数（离线默认）"

# ------------------------ 3. Tkinter interactive ------------------------
class _GUI:
    def __init__(self, df: pd.DataFrame, issues: pd.DataFrame):
        self.df, self.issues = df, issues.reset_index(drop=True)
        self.i = 0
        self.root = tk.Tk(); self.root.title("Pre-processing Inspector")
        self._build(); self._show(); self.root.mainloop()

    def _build(self):
        self.lbl = tk.Label(self.root, anchor="w", justify="left",
                            font=("Consolas", 10)); self.lbl.pack(fill="x", padx=8, pady=4)
        self.tree = ttk.Treeview(self.root, show="headings", height=6)
        self.tree.pack(fill="x", padx=8)
        self.hint = tk.Label(self.root, anchor="w", justify="left",
                             fg="#c44"); self.hint.pack(fill="x", padx=8, pady=2)
        self.sol = tk.Label(self.root, anchor="w", justify="left",
                            fg="#070"); self.sol.pack(fill="x", padx=8, pady=2)
        frm = tk.Frame(self.root); frm.pack(pady=6)
        tk.Button(frm, text="Replace", command=self._replace, width=15,
                  bg="#60c979").grid(row=0, column=0, padx=4)
        tk.Button(frm, text="Leave Unchanged", command=self._skip, width=15,
                  bg="#eeeeee").grid(row=0, column=1, padx=4)

    def _show(self):
        if self.i >= len(self.issues):
            messagebox.showinfo("Done", "全部异常已浏览完毕")
            self.root.destroy(); return
        iss = self.issues.loc[self.i]
        self.lbl.config(text=f"Issue {self.i+1}/{len(self.issues)}  "
                             f"row={iss.row}  col={iss.col}  cause={iss.cause}")
        # table
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.df.columns)
        for c in self.df.columns:
            self.tree.heading(c, text=c); self.tree.column(c, width=120)
        values = [self.df.at[iss.row, c] if iss.row in self.df.index else "—"
                  for c in self.df.columns]
        self.tree.insert("", "end", values=values)
        # hint & solution
        self.hint.config(text=f"Hint : {iss.cause}")
        self.sol.config(text=f"Solution : {iss.sugg_text}\n→ 替换值: {iss.sugg_val}")

    # ---- actions
    def _replace(self):
        self._apply(True)
        self.i += 1; self._show()

    def _skip(self):
        self._apply(False)
        self.i += 1; self._show()

    def _apply(self, do_replace: bool):
        iss = self.issues.loc[self.i]
        if not do_replace:
            return
        if iss.col == "__ROW__":
            if iss.sugg_val == "DROP" and iss.row in self.df.index:
                self.df.drop(index=iss.row, inplace=True)
        else:
            if iss.row in self.df.index:
                self.df.at[iss.row, iss.col] = iss.sugg_val

# ------------------------------ 4. API ----------------------------------
def preprocess(csv_path: str, miss_thr=.5) -> pd.DataFrame:
    """Read → scan → manual GUI fix → return cleaned DataFrame"""
    df = _standardise(_read_csv(csv_path))
    # keep copy for inf detection then convert inf→NaN for downstream
    issues = _scan(df, miss_thr=miss_thr)
    if not issues.empty:
        _GUI(df, issues)     # blocks until user finishes
    else:
        print("[INFO] no abnormality detected")
    # 最终统一将 ±inf 置 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ------------------------------ 5. CLI ----------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_strict.py data.csv"); sys.exit(0)
    cleaned = preprocess(sys.argv[1])
    print("\n预处理中完成，前 5 行示例：")
    print(cleaned.head())