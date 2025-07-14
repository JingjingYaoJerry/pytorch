# preprocess_gui.py
# 纯标准库 + pandas + numpy 即可运行
import csv, os, sys, math
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# =============== 1. Load CSV（Encoding/Delimeter Sniffing） ================= #
def sniff_delim(path: str) -> str:
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        sample = "".join(f.readline() for _ in range(5))
    #============ Delimiter Sniffing ============#
    # https://docs.python.org/zh-cn/3.13/library/csv.html#csv.Sniffer
    return csv.Sniffer().sniff(sample).delimiter # return the detected dialect's delimiter


def load_csv(path: str) -> pd.DataFrame:
    #============ File Existance Checking ============#
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    #============ Encoding Sniffing ============#
    for enc in ("utf8", "utf-16", "latin1"):
        try:
            df = pd.read_csv(path, delimiter=sniff_delim(path), encoding=enc)
            print(f"[INFO] Successfully loaded {path} with encoding {enc}")
            return df
        except UnicodeDecodeError:
            print(f"[WARN] Encoding {enc} failed，trying next...")
    raise UnicodeDecodeError("No Suitable Encoding Found",)


# =============== 2. Basic Standardization ================= #
def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    #============ Column Name Standardization ============#
    df.columns = (df.columns.astype(str) # get all column names
                  .str.strip() # strip leading/trailing spaces
                  .str.lower() # convert to lower case
                  .str.replace(r"\s+", "_", regex=True))
                # replace all whitespace with underscore
    # replace infinities with NaN for consistency
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # replace
    return df


# =============== 3. 扫描异常，生成 issue 表 ================= #
def scan_issues(df: pd.DataFrame,
                miss_thresh: float = 0.5,
                iqr_k: float = 1.5) -> pd.DataFrame:
    """
    返回 issue DataFrame:
        col, row_idx, reason, suggestion
    """
    issues: List[Dict[str, Any]] = []

    # 高缺失列
    na_ratio = df.isna().mean()
    for col, ratio in na_ratio.items():
        if ratio > miss_thresh:
            for idx in df[df[col].isna()].index:
                issues.append(dict(col=col, row_idx=int(idx),
                                   reason="缺失值 (>50%)",
                                   suggestion=("median" if pd.api.types.is_numeric_dtype(df[col])
                                               else "UNK")))

    # 3-2 一般缺失
    for col in df.columns:
        miss_rows = df[df[col].isna()].index
        for idx in miss_rows:
            issues.append(dict(col=col, row_idx=int(idx),
                               reason="缺失值",
                               suggestion=("median" if pd.api.types.is_numeric_dtype(df[col])
                                           else "mode")))

    # 3-3 负值示例
    for col in df.select_dtypes(include=[np.number]).columns:
        bad = df[col] < 0
        for idx in df[bad].index:
            issues.append(dict(col=col, row_idx=int(idx),
                               reason="负值(非法)",
                               suggestion=0))

    # 3-4 离群点（IQR）
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k*iqr, q3 + iqr_k*iqr
        outliers = df[(df[col] < lo) | (df[col] > hi)].index
        for idx in outliers:
            issues.append(dict(col=col, row_idx=int(idx),
                               reason="离群值",
                               suggestion=float(df[col].median())))

    return pd.DataFrame(issues)


# =============== 4. GUI 逻辑 ================= #
class IssueFixer:
    def __init__(self, df: pd.DataFrame, issues: pd.DataFrame):
        self.df = df
        self.issues = issues.reset_index(drop=True)
        self.i = 0                          # 当前指针
        self.root = tk.Tk()
        self.root.title("数据预处理修复窗口")
        self.build_widgets()
        self.show_issue()

        self.root.mainloop()               # 阻塞直到窗口关闭

    # ---------- 构建界面 ---------- #
    def build_widgets(self):
        self.info = tk.Label(self.root, text="", justify="left", font=("Consolas", 10))
        self.info.pack(padx=10, pady=5, anchor="w")

        self.tree = ttk.Treeview(self.root, show="headings")
        self.tree.pack(fill="x", padx=10)

        self.new_val_entry = tk.Entry(self.root, width=30)
        self.new_val_entry.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        self.btn_rep = tk.Button(btn_frame, text="Replace", width=12,
                                 command=self.replace_one, bg="#70c1b3")
        self.btn_all = tk.Button(btn_frame, text="Replace All", width=12,
                                 command=self.replace_all, bg="#f2c14e")
        self.btn_skip = tk.Button(btn_frame, text="Skip", width=12,
                                  command=self.skip, bg="#ccc")
        self.btn_finish = tk.Button(btn_frame, text="Finish", width=12,
                                    command=self.finish, bg="#247ba0")

        self.btn_rep.grid(row=0, column=0, padx=4)
        self.btn_all.grid(row=0, column=1, padx=4)
        self.btn_skip.grid(row=0, column=2, padx=4)
        self.btn_finish.grid(row=0, column=3, padx=4)

    # ---------- 显示当前 issue ---------- #
    def show_issue(self):
        if self.i >= len(self.issues):
            messagebox.showinfo("Done", "所有问题已处理完毕")
            return

        issue = self.issues.loc[self.i]
        col, idx, rsn, sug = issue.col, issue.row_idx, issue.reason, issue.suggestion

        self.info.config(text=f"Issue {self.i+1}/{len(self.issues)}  "
                              f"列: {col}  行索引: {idx}  原因: {rsn}  建议: {sug}")

        # 重建表格
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.df.columns)
        for c in self.df.columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")
        row_vals = [self.df.at[idx, c] for c in self.df.columns]
        self.tree.insert("", "end", values=row_vals)

        self.new_val_entry.delete(0, tk.END)
        self.new_val_entry.insert(0, str(sug))

    # ---------- 按钮回调 ---------- #
    def replace_one(self):
        val = self.parse_val(self.new_val_entry.get())
        self.apply_fix(self.i, val)
        self.i += 1
        self.show_issue()

    def replace_all(self):
        val = self.parse_val(self.new_val_entry.get())
        # 同列同原因全部替换
        issue_now = self.issues.loc[self.i]
        same = (self.issues.col == issue_now.col) & (self.issues.reason == issue_now.reason)
        idxs = self.issues[same].index.tolist()
        for j in idxs:
            self.apply_fix(j, val)
        self.i += 1
        self.show_issue()

    def skip(self):
        self.i += 1
        self.show_issue()

    def finish(self):
        self.root.destroy()   # 关闭窗口

    # ---------- 实际执行替换 ---------- #
    def apply_fix(self, issue_index: int, new_val):
        row = self.issues.loc[issue_index]
        self.df.at[row.row_idx, row.col] = new_val

    # ---------- 字符串转数值（尽量保持 dtype） ---------- #
    @staticmethod
    def parse_val(s: str):
        if s.lower() in ("nan", "none", ""):
            return np.nan
        # 尝试转 int / float
        try:
            if "." in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            return s


# =============== 5. 对外的简单函数 ================= #
def preprocess(csv_path: str,
               miss_thresh: float = 0.5) -> pd.DataFrame:
    """
    载入 csv → 扫描异常 → 弹 GUI 让用户修复 → 返回修复后的 DataFrame
    """
    df = basic_cleanup(load_csv(csv_path))
    issues = scan_issues(df, miss_thresh=miss_thresh)
    if issues.empty:
        print("[INFO] 未发现异常，直接返回 DataFrame")
        return df
    IssueFixer(df, issues)    # 阻塞直到用户点击 Finish
    return df


# =============== 6. CLI 调用示例 ================= #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python preprocess_gui.py data.csv")
        sys.exit(1)
    out_df = preprocess(sys.argv[1])
    print("\n=== 预处理结束，前 5 行预览 ===")
    print(out_df.head())