"""查看原始研究资料的已有摘要,或运行网络合成与逐研究删除敏感性。"""
import argparse
import json
from .io import load
from .domain import summarize
from .nma import synthesize,leave_one_out

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','synthesize']);p.add_argument('file');a=p.parse_args()
    value,digest=load(a.file)
    if a.command=='inspect':out={'source_sha256':digest,'summary':summarize(value)}
    else:out={'source_sha256':digest,'synthesis':synthesize(value),'leave_one_out':leave_one_out(value)}
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
