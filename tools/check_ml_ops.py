#!/usr/bin/env python3

import onnx 
import argparse 
import os


def main():
    ap = argparse.ArgumentParser() 
    ap.add_argument("--onnx", required = True) 
    args = ap.parse_args()

    onnx_path = args.onnx 
    
    if os.path.exists(onnx_path) and '.onnx' in onnx_path:
        m = onnx.load(onnx_path) 
        domains = sorted({n.domain for n in m.graph.node }) 
        ops = sorted({(n.domain, n.op_type) for n in m.graph.node })
        print("Domains: ", domains) 
        print("Has ai.onnx.ml: ", "ai.onnx.ml" in domains) 

    else:
        raise FileNotFoundError('ONNX model not found')


if __name__ == "__main__":
    main()
    