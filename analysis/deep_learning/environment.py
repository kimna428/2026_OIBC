"""Read-only environment check plus optional tiny CUDA matmul/backward probe."""
import argparse,json,platform,shutil,subprocess,sys
from pathlib import Path
import torch
from .config import ROOT
from .features import contract
from .models import build_model,parameter_count

def parameter_report():
    import yaml
    result={}
    dims={'weather_dim':len(contract()['weather_features']),'temporal_dim':8,'static_dim':146,'num_sites':253}
    for file in sorted((ROOT/'configs/deep_learning').glob('*.yaml')):
        config=yaml.safe_load(file.read_text())
        with torch.device('meta'): model=build_model(config['model'],dims)
        result[file.stem]={'parameters':parameter_count(model),'model':config['model'],'dimensions':dims}
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--require-cuda',action='store_true'); p.add_argument('--cuda-probe',action='store_true'); p.add_argument('--parameters',action='store_true'); p.add_argument('--output'); args=p.parse_args()
    version=tuple(int(x) for x in str(torch.__version__).split('+')[0].split('.')[:2])
    if version[0]!=2 or version[1]<5: raise SystemExit('This pipeline requires PyTorch >=2.5,<3')
    info={'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'torch_cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'remote_connection_attempted':False}
    if shutil.which('nvidia-smi'):
        run=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
        info['nvidia_smi']=run.stdout.strip() or run.stderr.strip()
    if torch.cuda.is_available():
        info['gpu_name']=torch.cuda.get_device_name(0); info['total_vram_bytes']=torch.cuda.get_device_properties(0).total_memory; info['bf16_supported']=torch.cuda.is_bf16_supported()
        if args.cuda_probe:
            x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.T).mean().backward(); torch.cuda.synchronize(); info['tiny_cuda_backward']='passed'
    if args.parameters: info['parameter_counts']=parameter_report()
    text=json.dumps(info,indent=2); print(text)
    if args.output: Path(args.output).write_text(text)
    if args.require_cuda and not info['cuda_available']: raise SystemExit('CUDA is unavailable. Select a driver-compatible CUDA wheel via the official PyTorch installer.')
if __name__=='__main__': main()
