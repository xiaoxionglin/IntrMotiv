import torch,json,hashlib
from pathlib import Path
root=Path('/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir')
items=[('intrmotiv_source_credit_retirement_20260904','SCR_C15_ARR_DIRS_S123'),('intrmotiv_saturday_batch_20260905','SAT_C15_ARR_DIRO_FILM_S8'),('intrmotiv_dg_policy_gradient_first_outcome_20260906','DGP_C15_HIT_JOINT_FILM_S99')]
result=[]
for batch,name in items:
 run=root/batch/(name+'_')/('00_'+name)
 paths=list((run/'checkpoint_p0').glob('checkpoint_*.pth'))
 p=min(paths,key=lambda p:abs(int(p.stem.split('_')[-1])-75000000))
 c=torch.load(p,map_location='cpu',weights_only=False)
 sd=c['model']; config=json.load(open(run/'config.json'))
 keys=[k for k in sd if any(t in k for t in ['decoder.state_layer','decoder.target_modulation','decoder.output_layer','action_parameterization','critic_linear'])]
 result.append({'run':name,'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'config':config,'tensor_shapes':{k:list(v.shape) for k,v in sd.items()},'weights':{k:sd[k].tolist() for k in keys}})
print(json.dumps(result))
