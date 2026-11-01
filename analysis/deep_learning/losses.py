"""Masked CF losses, with a capacity-normalized full-fleet-only group term."""
import torch
import torch.nn.functional as F

def masked_mean(value,weight): return (value*weight).sum()/weight.sum().clamp_min(1e-12)

def compute_loss(pred,batch,loss_config,group_config,group_index):
    pred=pred.float(); target=batch['target_cf'].float(); mask=batch['loss_mask'].bool()
    if pred.shape!=target.shape or mask.shape!=target.shape: raise ValueError('Prediction/target/mask shapes differ')
    if not torch.isfinite(pred).all(): raise FloatingPointError('NaN/Inf model prediction')
    if (mask&~torch.isfinite(target)).any(): raise ValueError('Observed target mask includes NaN/Inf')
    # Local arithmetic substitution only; missing targets in cache/batch remain NaN.
    safe=torch.where(mask,target,torch.zeros_like(target))
    name=loss_config['name']
    error=F.huber_loss(pred,safe,reduction='none',delta=loss_config.get('huber_delta',.05)) if name=='huber' else (pred-safe).abs()
    price=batch['smp'].float()*batch['k'].float(); capacity=batch['capacity_kw'].float()
    if not torch.isfinite(price).all() or (price<0).any(): raise ValueError('Invalid price weights')
    weight=mask.float()
    if name=='weighted_l1':
        p=price.unsqueeze(1) if pred.ndim==3 else price
        weight=weight*capacity.unsqueeze(-1)*p
    site=masked_mean(error,weight)
    group=pred.sum()*0; group_mass=pred.new_zeros(())
    if group_config.get('enabled',False):
        if pred.ndim!=3 or pred.shape[1]!=len(group_index): raise ValueError('Group loss requires all canonical sites')
        complete=mask.all(dim=1)&batch['group_complete'].bool()
        group_errors=[]; group_weights=[]
        for index in [0,1]:
            members=group_index==index
            if not members.any(): raise ValueError('Both resource groups must be represented')
            cap=capacity[:,members].sum(dim=1)
            actual=(safe[:,members]*capacity[:,members,None]).sum(dim=1)
            forecast=(pred[:,members]*capacity[:,members,None]).sum(dim=1)
            group_errors.append((forecast-actual).abs()/cap[:,None].clamp_min(1e-12))
            group_weights.append(complete.float()*price*cap[:,None])
        error_g=torch.stack(group_errors,dim=1); weight_g=torch.stack(group_weights,dim=1)
        group=masked_mean(error_g,weight_g); group_mass=weight_g.sum()
    total=group_config.get('alpha',1.)*site+group_config.get('beta',0.)*group
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite training loss')
    return total,{'site_loss':float(site.detach()),'group_loss':float(group.detach()),'site_mass':float(weight.sum().detach()),'group_mass':float(group_mass.detach())}
