"""Lossless station-month contract and explicit disk budgets; only small fixtures."""
import copy,json,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from analysis.deep_learning.dataset import DayCache,DayDataset
from analysis.deep_learning.compact import MonthlyWriter,estimate
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.features import contract,split_plan,fit_normalizer
from analysis.deep_learning.smoke import synthetic_cache,smoke_config
from analysis.deep_learning.storage import guard,tree_bytes,GIB

class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.env=patch.dict('os.environ',{'OIBC_STORAGE_ROOT':str(cls.root)}); cls.env.start()
        synthetic_cache(cls.root/'old'); c=DayCache(cls.root/'old/train')
        # Synthetic weather must obey the actual station->site join invariant.
        w=np.load(c.root/'weather.npy',mmap_mode='r+'); names=c.manifest['weather_features']
        for p in ['wa','wb']:
            for station in c.sites[p+'_station'].dropna().unique():
                members=np.flatnonzero(c.sites[p+'_station'].to_numpy()==station)
                for base in [v for v in contract()['weather_base'] if v.startswith(p+'_')]:
                    for name in [base,base+'_r10']:
                        j=names.index(name); shared=np.array(w[:,members[0],:,j])
                        for member in members: w[:,member,:,j]=shared
                    j=names.index(base+'_revision')
                    for member in members: w[:,member,:,j]=w[:,member,:,names.index(base)]-w[:,member,:,names.index(base+'_r10')]
        w.flush()
        y=np.load(c.root/'target.npy',mmap_mode='r+'); y[:]=np.asarray(c.energy)/c.sites.capacity_kw.to_numpy()[None,:,None]; y.flush()
        cls.old=DayCache(c.root)
        dest=cls.root/'new'; dest.mkdir()
        for name in ['sites.csv','composition.csv','static.npy']: (dest/name).write_bytes((c.root/name).read_bytes())
        writer=MonthlyWriter(dest,c.sites,names)
        labels=np.array([d.strftime('%Y-%m') for d in c.days])
        for month in dict.fromkeys(labels):
            ids=np.flatnonzero(labels==month)
            writer.write(month,ids,np.array(cls.old.weather[ids]),np.array(cls.old.energy[ids]),np.array(cls.old.missing_kind[ids]),np.array(cls.old.smp[ids]),np.array(cls.old.k[ids]))
        manifest=copy.deepcopy(c.manifest); manifest.pop('signature'); manifest.update(writer.metadata()); manifest['signature']=fingerprint(manifest)
        (dest/'manifest.json').write_text(json.dumps(manifest)); cls.new=DayCache(dest)
        cfg=smoke_config('cnn',cls.root/'old',cls.root); cfg['_smoke_test']=True
        cls.plan=split_plan(cfg,cls.old,'development'); cls.a=fit_normalizer(cls.old,cls.plan['cutoff']); cls.b=fit_normalizer(cls.new,cls.plan['cutoff'])
    @classmethod
    def tearDownClass(cls): cls.env.stop(); cls.temp.cleanup()
    def test_all_values_masks_and_normalizer_exact(self):
        for name in ['weather','temporal','energy','target','smp','k','missing_kind']:
            for d in range(len(self.old.days)): np.testing.assert_array_equal(getattr(self.old,name)[d],getattr(self.new,name)[d],strict=True)
        self.assertEqual({k:v for k,v in self.a.items() if k!='cache_signature'},{k:v for k,v in self.b.items() if k!='cache_signature'})
    def test_site_and_fleet_samples_all_fields(self):
        for fleet in [False,True]:
            for split in ['train','dev','holdout']:
                a=DayDataset(self.old,self.a,self.plan,split,fleet); b=DayDataset(self.new,self.b,self.plan,split,fleet)
                for i in [0,len(a)-1]:
                    aa=a[i]; bb=b[i]
                    for key in aa: torch.testing.assert_close(aa[key],bb[key],rtol=0,atol=0,equal_nan=True)
    def test_three_models_identical_forward_backward(self):
        from analysis.deep_learning.models import build_model,model_inputs
        from analysis.deep_learning.losses import compute_loss
        from analysis.deep_learning.inference import dimensions
        torch.set_num_threads(2)
        for name in ['cnn','transformer','cross_site']:
            fleet=name=='cross_site'; cfg=smoke_config(name,self.root/'old',self.root)
            aa=DayDataset(self.old,self.a,self.plan,'train',fleet)[-1]; bb=DayDataset(self.new,self.b,self.plan,'train',fleet)[-1]
            aa={k:v.unsqueeze(0) for k,v in aa.items()}; bb={k:v.unsqueeze(0) for k,v in bb.items()}
            model=build_model(cfg['model'],dimensions(self.old)).eval(); clone=copy.deepcopy(model)
            x=model(model_inputs(aa)); y=clone(model_inputs(bb)); torch.testing.assert_close(x,y,rtol=0,atol=0)
            groups=torch.tensor([0]*126+[1]*127)
            la,_=compute_loss(x,aa,cfg['loss'],cfg['group'],groups); lb,_=compute_loss(y,bb,cfg['loss'],cfg['group'],groups); la.backward(); lb.backward()
            for p,q in zip(model.parameters(),clone.parameters()):
                if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_weather_scalar_slice_boolean_selection(self):
        for sel in [0,slice(2,5),np.arange(253)%2==0]:
            np.testing.assert_array_equal(self.old.weather[0,sel],self.new.weather[0,sel])
    def test_lazy_spawn_state_has_no_open_months(self):
        self.new.weather[0]
        clone=pickle.loads(pickle.dumps(self.new)); self.assertEqual(len(clone._months),0)
        np.testing.assert_array_equal(clone.weather[0],self.old.weather[0]); self.assertLessEqual(len(clone._months),2)
    def test_spawn_workers_compact(self):
        from analysis.deep_learning.runtime import loader
        ds=DayDataset(self.new,self.b,self.plan,'train',False)
        b=next(iter(loader(ds,False,{'batch_size':2,'num_workers':2,'persistent_workers':False,'pin_memory':False},torch.Generator().manual_seed(1))))
        self.assertEqual(tuple(b['weather'].shape),(2,24,90))
    def test_duplicate_station_mismatch_fails(self):
        w=np.array(self.old.weather[:]); p='wa'; counts=self.old.sites[p+'_station'].value_counts(); station=counts[counts>1].index[0]; members=np.flatnonzero(self.old.sites[p+'_station'].to_numpy()==station)
        w[0,members[1],0,0]=98765
        with self.assertRaisesRegex(ValueError,'differ within'):
            MonthlyWriter(self.root/'bad',self.old.sites,self.old.manifest['weather_features']).write('2025-01',np.arange(len(w)),w,np.array(self.old.energy),np.array(self.old.missing_kind),np.array(self.old.smp),np.array(self.old.k))
    def test_cache_budget_and_quota_fail_before_allocation(self):
        dest=self.root/'not_created'
        with self.assertRaises(OSError): guard(dest,100,budget_root=dest,limit=1)
        with patch.dict('os.environ',{'OIBC_QUOTA_GIB':'0.00001'}):
            with self.assertRaises(OSError): guard(dest,1)
        self.assertFalse(dest.exists())
    def test_external_quota_usage_included(self):
        with patch.dict('os.environ',{'OIBC_EXTERNAL_USED_GIB':'12'}):
            with self.assertRaises(OSError): guard(self.root/'unused',0)

    def test_filesystem_free_space_fail(self):
        with patch('analysis.deep_learning.storage.shutil.disk_usage',return_value=type('D',(),{'free':1})()):
            with self.assertRaises(OSError): guard(self.root/'unused',1024)
    def test_foreign_path_rejected_and_symlink_not_counted(self):
        with self.assertRaises(ValueError): guard(self.root.parent/'foreign',0)
        (self.root/'link').symlink_to(self.old.root,target_is_directory=True)
        before=tree_bytes(self.root); (self.root/'link').unlink(); self.assertEqual(before,tree_bytes(self.root))
    def test_checkpoint_budget_preserves_previous(self):
        from analysis.deep_learning.checkpoint import save_checkpoint
        f=self.root/'last.pt'; f.write_bytes(b'previous')
        with patch.dict('os.environ',{'OIBC_RESULTS_MAX_GIB':'0.00001'}):
            with self.assertRaises(OSError): save_checkpoint(f,{'format_version':1,'model':{'x':torch.zeros(10)}})
        self.assertEqual(f.read_bytes(),b'previous'); self.assertFalse(f.with_suffix('.pt.tmp').exists())
    def test_estimate_savings(self):
        self.assertLess(estimate(100,self.old.sites),estimate(100,self.old.sites,'legacy')*.5)
if __name__=='__main__': unittest.main()
