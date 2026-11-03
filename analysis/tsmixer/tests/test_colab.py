"""Local fixtures only: no Colab/Drive/network/GPU/full-training side effects."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from analysis.tsmixer.checkpoint import save, load, capture_rng, restore_rng
from analysis.tsmixer.config import load_config
from analysis.tsmixer.model import TSMixer
from analysis.tsmixer.train import train_epoch, run
from analysis.tsmixer.colab import bind_session, export_development, stage_dataset, run_logged
from analysis.tsmixer.persistence import backup_run, restore_latest, verify_generation, sha256, durable_json


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name).resolve(); self.local=self.root/'local'; self.drive=self.root/'drive'
        self.local.mkdir()
        torch.set_num_threads(2)
        self.state={'model':{'weight':torch.ones(1)}, 'optimizer':{}, 'scheduler':{'last_epoch':1},
                    'scaler':{}, 'epoch':1, 'global_step':5, 'best':.2, 'rng':capture_rng(),
                    'loader_rng':torch.Generator().get_state(), 'normalizer':{}, 'config':{},
                    'run_signature':'signature', 'cache_signature':'cache', 'feature_signature':'feature',
                    'sources':{}, 'environment':{'git_commit':'a'*40}, 'smoke_test':True}
        self.write_epoch(1)

    def tearDown(self): self.tmp.cleanup()

    def write_epoch(self, epoch):
        self.state.update(epoch=epoch,global_step=epoch*5)
        save(self.local/'last.pt',self.state); save(self.local/'best.pt',self.state)
        (self.local/'run.json').write_text('{}')
        (self.local/'epochs.jsonl').write_text(json.dumps({'epoch':epoch})+'\n')

    def test_backup_restore_preserves_all_state(self):
        generation=backup_run(self.local,self.drive)
        self.assertTrue((generation/'COMMITTED.json').is_file())
        restored=self.root/'restored'
        self.assertEqual(restore_latest(self.drive,restored),generation)
        self.assertEqual(load(restored/'last.pt')['scheduler'],{'last_epoch':1})
        self.assertEqual(sha256(restored/'best.pt'),sha256(self.local/'best.pt'))
        self.assertEqual(load(restored/'last.pt')['best'],.2)

    def test_partial_newer_generation_ignored(self):
        old=backup_run(self.local,self.drive)
        self.write_epoch(2); newer=backup_run(self.local,self.drive)
        (newer/'COMMITTED.json').unlink()
        restored=self.root/'restored'
        self.assertEqual(restore_latest(self.drive,restored),old)
        self.assertEqual(load(restored/'last.pt')['epoch'],1)

    def test_corrupt_latest_falls_back(self):
        old=backup_run(self.local,self.drive)
        self.write_epoch(2); newer=backup_run(self.local,self.drive)
        (newer/'last.pt').write_bytes(b'broken upload')
        self.assertEqual(restore_latest(self.drive,self.root/'restored'),old)

    def test_corrupt_best_falls_back(self):
        old=backup_run(self.local,self.drive)
        self.write_epoch(2); newer=backup_run(self.local,self.drive)
        (newer/'best.pt').write_bytes(b'broken best')
        self.assertEqual(restore_latest(self.drive,self.root/'restored'),old)

    def test_failed_copy_has_no_commit_marker_and_old_backup_survives(self):
        old=backup_run(self.local,self.drive)
        with patch('analysis.tsmixer.persistence.copy_verified',side_effect=OSError('Drive disconnected')):
            with self.assertRaises(OSError): backup_run(self.local,self.drive)
        markers=list((self.drive/'snapshots').glob('*/COMMITTED.json'))
        self.assertEqual(len(markers),1)
        self.assertEqual(restore_latest(self.drive,self.root/'restored'),old)

    def test_all_corrupt_fail_not_silent_fresh_training(self):
        folder=backup_run(self.local,self.drive); (folder/'manifest.json').write_text('broken')
        with self.assertRaisesRegex(RuntimeError,'No fully verified'): restore_latest(self.drive,self.root/'restored')
        self.assertFalse((self.root/'restored').exists())

    def test_fuse_without_fsync_still_verifies_generation(self):
        import errno
        with patch('os.fsync',side_effect=OSError(errno.ENOTSUP,'FUSE fsync unsupported')):
            folder=backup_run(self.local,self.drive)
        self.assertEqual(verify_generation(folder)['epoch'],1)

    def test_real_io_error_is_not_ignored(self):
        import errno
        old=backup_run(self.local,self.drive)
        with patch('os.fsync',side_effect=OSError(errno.EIO,'Drive I/O error')):
            with self.assertRaises(OSError): backup_run(self.local,self.drive)
        self.assertEqual(restore_latest(self.drive,self.root/'restored'),old)

    def test_empty_backup_returns_none(self):
        self.assertIsNone(restore_latest(self.drive,self.root/'restored'))

    def test_restore_never_overwrites_local_run(self):
        backup_run(self.local,self.drive)
        with self.assertRaises(FileExistsError): restore_latest(self.drive,self.local)

    def test_backup_excludes_cache_and_temporary_files(self):
        (self.local/'synthetic_cache').mkdir(); (self.local/'synthetic_cache/weather.npy').write_bytes(b'not a result')
        (self.local/'last.pt.tmp').write_bytes(b'incomplete')
        folder=backup_run(self.local,self.drive)
        names={r['path'] for r in verify_generation(folder)['files']}
        self.assertNotIn('last.pt.tmp',names); self.assertFalse((folder/'synthetic_cache').exists())

    def test_backup_path_traversal_and_nested_dest_rejected(self):
        with self.assertRaises(ValueError): backup_run(self.local,self.local/'drive')
        folder=backup_run(self.local,self.drive)
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest['files'][0]['path']='../outside'
        durable_json(folder/'manifest.json',manifest)
        durable_json(folder/'COMMITTED.json',{'manifest_sha256':sha256(folder/'manifest.json')})
        with self.assertRaises(ValueError): verify_generation(folder)

    def test_session_pin_config_and_source(self):
        a=bind_session(self.drive,'a'*40,{'seed':2026},'smoke')
        self.assertEqual(a,bind_session(self.drive,'a'*40,{'seed':2026},'smoke'))
        with self.assertRaises(ValueError): bind_session(self.drive,'b'*40,{'seed':2026},'smoke')
        with self.assertRaises(ValueError): bind_session(self.drive,'a'*40,{'seed':2027},'smoke')
        with self.assertRaises(ValueError): bind_session(self.drive,'a'*40,{'seed':2026},'development')


class DataAndExportTests(unittest.TestCase):
    def test_verified_local_dataset_copy_and_conflict(self):
        from analysis.deep_learning.prepare import REQUIRED
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); source=root/'drive'; target=root/'local'; records={}
            for name in REQUIRED:
                p=source/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(name.encode())
                records[name]={'sha256':sha256(p),'bytes':p.stat().st_size}
            reference=root/'reference.json'; reference.write_text(json.dumps({'files':records}))
            self.assertEqual(len(stage_dataset(source,target,reference)),8)
            self.assertEqual(len(stage_dataset(source,target,reference)),8)
            (target/'sites.csv').write_text('user modification')
            with self.assertRaisesRegex(ValueError,'Existing local dataset differs'):
                stage_dataset(source,target,reference)
            self.assertEqual((target/'sites.csv').read_text(),'user modification')

    def test_missing_or_corrupt_drive_data_not_copied(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); source=root/'source'; source.mkdir()
            reference=root/'reference.json'; reference.write_text('{"files":{}}')
            with self.assertRaises(FileNotFoundError): stage_dataset(source,root/'local',reference)

    def test_csv_uses_canonical_sites_capacity_groups_and_kst(self):
        from analysis.deep_learning.smoke import synthetic_cache
        from analysis.tsmixer.data import open_cache
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); cache=open_cache(synthetic_cache(root/'cache')/'train')
            out=root/'eval'; out.mkdir()
            plan={'cutoff':'2025-01-31 17:00:00+09:00','dev_start':'2025-02-01','dev_end':'2025-02-02'}
            (out/'metrics.json').write_text(json.dumps({'split':'dev','holdout_used':False,'training_cutoff':plan['cutoff'],
                                                       'cache_signature':cache.manifest['signature']}))
            np.save(out/'predictions.npy',np.full((1,253,24),.2,dtype='float32'))
            result=export_development(out,cache,plan)
            with (result/'development_site_predictions.csv').open() as f: sites=list(csv.DictReader(f))
            with (result/'development_group_forecast.csv').open() as f: groups=list(csv.DictReader(f))
            self.assertEqual(len(sites),253*24); self.assertEqual(len(groups),48)
            self.assertEqual(sites[0]['site_id'],'PV001')
            self.assertEqual(sites[0]['ts'],'2025-02-01T01:00:00+09:00')
            self.assertEqual(sites[23]['ts'],'2025-02-02T00:00:00+09:00')
            self.assertAlmostEqual(sum(float(x['pred_kwh']) for x in sites),sum(float(x['forecast_kwh']) for x in groups),places=4)
            self.assertFalse(json.loads((result/'metadata.json').read_text())['independent_oof'])
            with self.assertRaises(FileExistsError): export_development(out,cache,plan)
            (out/'metrics.json').write_text(json.dumps({'split':'holdout','holdout_used':True,'training_cutoff':plan['cutoff']}))
            with self.assertRaises(ValueError): export_development(out,cache,plan)

    def test_logging_subprocess_success_failure(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'console.log'
            run_logged([sys.executable,'-c','print("small test")'],file)
            self.assertIn('small test',file.read_text())
            with self.assertRaises(subprocess.CalledProcessError):
                run_logged([sys.executable,'-c','raise SystemExit(3)'],file)


class ResumeAndNotebookTests(unittest.TestCase):
    def test_cpu_smoke_backup_restore_resume_missing_synthetic_cache(self):
        torch.set_num_threads(2)
        c=load_config('configs/tsmixer/colab.yaml')
        c['model'].update(hidden_dim=16,blocks=1,time_hidden=12,feature_hidden=24)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); first=root/'first'; restored=root/'restored'; backup=root/'backup'
            result=run(c,first,'cpu',smoke=True,steps=5,backup_root=backup)
            self.assertEqual(result['global_step'],5)
            restore_latest(backup,restored)
            self.assertFalse((restored/'synthetic_cache').exists())
            result=run(c,restored,'cpu',smoke=True,steps=10,resume=restored/'last.pt',backup_root=backup)
            self.assertEqual(result['global_step'],10)
            state=load(restored/'last.pt')
            self.assertIn('scheduler',state); self.assertIn('optimizer',state); self.assertIn('rng',state)
            self.assertEqual(state['optimizer']['param_groups'][0]['lr'],c['training']['learning_rate'])
            self.assertTrue(result['checkpoint_reload_matches'])

    def test_constant_scheduler_epoch_resume_is_exact(self):
        from analysis.tsmixer.tests.test_tsmixer import inputs
        torch.set_num_threads(2); torch.manual_seed(2026)
        b=inputs(); b.update(target_cf=torch.rand(2,24),loss_mask=torch.ones(2,24,dtype=torch.bool),
                             capacity_kw=torch.full((2,),160.),smp=torch.full((2,24),100.),
                             k=torch.ones(2,24),group_complete=torch.ones(2,24,dtype=torch.bool))
        config=load_config('configs/tsmixer/colab.yaml')
        params=dict(hidden_dim=16,blocks=1,time_hidden=12,feature_hidden=24,dropout=.1,site_dropout=.1)
        m=TSMixer(**params); opt=torch.optim.AdamW(m.parameters(),lr=.001)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        train_epoch(m,[b],opt,scaler,config,torch.device('cpu')); sched.step()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt'; save(p,{'model':m.state_dict(),'optimizer':opt.state_dict(),
                                        'scheduler':sched.state_dict(),'rng':capture_rng()})
            train_epoch(m,[b],opt,scaler,config,torch.device('cpu')); sched.step()
            saved=load(p); n=TSMixer(**params); n.load_state_dict(saved['model'])
            no=torch.optim.AdamW(n.parameters(),lr=.001); ns=torch.optim.lr_scheduler.LambdaLR(no,lambda _:1.)
            no.load_state_dict(saved['optimizer']); ns.load_state_dict(saved['scheduler']); restore_rng(saved['rng'])
            train_epoch(n,[b],no,scaler,config,torch.device('cpu')); ns.step()
            for k,v in m.state_dict().items(): torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(sched.state_dict(),ns.state_dict()); self.assertEqual(ns.get_last_lr(),[.001])

    def test_notebook_is_unexecuted_valid_python_and_smoke_default(self):
        import ast
        notebook=json.loads(Path('train_colab.ipynb').read_text())
        self.assertEqual(notebook['nbformat'],4)
        self.assertEqual(len({x['id'] for x in notebook['cells']}),len(notebook['cells']))
        code=[]
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[]); self.assertIsNone(cell['execution_count'])
                source=''.join(cell['source']); ast.parse(source); code.append(source)
        self.assertIn('MODE = "smoke"',code[0]); self.assertIn('ALLOW_FULL_TRAIN = False',code[0])
        self.assertTrue(any('sys.path.insert(0, str(REPO_DIR))' in s for s in code))
        self.assertTrue(any('restore_latest' in s for s in code))
        self.assertTrue(any('"--backup-root"' in s for s in code))
        self.assertNotIn('/content/',Path('analysis/tsmixer/train.py').read_text())
        self.assertNotIn('google.colab',Path('analysis/tsmixer/runtime.py').read_text())


if __name__=='__main__': unittest.main()
