"""Offline-only remote automation tests. SSH/rsync/GPU calls are mocked."""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.tsmixer import remote_agent as a, remote_control as c, remote_worker as w

COMMIT = 'a'*40


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.repo = self.base/'repo'
        (self.repo/'.git').mkdir(parents=True)
        (self.repo/'configs/tsmixer').mkdir(parents=True)
        (self.repo/'configs/tsmixer/base.yaml').write_text('name: fixture\n')
        self.cfg = {
            'SSH_HOST': 'mock-gpu', 'REMOTE_REPO_DIR': str(self.repo),
            'REMOTE_BRANCH': 'feature/tsmixer-v0-mac-cuda',
            'REMOTE_DATA_ROOT': str(self.base/'data'),
            'REMOTE_CACHE_ROOT': str(self.base/'cache'),
            'REMOTE_PYTHON': '/mock/venv/bin/python',
            'REMOTE_RUN_ROOT': str(self.base/'runs')}
        self.info = {'branch': self.cfg['REMOTE_BRANCH'], 'commit': COMMIT,
                     'tracked_changes': '', 'work_tree': '?? personal.txt'}

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, eid='job1', status='queued'):
        p = a.job_path(self.cfg, eid)
        p.mkdir(parents=True)
        state = {'experiment_id': eid, 'status': status, 'phase': 'training',
                 'commit': COMMIT, 'train_exit_code': None,
                 'evaluation_exit_code': None, 'exit_code': None}
        a.atomic_json(p/'job.json', {'config': self.cfg, 'experiment_id': eid, 'expected_commit': COMMIT})
        a.atomic_json(p/'state.json', state)
        return p

    def finished_bundle(self):
        p = self.job(status='completed')
        a.atomic_json(p/'state.json', {**a.read_json(p/'state.json'), 'train_exit_code': 0,
                                      'evaluation_exit_code': 0, 'exit_code': 0})
        (p/'train').mkdir(); (p/'eval_dev').mkdir()
        (p/'config.yaml').write_text('seed: 2026\n')
        (p/'train/best.pt').write_bytes(b'fake checkpoint, never copied')
        (p/'train/last.pt').write_bytes(b'fake last checkpoint')
        (p/'train/run.json').write_text('{"config":{"model":{"hidden_dim":128}}}')
        (p/'train/epochs.jsonl').write_text('{"epoch":1,"loss":0.1}\n')
        (p/'eval_dev/metrics.json').write_text('{"split":"dev","metrics":{"site_cf_mae":0.02}}')
        (p/'eval_dev/site_and_cohort_metrics.csv').write_text('site_id,cf_mae\nPV001,0.02\n')
        (p/'eval_dev/predictions.npy').write_bytes(b'never copy')
        (p/'eval_dev/settlement_ledger.parquet').write_bytes(b'never copy')
        (p/'train.log').write_text('training finished\n')
        a.write_summary(p, a.read_json(p/'state.json'))
        with patch.object(a, 'live', return_value=False):
            return a.bundle(self.cfg, 'job1')


class ConfigTests(Fixture):
    def test_config_required_and_no_password(self):
        self.assertEqual(a.validate_config(self.cfg), self.cfg)
        for key in a.REQUIRED:
            bad = dict(self.cfg); del bad[key]
            with self.assertRaises(ValueError): a.validate_config(bad)
        with self.assertRaises(ValueError): a.validate_config({**self.cfg, 'SSH_PASSWORD': 'not-accepted'})

    def test_host_options_and_injection_rejected(self):
        for host in ('-oProxyCommand=touch x', 'gpu;touch x', 'gpu$(id)', 'gpu\nnext', 'host:22'):
            with self.subTest(host=host), self.assertRaises(ValueError):
                a.validate_config({**self.cfg, 'SSH_HOST': host})
        a.validate_config({**self.cfg, 'SSH_HOST': 'user@gpu.example'})

    def test_path_and_branch_safety(self):
        for path in ('/', '../data', '/srv/../data', '/srv/cache/', '/srv/a\nb'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                a.validate_config({**self.cfg, 'REMOTE_DATA_ROOT': path})
        for branch in ('--help', '../main', 'main;id', 'a..b', 'main.lock'):
            with self.assertRaises(ValueError):
                a.validate_config({**self.cfg, 'REMOTE_BRANCH': branch})
        for path in (self.cfg['REMOTE_REPO_DIR'], self.cfg['REMOTE_DATA_ROOT'], str(self.base)):
            with self.assertRaises(ValueError): a.validate_config({**self.cfg, 'REMOTE_RUN_ROOT': path})

    def test_config_json_is_not_shell_sourced_and_env_overrides(self):
        file = self.base/'settings.json'
        file.write_text(json.dumps(self.cfg))
        with patch.dict(os.environ, {'SSH_HOST': 'override-alias'}, clear=True):
            self.assertEqual(c.config(file)['SSH_HOST'], 'override-alias')
        file.write_text('SSH_HOST=$(touch forbidden)')
        with self.assertRaises(json.JSONDecodeError): c.config(file)
        self.assertFalse((self.base/'forbidden').exists())

    def test_experiment_id_validation(self):
        for eid in ('../x', '/tmp/x', '.hidden', 'a/b', 'a;b', '', 'a'*81):
            with self.assertRaises(ValueError): a.experiment_id(eid)
        self.assertEqual(a.experiment_id('tsmixer_20261102T120000Z_ab12'), 'tsmixer_20261102T120000Z_ab12')


class GitAndTransportTests(Fixture):
    def test_mac_requires_pushed_commit_before_ssh(self):
        def git(*args):
            if args[0] == 'branch': return self.cfg['REMOTE_BRANCH']
            if args[0] == 'status': return ''
            if args[0] == 'rev-parse': return COMMIT
            return 'b'*40+'\trefs/heads/'+self.cfg['REMOTE_BRANCH']
        with patch.object(c, 'local_git', side_effect=git), patch.object(c, 'request') as ssh:
            with self.assertRaisesRegex(ValueError, 'not the pushed'): c.pushed_commit(self.cfg)
            ssh.assert_not_called()

    def test_uncommitted_controller_source_blocks_launch(self):
        def git(*args):
            if args[0]=='branch': return self.cfg['REMOTE_BRANCH']
            if args[0]=='status': return ''
            if args[0]=='ls-remote': return COMMIT+'\trefs/heads/'+self.cfg['REMOTE_BRANCH']
            if args[0]=='hash-object': return 'b'*40
            return COMMIT
        with patch.object(c,'local_git',side_effect=git):
            with self.assertRaisesRegex(ValueError,'controller sources'): c.pushed_commit(self.cfg)

    def test_remote_commit_mismatch_no_merge(self):
        commands = []
        def git(cfg, *args):
            commands.append(args)
            return 'b'*40 if args == ('rev-parse', 'FETCH_HEAD') else ''
        with patch.object(a, 'metadata', return_value=self.info), patch.object(a, 'git', side_effect=git):
            with self.assertRaisesRegex(ValueError, 'commit mismatch'): a.synchronize(self.cfg, COMMIT)
        self.assertFalse(any(x[0] == 'merge' for x in commands))

    def test_dirty_or_wrong_branch_no_fetch(self):
        for info in ({**self.info, 'tracked_changes': ' M model.py'}, {**self.info, 'branch': 'main'}):
            with patch.object(a, 'metadata', return_value=info), patch.object(a, 'git') as git:
                with self.assertRaises(ValueError): a.synchronize(self.cfg, COMMIT)
                git.assert_not_called()

    def test_fast_forward_only_no_reset(self):
        commands = []
        def git(cfg, *args):
            commands.append(args)
            return COMMIT if args[0] == 'rev-parse' else ''
        with patch.object(a, 'metadata', return_value=self.info), patch.object(a, 'git', side_effect=git):
            a.synchronize(self.cfg, COMMIT)
        self.assertIn(('merge', '--ff-only', '--no-edit', COMMIT), commands)
        self.assertFalse(any(x[0] in ('reset', 'clean', 'push', 'rebase') for x in commands))

    def test_ssh_quoting_and_json_arguments(self):
        special = str(self.base/"data space 'quoted';$(touch bad)")
        cfg = {**self.cfg, 'REMOTE_DATA_ROOT': special}
        a.validate_config(cfg)
        seen = {}
        def mock_run(args, **kwargs):
            seen.update(args=args, kwargs=kwargs)
            return subprocess.CompletedProcess(args, 0, '{"ok":true,"result":{"status":"mock"}}', '')
        with patch.object(c.subprocess, 'run', side_effect=mock_run):
            self.assertEqual(c.request(cfg, 'setup')['status'], 'mock')
        args = seen['args']
        self.assertEqual(args[0], 'ssh'); self.assertIn('BatchMode=yes', args)
        command = shlex.split(args[-1])
        self.assertEqual(command[:2], ['python3', '-c'])
        self.assertNotIn(special, command[2])
        self.assertEqual(json.loads(seen['kwargs']['input'])['config']['REMOTE_DATA_ROOT'], special)
        self.assertNotIn('shell', seen['kwargs'])

    def test_ssh_failure_checked(self):
        with patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([],255,'','mock SSH failed')):
            with self.assertRaisesRegex(RuntimeError,'SSH/protocol failed'): c.request(self.cfg,'status')

    def test_setup_default_inspection_only(self):
        with patch.object(a,'metadata',return_value=self.info), patch.object(a,'preflight',return_value={'returncode':0}), \
                patch.object(a,'execute') as execute:
            result=a.setup(self.cfg,{'expected_commit':COMMIT,'options':[]})
        execute.assert_not_called()
        self.assertFalse(Path(self.cfg['REMOTE_RUN_ROOT']).exists())
        self.assertFalse(result['cache_present'])
        self.assertIn('--prepare-cache',result['next'])

    def test_setup_missing_environment_provides_guidance_no_install(self):
        with patch.object(a,'metadata',return_value=self.info), patch.object(a,'preflight',side_effect=FileNotFoundError('python missing')):
            result=a.setup(self.cfg,{'expected_commit':COMMIT,'options':[]})
        self.assertFalse(result['checks_passed'])
        self.assertTrue(any('install_cuda.sh' in x for x in result['prepare_commands']))
        self.assertFalse(Path(self.cfg['REMOTE_RUN_ROOT']).exists())

    def test_setup_sync_is_explicit_and_fast_forward_only(self):
        with patch.object(a,'metadata',return_value=self.info), patch.object(a,'preflight',return_value={}), \
                patch.object(a,'synchronize',return_value=self.info) as sync:
            result=a.setup(self.cfg,{'expected_commit':COMMIT,'options':['sync']})
        sync.assert_called_once_with(self.cfg,COMMIT)
        self.assertTrue(result['git_sync_performed'])

    def test_setup_tests_and_smoke_only_on_explicit_options(self):
        with patch.object(a,'metadata',return_value=self.info), patch.object(a,'preflight',return_value={}), \
                patch.object(a,'execute',return_value={'returncode':0}) as execute:
            a.setup(self.cfg,{'expected_commit':COMMIT,'options':['tests','smoke']})
        calls=[x.args[1] for x in execute.call_args_list]
        self.assertEqual(len(calls),2)
        self.assertIn('unittest',calls[0])
        self.assertIn('scripts/tsmixer/smoke.sh',calls[1])
        self.assertFalse(any('--full-train' in x for x in calls))
        self.assertFalse(any('install_cuda.sh' in ' '.join(x) for x in calls))

    def test_actual_local_git_fast_forward_and_divergence(self):
        # All remotes are local tempfile paths. No SSH, GitHub or network use.
        origin=self.base/'origin.git'; sender=self.base/'sender'; server=self.base/'server'
        branch=self.cfg['REMOTE_BRANCH']
        def git(path,*args):
            p=subprocess.run(['git',*args],cwd=path,text=True,capture_output=True,check=True)
            return p.stdout.strip()
        git(self.base,'init','--bare',str(origin))
        git(self.base,'init','-b',branch,str(sender))
        for path in [sender]:
            git(path,'config','user.email','tests@example.invalid'); git(path,'config','user.name','Offline Tests')
        (sender/'tracked.txt').write_text('first')
        git(sender,'add','tracked.txt'); git(sender,'commit','-m','first')
        git(sender,'remote','add','origin',str(origin)); git(sender,'push','origin',branch)
        git(self.base,'clone','--branch',branch,str(origin),str(server))
        git(server,'config','user.email','tests@example.invalid'); git(server,'config','user.name','Offline Tests')
        (server/'personal.txt').write_text('keep user data')
        (sender/'tracked.txt').write_text('second')
        git(sender,'commit','-am','second'); git(sender,'push','origin',branch)
        head=git(sender,'rev-parse','HEAD')
        cfg={**self.cfg,'REMOTE_REPO_DIR':str(server)}
        result=a.synchronize(cfg,head)
        self.assertEqual(result['commit'],head)
        self.assertEqual((server/'personal.txt').read_text(),'keep user data')
        (server/'local.txt').write_text('local branch commit')
        git(server,'add','local.txt'); git(server,'commit','-m','server local work')
        before=git(server,'rev-parse','HEAD')
        (sender/'tracked.txt').write_text('third')
        git(sender,'commit','-am','third'); git(sender,'push','origin',branch)
        with self.assertRaises(RuntimeError): a.synchronize(cfg,git(sender,'rev-parse','HEAD'))
        self.assertEqual(git(server,'rev-parse','HEAD'),before)
        self.assertTrue((server/'local.txt').is_file())

    def test_launch_duplicate_id_no_update(self):
        self.job()
        with patch.object(a,'synchronize') as sync, patch.object(a,'live',return_value=False):
            with self.assertRaisesRegex(FileExistsError,'already exists'):
                a.launch(self.cfg,{'experiment_id':'job1','expected_commit':COMMIT})
            sync.assert_not_called()

    def test_active_job_blocks_checkout(self):
        self.job()
        with patch.object(a,'live',return_value=True), patch.object(a,'synchronize') as sync:
            with self.assertRaisesRegex(ValueError,'active'):
                a.launch(self.cfg,{'experiment_id':'job2','expected_commit':COMMIT})
            sync.assert_not_called()

    def test_detached_launch_passes_no_shell(self):
        with patch.object(a,'synchronize',return_value=self.info), patch.object(a,'preflight'), \
                patch.object(a,'execute'), patch.object(a.subprocess,'Popen',return_value=Mock(pid=987654)) as popen, \
                patch.object(a,'process_identity',return_value='start-token'):
            result=a.launch(self.cfg,{'experiment_id':'launch1','expected_commit':COMMIT})
        self.assertEqual(result['status'],'submitted')
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(popen.call_args.kwargs['stdin'],subprocess.DEVNULL)
        self.assertNotIn('shell',popen.call_args.kwargs)
        self.assertEqual(a.job_state(a.job_path(self.cfg,'launch1'))['process_start'],'start-token')


class StatusAndWorkerTests(Fixture):
    def test_running_failed_complete_and_lost_statuses(self):
        p=self.job()
        (p/'train').mkdir()
        (p/'train/epochs.jsonl').write_text('{"epoch":3,"loss":0.1,"development":{"site_cf_mae":0.02}}\n')
        (p/'train.log').write_text(''.join(f'line {i}\n' for i in range(30)))
        for stored,active,expected in [('running',True,'running'),('completed',False,'completed'),
                                      ('training_failed',False,'training_failed'),('running',False,'interrupted_or_lost')]:
            state=a.read_json(p/'state.json'); state['status']=stored
            a.atomic_json(p/'state.json',state)
            with patch.object(a,'live',return_value=active), patch.object(a,'execute',return_value={'returncode':0,'stdout':'mock GPU'}):
                result=a.status(self.cfg,'job1')
            self.assertEqual(result['effective_status'],expected)
            self.assertEqual(result['recent_epoch'],3)
            self.assertEqual(len(result['recent_log']),20)
            self.assertIsNone(result['validation_loss'])
            self.assertEqual(result['development']['site_cf_mae'],0.02)

    def test_no_experiments(self):
        self.assertEqual(a.status(self.cfg)['status'],'no_experiments')

    def test_worker_training_and_eval_states(self):
        for mode,expected in [('ok','completed'),('train_fail','training_failed'),('eval_fail','evaluation_failed')]:
            with self.subTest(mode=mode):
                p=self.job(mode)
                calls=[]
                def logged(cfg,args,file):
                    calls.append(args); Path(file).write_text('mock log\n')
                    if '--full-train' in args:
                        (p/'train').mkdir()
                        (p/'train/run.json').write_text('{"config":{"model":{"hidden_dim":128}}}')
                        (p/'train/best.pt').write_bytes(b'mock')
                        return 7 if mode=='train_fail' else 0
                    (p/'eval_dev').mkdir()
                    (p/'eval_dev/metrics.json').write_text('{"split":"dev","metrics":{"site_cf_mae":0.02}}')
                    return 9 if mode=='eval_fail' else 0
                with patch.object(a,'git',side_effect=lambda cfg,*args: COMMIT if args[0]=='rev-parse' else ''), \
                        patch.object(a,'preflight'), patch.object(a,'execute',return_value={'stdout':'{"device":"cuda"}'}), \
                        patch.object(w,'logged',side_effect=logged):
                    code=w.run(p/'job.json')
                state=a.read_json(p/'state.json')
                self.assertEqual(state['status'],expected)
                self.assertEqual(code,0 if mode=='ok' else (7 if mode=='train_fail' else 9))
                self.assertIn('--full-train',calls[0])
                if mode=='train_fail': self.assertEqual(len(calls),1)
                else:
                    self.assertIn('evaluate',calls[1]); self.assertIn('eval_dev',calls[1][-1])
                    self.assertEqual(state['train_exit_code'],0)
                self.assertNotIn('holdout',' '.join(sum(calls,[])))
                self.assertNotIn('predict',' '.join(sum(calls,[])))
                self.assertIn('기존 weighted L1',(p/'summary.md').read_text())

    def test_worker_commit_changed_never_trains(self):
        p=self.job()
        with patch.object(a,'git',return_value='b'*40), patch.object(w,'logged') as logged:
            self.assertEqual(w.run(p/'job.json'),1)
        logged.assert_not_called()
        self.assertEqual(a.read_json(p/'state.json')['status'],'worker_failed')


class FetchTests(Fixture):
    def test_bundle_excludes_large_files_and_records_checkpoint_hash(self):
        manifest=self.finished_bundle()
        names={f['path'] for f in manifest['files']}
        self.assertIn('eval_dev/metrics.json',names)
        self.assertFalse(any(x.endswith(('.pt','.npy','.parquet')) for x in names))
        cp=a.read_json(Path(manifest['artifact_root'])/'checkpoint_manifest.json')
        self.assertFalse(cp['copied']); self.assertEqual(len(cp['checkpoints']),2)
        self.assertEqual(len(cp['checkpoints'][0]['sha256']),64)

    def test_fetch_running_rejected(self):
        self.job(status='running')
        with patch.object(a,'live',return_value=True):
            with self.assertRaisesRegex(ValueError,'finished'): a.bundle(self.cfg,'job1')

    def test_remote_result_symlink_rejected(self):
        p=self.job(status='completed')
        (p/'train.log').symlink_to(self.repo/'configs/tsmixer/base.yaml')
        with patch.object(a,'live',return_value=False):
            with self.assertRaisesRegex(ValueError,'Unsafe export'): a.bundle(self.cfg,'job1')

    def test_export_parent_symlink_rejected(self):
        p=self.job(status='completed')
        (p/'exports').symlink_to(self.repo,target_is_directory=True)
        with patch.object(a,'live',return_value=False):
            with self.assertRaisesRegex(ValueError,'Export directory'): a.bundle(self.cfg,'job1')

    def test_unsafe_manifest_paths_duplicates_checkpoint_rejected(self):
        manifest=self.finished_bundle()
        for name in ('../oops','/tmp/oops','train/best.pt','eval_dev/predictions.npy'):
            bad=copy.deepcopy(manifest); bad['files'][0]['path']=name
            with self.assertRaises(ValueError): c.validate_manifest(self.cfg,'job1',bad)
        bad=copy.deepcopy(manifest); bad['files'].append(bad['files'][0])
        with self.assertRaises(ValueError): c.validate_manifest(self.cfg,'job1',bad)
        bad=copy.deepcopy(manifest); bad['artifact_root']='/tmp/unsafe'
        with self.assertRaises(ValueError): c.validate_manifest(self.cfg,'job1',bad)

    def test_rsync_success_checksum_and_quoting(self):
        # Include apostrophe/space/shell metacharacters in a real local fixture
        # directory. Only shutil simulates transfer; rsync/ssh are never run.
        self.cfg['REMOTE_RUN_ROOT']=str(self.base/"runs space 'quoted';$literal")
        manifest=self.finished_bundle()
        target=self.base/'fetched'; commands=[]
        def rsync(args,**kwargs):
            commands.append(args)
            stage=Path(args[-1])
            for f in manifest['files']:
                src=Path(manifest['artifact_root'])/f['path']; dest=stage/f['path']
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
            return subprocess.CompletedProcess(args,0,'','')
        with patch.object(c,'request',return_value=manifest), patch.object(c.subprocess,'run',side_effect=rsync):
            result=c.fetch(self.cfg,'job1',target)
        self.assertEqual(result['verified_files'],len(manifest['files']))
        quoted=commands[0][-2].split(':',1)[1]
        self.assertEqual(shlex.split(quoted),[manifest['artifact_root']+'/'])
        self.assertTrue((target/'job1/transfer_manifest.json').is_file())
        self.assertFalse(list((target/'job1').rglob('*.pt')))

    def test_rsync_failure_keeps_partial_no_success_result(self):
        manifest=self.finished_bundle(); target=self.base/'fetch_failed'
        with patch.object(c,'request',return_value=manifest), \
                patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],23,'','mock rsync failure')):
            with self.assertRaisesRegex(RuntimeError,'rsync failed'): c.fetch(self.cfg,'job1',target)
        self.assertFalse((target/'job1').exists())
        self.assertTrue(list(target.glob('.job1.partial_*')))

    def test_corrupt_transfer_rejected(self):
        manifest=self.finished_bundle(); target=self.base/'corrupt'
        with patch.object(c,'request',return_value=manifest), \
                patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')):
            with self.assertRaisesRegex(ValueError,'Integrity check failed'): c.fetch(self.cfg,'job1',target)
        self.assertFalse((target/'job1').exists())

    def test_local_result_not_overwritten(self):
        target=self.base/'fetch'; (target/'job1').mkdir(parents=True)
        (target/'job1/user.txt').write_text('preserve')
        with patch.object(c,'request') as request:
            with self.assertRaises(FileExistsError): c.fetch(self.cfg,'job1',target)
            request.assert_not_called()
        self.assertEqual((target/'job1/user.txt').read_text(),'preserve')

    def test_local_result_symlink_rejected(self):
        target=self.base/'link'; target.symlink_to(self.repo,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlinks'): c.fetch(self.cfg,'job1',target)


class ShellBoundaryTests(Fixture):
    def test_wrappers_with_fake_ssh_and_fake_git_no_network(self):
        import sys
        bindir=self.base/'bin'; bindir.mkdir()
        configfile=self.base/'private.json'; configfile.write_text(json.dumps(self.cfg))
        requests=self.base/'requests.jsonl'
        fake=self.base/'fake_tool.py'
        fake.write_text("""import json,sys
from pathlib import Path
mode=sys.argv[1]
if mode=='ssh':
 request=json.load(sys.stdin)
 with Path(sys.argv[2]).open('a') as f: f.write(json.dumps(request)+'\\n')
 op=request['operation']
 print(json.dumps({'ok':True,'result':{'checks_passed':True,'operation':op,'experiment_id':request.get('experiment_id')}}))
else:
 args=sys.argv[2:]
 if args[:2]==['branch','--show-current']: print('feature/tsmixer-v0-mac-cuda')
 elif args[0]=='status': pass
 elif args[0]=='rev-parse': print('a'*40)
 elif args[0]=='hash-object': print('a'*40)
 elif args[0]=='ls-remote': print('a'*40+'\\trefs/heads/feature/tsmixer-v0-mac-cuda')
 else: raise SystemExit('Unexpected fake Git command')
""")
        for tool in ('ssh','git'):
            file=bindir/tool
            file.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(fake))+' '+tool+' '+
                            (shlex.quote(str(requests))+' ' if tool=='ssh' else '')+'"$@"\n')
            file.chmod(0o700)
        env={k:v for k,v in os.environ.items() if k not in a.REQUIRED+a.OPTIONAL}
        env.update(PATH=str(bindir)+os.pathsep+env['PATH'],OIBC_LOCAL_PYTHON=sys.executable)
        for op,extra in [('setup',[]),('status',[]),('launch',['offline001'])]:
            command=['bash',str(c.ROOT/'scripts/tsmixer'/('remote_'+op+'.sh')),*extra,'--config',str(configfile)]
            p=subprocess.run(command,cwd=c.ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['operation'],op)
        recorded=[json.loads(x) for x in requests.read_text().splitlines()]
        self.assertEqual(len(recorded),3)
        self.assertEqual(recorded[0]['options'],[])
        self.assertEqual(recorded[-1]['expected_commit'],COMMIT)
        self.assertEqual(recorded[-1]['experiment_id'],'offline001')
        # No configuration and wrong-directory failures must not reach fake SSH.
        bad=self.base/'bad.json'; bad.write_text('{}')
        for command,cwd in [(['bash',str(c.ROOT/'scripts/tsmixer/remote_setup.sh'),'--config',str(bad)],c.ROOT),
                            (['bash',str(c.ROOT/'scripts/tsmixer/remote_setup.sh')],self.base)]:
            p=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,2)
        self.assertEqual(len(requests.read_text().splitlines()),3)

    @unittest.skipUnless(shutil.which('rsync'),'rsync is not installed')
    def test_real_local_rsync_flags_and_allowlist(self):
        # Local directories only: validates the Mac rsync flags, never SSH.
        source=self.base/"source space 'quoted'"; dest=self.base/'destination'
        (source/'train').mkdir(parents=True); dest.mkdir()
        (source/'train/run.json').write_text('{"fixture":true}')
        (source/'train/best.pt').write_bytes(b'never copy')
        filelist=self.base/'files'; filelist.write_text('train/run.json\n')
        p=subprocess.run(['rsync','-rt','--checksum','--files-from='+str(filelist),
                          '-e','ssh -T -o BatchMode=yes','--',str(source)+'/',str(dest)+'/'],
                         capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual((dest/'train/run.json').read_text(),'{"fixture":true}')
        self.assertFalse((dest/'train/best.pt').exists())

    def test_all_shell_syntax(self):
        for file in (c.ROOT/'scripts/tsmixer').glob('*.sh'):
            p=subprocess.run(['bash','-n',str(file)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':
    unittest.main()
