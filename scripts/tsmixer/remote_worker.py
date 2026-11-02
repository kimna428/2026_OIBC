"""Detached server job: call existing training wrapper, then existing dev CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys

# Works both as a server script and a namespace-package import in offline tests.
if __package__:
    from . import remote_agent as agent
else:
    import remote_agent as agent


def logged(c, args, file):
    with Path(file).open('xb') as stream:
        child = subprocess.Popen(args, cwd=c['REMOTE_REPO_DIR'], env=agent.env_for(c),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        return child.wait()


def run(job_file):
    job = agent.read_json(job_file)
    c = agent.validate_config(job['config'])
    dest = agent.job_path(c, job['experiment_id'])
    if Path(job_file).resolve() != dest/'job.json':
        raise ValueError('Job file must reside inside the configured experiment directory')
    state = agent.read_json(dest/'state.json')
    if state['status'] != 'queued':
        raise ValueError('Worker cannot restart or overwrite an existing job')
    state.update(status='running', phase='preflight', started_at=agent.now(),
                 pid=os.getpid(), process_start=agent.process_identity(os.getpid()))
    agent.atomic_json(dest/'state.json', state)
    try:
        if agent.git(c, 'rev-parse', 'HEAD') != job['expected_commit']:
            raise ValueError('Checkout changed after launch; worker will not train')
        if agent.git(c, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('Tracked files changed after launch; worker will not train')
        agent.preflight(c, require_cache=True)
        env = agent.execute(c, [c['REMOTE_PYTHON'], '-m', 'analysis.tsmixer',
                                'environment', '--device', 'cuda'])
        agent.atomic_json(dest/'environment.json', json.loads(env['stdout']))
        state.update(phase='training')
        agent.atomic_json(dest/'state.json', state)
        code = logged(c, ['bash', 'scripts/tsmixer/train_development.sh', '--full-train',
                          '--config', 'configs/tsmixer/base.yaml', '--output', str(dest/'train')],
                      dest/'train.log')
        state['train_exit_code'] = code
        if code:
            state.update(status='training_failed', exit_code=code)
        else:
            state.update(phase='evaluation')
            agent.atomic_json(dest/'state.json', state)
            code = logged(c, [c['REMOTE_PYTHON'], '-m', 'analysis.tsmixer', 'evaluate',
                              '--device', 'cuda', '--checkpoint', str(dest/'train/best.pt'),
                              '--cache-root', c['REMOTE_CACHE_ROOT'], '--output', str(dest/'eval_dev')],
                          dest/'evaluate.log')
            state.update(evaluation_exit_code=code, exit_code=code,
                         status='evaluation_failed' if code else 'completed')
    except BaseException as exc:
        # SIGKILL/power loss cannot be caught; remote_status reports lost jobs
        # rather than inventing an exit code or overwriting their files.
        state.update(status='worker_failed', error=f'{type(exc).__name__}: {exc}', exit_code=1)
    finally:
        state['finished_at'] = agent.now()
        agent.atomic_json(dest/'state.json', state)
        agent.write_summary(dest, state)
    return state['exit_code']


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
