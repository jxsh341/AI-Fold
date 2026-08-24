"""Silent no-op audit: trace every knob to its consumption site."""
import re, pathlib

root = pathlib.Path('aifold_discovery')
live = root / 'live'

checks = []

# --- 1. CLI flags -> consumption sites ---
runner = pathlib.Path('run_live_discovery.py').read_text(encoding='utf-8-sig')
flags = ['generations', 'pop_size', 'group_size', 'n_envs', 'max_calls',
         'max_concurrency', 'difficulty', 'pure_baseline']
for f in flags:
    uses = len(re.findall(rf'\bargs\.{f}\b', runner))
    checks.append((f'CLI --{f}', 'args.' + f in runner,
                   f'{uses} refs in runner'))

# group_size propagation chain (the bug we just caught)
reg = (live / 'registry_live.py').read_text(encoding='utf-8-sig')
checks.append(('spec.group_size -> make_env',
               'group_size=gs' in reg, 'registry closure'))
env_src = (live / 'environments.py').read_text(encoding='utf-8-sig')
checks.append(('make_env sets env.group_size',
               'env.group_size = group_size' in env_src, 'instance attr'))

# --- 2. Genome genes -> scaffold consumption ---
sc = (live / 'scaffolding.py').read_text(encoding='utf-8-sig')
genes = {
    'verifier_enabled': 'model.verifier_enabled',
    'episodic_memory': 'memory.episodic_memory',
    'semantic_memory': 'memory.semantic_memory',
    'working_memory_size': 'working_memory_size',
    'planning.decomposition': 'planning.decomposition',
    'search_algorithm(beam)': 'search_algorithm',
    'critic_enabled': 'control.critic_enabled',
    'retry_on_failure': 'retry_on_failure',
    'router_type': 'router_type',
}
for name, needle in genes.items():
    checks.append((f'gene {name}', needle in sc, 'consumed in scaffolding'))
checks.append(('gene tools[code]', '"code" in self.g.tools.enabled_tools' in sc
               or "'code' in self.g.tools.enabled_tools" in sc,
               'sandbox budget'))
# browser: expected NOT found -> latent silent gene
checks.append(('gene tools[browser]', 'browser' in sc,
               'SILENT IF ABSENT - known gap'))

# raise_tool_budget w/o code tool: check mutator gating
mut = (root / 'evolution' / 'mutation.py').read_text(encoding='utf-8-sig')
seg = mut.split('def _m_raise_tool_budget')[1][:400]
checks.append(('raise_tool_budget gated on code tool',
               'enabled_tools' in seg, 'needs gate if absent'))

for name, ok, note in checks:
    status = 'OK ' if ok else 'GAP'
    print(f'[{status}] {name:42s} {note}')

