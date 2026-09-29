"""Apply a reviewed startup-limit guard to a copied isolation boundary."""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected boundary source structure')
    return source.replace(old, new, 1)


def worker(source):
    source = replace_once(source, 'import subprocess\n', 'import subprocess\nimport time\n')
    source = replace_once(source, "    args = parser.parse_args()\n", """    parser.add_argument('--memory', type=int, required=True)
    parser.add_argument('--tasks', type=int, required=True)
    parser.add_argument('--cpus', type=int, required=True)
    args = parser.parse_args()
""")
    source = replace_once(source, "    fd = seccomp_fd()\n", """    if args.cpus not in (4, 6) or not 128 * 1024**2 <= args.memory <= 12 * 1024**3 or not 16 <= args.tasks <= 64:
        raise RuntimeError('Invalid trusted resource bounds')
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    cgroup_root = Path('/sys/fs/cgroup')
    cgroup = (cgroup_root / relative.lstrip('/')).resolve()
    if not cgroup.is_relative_to(cgroup_root):
        raise RuntimeError('Unexpected cgroup path')
    matched = False
    for _ in range(20):
        limits = {name: (cgroup / name).read_text().strip()
                  for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
        cpu = limits['cpu.max'].split()
        matched = (limits['memory.max'] == str(args.memory) and limits['memory.swap.max'] == '0'
                   and limits['pids.max'] == str(args.tasks) and len(cpu) == 2
                   and cpu[0] != 'max' and int(cpu[0]) == args.cpus * int(cpu[1]))
        if matched:
            break
        time.sleep(0.05)
    if not matched:
        raise RuntimeError('Resource limits not verified; refuse to start worker payload')
    import json
    (output / 'entry_limits.json').write_text(json.dumps({'passed': True, 'limits': limits,
         'payload_started_only_after_verification': True}, indent=2))
    fd = seccomp_fd()
""")
    return source


def supervisor(source, cpus):
    source = replace_once(source, "                '--spec', spec]\n", """                '--spec', spec, '--memory', str(memory), '--tasks', str(tasks), '--cpus', '%s']
""" % cpus)
    source = replace_once(source, '    actual_limits = {}\n', '    actual_limits = {}\n    startup_limit_samples = []\n')
    old = """                    actual_limits = {name: (cgroup / name).read_text().strip()
                                     for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
"""
    new = """                    sampled = {name: (cgroup / name).read_text().strip()
                               for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
                    cpu = sampled['cpu.max'].split()
                    matched = (sampled['memory.max'] == str(memory) and sampled['memory.swap.max'] == '0'
                               and sampled['pids.max'] == str(tasks) and len(cpu) == 2
                               and cpu[0] != 'max' and int(cpu[0]) == %s * int(cpu[1]))
                    if matched:
                        actual_limits = sampled
                    elif len(startup_limit_samples) < 10:
                        startup_limit_samples.append(sampled)
""" % cpus
    source = replace_once(source, old, new)
    source = replace_once(source, "                   'requested_limits': properties, 'actual_cgroup_limits': actual_limits,\n",
                          "                   'requested_limits': properties, 'actual_cgroup_limits': actual_limits,\n"
                          "                   'startup_limit_samples': startup_limit_samples,\n")
    source = replace_once(source, "        artifacts = ROOT / 'runs' / (prefix + '-' + label) / 'artifacts'\n",
        "        artifacts = ROOT / 'runs' / (prefix + '-' + label) / 'artifacts'\n"
        "        entry = artifacts / 'entry_limits.json'\n"
        "        checks[label + '_entry_guard'] = (entry.exists() and json.loads(entry.read_text()).get('passed') is True)\n")
    return source
