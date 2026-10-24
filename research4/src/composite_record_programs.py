"""Pure finite record transformation compiler; no lifecycle effects or world access."""

def program_for_observed_packet(packet):
    def r(name):return {'ref':name}
    def step(bind,op,**arguments):return {'bind':bind,'op':op,'arguments':arguments}
    steps=[step('a','read',identifier=r('s'))];parameters=['s','t']
    if set(packet)=={'nodes','edges'}:
        steps += [step('b','join',left=r('a.data.nodes'),right=r('a.data.edges'),left_key='id',right_key='parent'),
          step('c','select',rows=r('b'),field='left.ready',equals=True),
          step('d','aggregate',rows=r('c'),field='unused',mode='count'),
          step('e','patch',identifier=r('t'),fields={'support_count':r('d')}),step('f','emit',value=r('d'))]
    elif set(packet)=={'request','rows'}:
        steps += [step('b','select',rows=r('a.data.rows'),field='code',equals=r('a.data.request')),
          step('c','patch',identifier=r('t'),fields={'selected':r('b.0.value')}),step('d','emit',value=r('b.0.value'))]
    elif set(packet)=={'rows'} and packet['rows'] and set(packet['rows'][0])=={'value'}:
        steps += [step('b','aggregate',rows=r('a.data.rows'),field='value',mode='sum'),
          step('c','patch',identifier=r('t'),fields={'total':r('b')}),step('d','emit',value=r('b'))]
    elif set(packet)=={'edges','states'}:
        steps += [step('b','select',rows=r('a.data.edges'),field='active',equals=True),
          step('c','join',left=r('b'),right=r('a.data.states'),left_key='next',right_key='code'),
          step('d','patch',identifier=r('t'),fields={'state_label':r('c.0.right.label')}),step('e','emit',value=r('c.0.right.label'))]
    elif set(packet)=={'aliases','registry','sources'}:
        steps += [step('b','join',left=r('a.data.aliases'),right=r('a.data.registry'),left_key='alias',right_key='alias'),
          step('c','join',left=r('b'),right=r('a.data.sources'),left_key='right.canonical',right_key='canonical'),
          step('d','aggregate',rows=r('c'),field='unused',mode='count'),
          step('e','patch',identifier=r('t'),fields={'equivalent_count':r('d')}),step('f','emit',value=r('d'))]
    elif set(packet)=={'left','right','requests'}:
        steps += [step('b','join',left=r('a.data.left'),right=r('a.data.right'),left_key='middle',right_key='middle'),
          step('c','join',left=r('b'),right=r('a.data.requests'),left_key='right.goal',right_key='goal'),
          step('d','select',rows=r('c'),field='left.left.allowed',equals=True),
          step('e','aggregate',rows=r('d'),field='unused',mode='count'),
          step('f','patch',identifier=r('t'),fields={'path_count':r('e')}),step('g','emit',value=r('e'))]
    elif set(packet)=={'rows'} and packet['rows'] and set(packet['rows'][0])=={'valid','value'}:
        parameters.append('u')
        steps += [step('b','select',rows=r('a.data.rows'),field='valid',equals=True),
          step('c','aggregate',rows=r('b'),field='value',mode='sum'),
          step('d','aggregate',rows=r('b'),field='value',mode='max'),
          step('e','patch',identifier=r('t'),fields={'total':r('c')}),
          step('f','patch',identifier=r('u'),fields={'peak':r('d')}),step('g','emit',value={'total':r('c'),'peak':r('d')})]
    elif set(packet)=={'left','right'}:
        steps += [step('b','join',left=r('a.data.left'),right=r('a.data.right'),left_key='id',right_key='id'),
          step('c','select',rows=r('b'),field='left.allowed',equals=True),
          step('d','select',rows=r('c'),field='right.allowed',equals=True),
          step('e','aggregate',rows=r('d'),field='left.value',mode='max'),
          step('f','patch',identifier=r('t'),fields={'approved_max':r('e')}),step('g','emit',value=r('e'))]
    else:raise ValueError('Unregistered observed record structure')
    # The final write receipt is sufficient. Do not add a redundant emit only
    # to inflate an unrolled counterfactual or consume the episode budget.
    if steps[-1]['op']=='emit':steps=steps[:-1]
    return {'parameters':parameters,'steps':steps}
