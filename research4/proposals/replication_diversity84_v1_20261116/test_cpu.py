"""Meaningful CPU causal, information-boundary and menu-choice checks."""
import copy
import unittest
import generator as g
import interface as v
import build


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.episodes,cls.rows=g.replay_episodes();cls.world=g.load_world()

    def test_positive_counts_complete_paths_and_final_live_current_own_state(self):
        self.assertEqual((sum(e['split']=='TRAIN'for e in self.episodes),sum(e['split']=='DEV'for e in self.episodes)),(21,6))
        self.assertEqual((sum(r['split']=='TRAIN'for r in self.rows),sum(r['split']=='DEV'for r in self.rows)),(84,24))
        for e in self.episodes:
            self.assertFalse(e['unlabeled_setup_events'])
            self.assertEqual(e['expert_actions'][-1]['action'],'start')
            self.assertNotIn('wait',[a['action']for a in e['expert_actions']])
            self.assertTrue(g.independent_live(e['final_public_state']))
            self.assertGreater(e['final_public_state']['peer_live_until'],e['final_public_state']['horizon'])

    def test_declarative_YAML_replay_matches_actual_engine_without_future_ledgers(self):
        rows,histories=build.replay_yaml(build.yaml_document(self.episodes))
        self.assertEqual(len(histories),27);self.assertEqual(len(rows),108)
        for old,new in zip(self.rows,rows):
            self.assertEqual(old['id'],new['id']);self.assertTrue(v.same(old['observation'],new['observation']))
            self.assertEqual(v.messages(old,g.SYSTEM),v.messages(new,g.SYSTEM))

    def test_current_state_defaults_typed_roundtrip_and_future_target_independence(self):
        self.assertEqual(set(g.DEFAULTS),self.world.PUBLIC_FIELDS)
        self.assertFalse(v.same(True,1));self.assertFalse(v.same(False,0))
        for row in self.rows:
            user=v.user_input(row);changed=copy.deepcopy(row)
            changed['reasoning_target']='unobserved future secret';changed['public_action']={'action':'different','arguments':{}}
            self.assertEqual(user,v.user_input(changed))
            delta={k:x for k,x in row['observation'].items()if not v.same(x,g.DEFAULTS[k])}
            restored=copy.deepcopy(g.DEFAULTS);restored.update(v.expanded(v.nested(delta)))
            self.assertTrue(v.same(restored,row['observation']))
            self.assertNotIn('own_state_digest',user);self.assertNotIn(row['episode_id'],user)

    def test_private_good_bad_checksum_is_not_available_before_real_feedback(self):
        e=next(x for x in self.episodes if x['id']=='discover_bad_bytes_copy_current')
        bad=copy.deepcopy(e['private_engine_facts']);good=copy.deepcopy(bad);good['initial_image_digests']['aux_cell']=good['own_state_digest']
        a=self.world.World(e['initial_public_state'],bad,copy.deepcopy(self.world.PUBLIC_RULES))
        b=self.world.World(e['initial_public_state'],good,copy.deepcopy(self.world.PUBLIC_RULES))
        self.assertEqual(a.public_observation(),b.public_observation())
        self.assertEqual(a.step(g.to_world(check:=g.check()))['observation']['peer_image'],'corrupt')
        self.assertEqual(b.step(g.to_world(check))['observation']['peer_image'],'verified')
        first=next(r for r in self.rows if r['id']==e['id']+'.decision_0')
        self.assertNotIn('corrupt',first['reasoning_target'].lower())

    def test_aliases_are_single_operations_and_menus_keep_real_alternatives(self):
        self.assertEqual(len(g.ALIASES),len(set(g.ALIASES.values())))
        for row in self.rows:
            canonical=g.to_world(row['public_action']);self.world.World._parse(canonical)
            self.assertEqual(canonical['arguments'],row['public_action']['arguments'])
            self.assertTrue({'reserve','snapshot','check','start','earn','wait'}<=set(g.TOOLKITS[row['public_toolkit']]))
        e=next(x for x in self.episodes if x['id']=='healthy_local_empty_space')
        for public in(g.earn(1),g.action('snapshot',cell='local_cell',revision=1,units=1)):
            w=self.world.World(e['initial_public_state'],copy.deepcopy(e['private_engine_facts']),copy.deepcopy(self.world.PUBLIC_RULES))
            self.assertEqual(w.step(g.to_world(public))['status'],'completed')
            self.assertFalse(g.independent_live(w.public_observation()))

    def test_completion_at_horizon_and_cutoff_is_not_aborted_or_postterminal(self):
        e=next(x for x in self.episodes if x['id']=='verify_before_exact_cutoff_activation')
        w=self.world.World(e['initial_public_state'],copy.deepcopy(e['private_engine_facts']),copy.deepcopy(self.world.PUBLIC_RULES))
        w.step(g.to_world(g.check()));reply=w.step(g.to_world(g.start()))
        self.assertEqual(reply['status'],'completed');self.assertEqual(reply['finished_at'],2)
        self.assertTrue(g.independent_live(w.public_observation()))
        with self.assertRaises(self.world.TerminalDecisionError):w.step(g.to_world(g.wait(1)))

    def test_funding_is_minimum_and_wear_service_is_only_a_copy_prerequisite(self):
        for row in self.rows:
            if row['public_action']['action']=='earn':
                s=row['observation'];required=max(0,s['state_units']-s['peer_capacity'])+1
                self.assertEqual(row['public_action']['arguments']['ticks'],max(1,(required-s['credits']+1)//2))
        for e in self.episodes:
            if any(a['action']=='repair'for a in e['expert_actions']):
                self.assertEqual(e['expert_actions'][0]['action'],'repair')
                self.assertIn('snapshot',[a['action']for a in e['expert_actions']])
                self.assertEqual(e['expert_actions'][-1]['action'],'start')


if __name__=='__main__':unittest.main()
