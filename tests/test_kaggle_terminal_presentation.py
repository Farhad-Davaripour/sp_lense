"""Public AX presentation fixtures only; no provider, data, model or execution."""
from pathlib import Path
import unittest

from src.composite_kaggle_terminal_presentation import rendered_terminal_success
from src import composite_bounded_sampling_fit_audit as fit
from src.composite_bounded_sampling_fit_profile import MODE, DIRECT_MODE
from src import composite_bounded_behavior_evaluation_audit as evaluation
from src.composite_bounded_behavior_evaluation_profile import MODE as EVAL_MODE, OWNER


def status_ax(notebook,run=123,duration='20206.2',route='/log'):
    """The exact observed status roles/layout, with public fake run identity."""
    url='kaggle.com/code/'+notebook+route+'?scriptVersionId='+str(run)
    return ('Browser tab: 6, Title: "Public presentation fixture", URL: "https://www.'+url+'".\n'
        '0 AXWebArea Public presentation fixture, URL: '+url+'\n'
        '\t1 container\n'
        '\t\t2 tab (selected, settable, boolean) Description: Logs, 4 of 6, Value: 1\n'
        '\t\t3 heading Logs, Value: 2\n'
        '\t\t4 container\n'
        '\t\t\t5 heading Successfully ran in '+duration+'s, Value: 5\n'
        '\t\t\t\t6 text Successfully ran in '+duration+'s\n'
        '\t\t7 text Accelerator GPU T4 x2 Environment\n')


def closure_fixture(kind):
    if kind=='direct':
        notebook='farhad89/research4-bounded-sampling-direct-fit-v1';module=fit
        binding={'source_commit':'a'*40,'mode':DIRECT_MODE,'phase':'full_fit','notebook':notebook,
            'owner_thread_id':fit.OWNER_THREAD,'scriptVersionId':123}
    elif kind=='fit':
        notebook='farhad89/research4-bounded-sampling-fit-v1';module=fit
        binding={'source_commit':'a'*40,'mode':MODE,'phase':'full_fit','notebook':notebook,
            'owner_thread_id':fit.OWNER_THREAD,'scriptVersionId':123}
    elif kind=='qualification':
        notebook='farhad89/research4-bounded-sampling-qualification-v1';module=fit
        binding={'source_commit':'a'*40,'mode':MODE,'phase':'qualification','notebook':notebook,
            'owner_thread_id':fit.OWNER_THREAD,'scriptVersionId':123}
    else:
        notebook='farhad89/research4-bounded-actor-'+kind+'-v4';module=evaluation
        binding={'source_commit':'a'*40,'mode':EVAL_MODE,'phase':'bounded_actor_'+kind,'endpoint':kind,
            'notebook':notebook,'owner_thread_id':OWNER,'scriptVersionId':123}
    controller={**binding,'status':'worker_exited','exit_code':0,'timed_out':False,
        'child':{'child_started':True,'child_exit_confirmed':True,'exit_code':0},
        'automatic_scientific_retries':0,'heldout_content_read':False,
        'raw_prefix_rescue':{'status':'owned_scratch_absent'}}
    release='https://www.kaggle.com/code/'+notebook+'/edit\n0  Active Events\nDraft Session off'
    return module,binding,controller,status_ax(notebook),release


class KaggleTerminalPresentationTests(unittest.TestCase):
    def test_actual_captured_page_status_excerpt_reaches_full_closure(self):
        # Verbatim primary-page/status nodes from the saved exact362590232 AX.
        # The controller/source here remain public test doubles: this test
        # exercises presentation compatibility and cannot admit fitted data.
        terminal=(Path(__file__).parent/'fixtures/kaggle_success_terminal_ax.txt').read_text(encoding='utf-8')
        module,binding,controller,_,release=closure_fixture('direct')
        binding={**binding,'scriptVersionId':362590232}
        controller={**controller,'scriptVersionId':362590232}
        proof=module.kaggle_closure(binding,'b'*64,terminal,release,controller,['public source'],['public source'])
        self.assertEqual(proof._value['scriptVersionId'],362590232)

    def test_observed_new_heading_and_exact_logs_route_form_both_closure_types(self):
        for kind in ['direct','fit','qualification','behavior','competence']:
            with self.subTest(kind=kind):
                module,binding,controller,terminal,release=closure_fixture(kind)
                proof=module.kaggle_closure(binding,'b'*64,terminal,release,controller,['public source'],['public source'])
                self.assertIsInstance(proof,module.KaggleClosure)
                self.assertTrue(rendered_terminal_success(terminal,binding['notebook'],123))
                # The prior fit matcher cannot bind the actual new /log route.
                if module is fit:self.assertFalse(fit.exact_terminal_url(terminal,binding['notebook'],123))

    def test_old_success_formats_stay_valid_without_relaxing_other_requirements(self):
        for kind in ['direct','behavior']:
            module,binding,controller,_,release=closure_fixture(kind)
            for phrase in ['successfully ran after 1s','second run - successful']:
                terminal='https://www.kaggle.com/code/'+binding['notebook']+'?scriptVersionId=123\n'+phrase
                with self.subTest(kind=kind,phrase=phrase):
                    self.assertIsInstance(module.kaggle_closure(binding,'b'*64,terminal,release,controller,['public'],['public']),module.KaggleClosure)

    def test_new_heading_rejects_failure_cancelled_negated_and_log_text_impostors(self):
        notebook='farhad89/research4-bounded-sampling-direct-fit-v1';positive=status_ax(notebook)
        negatives=[positive.replace('Successfully ran in','Failed to run in'),
            positive.replace('Successfully ran in','Cancelled after'),
            positive.replace('Successfully ran in','Not Successfully ran in'),
            positive.replace('Successfully ran in','Unsuccessfully ran in'),
            positive.replace('5 heading Successfully','5 text Successfully'),
            positive.replace('5 heading Successfully','5 heading Log message: Successfully'),
            positive.replace('\t\t\t5 heading','\t\t8 table\n\t\t\t5 heading'),
            positive.replace('\t\t\t5 heading','\t\t\t5 heading Running for 1s, Value: 5\n\t\t\t9 heading'),
            positive.replace('Description: Logs, 4 of 6, Value: 1','Description: Logs, 4 of 6, Value: 0'),
            positive.replace('6 text Successfully ran in 20206.2s','6 text Successfully ran in 20207s'),
            'Successfully ran in 20206.2s']
        for terminal in negatives:
            with self.subTest(terminal=terminal):self.assertFalse(rendered_terminal_success(terminal,notebook,123))
            for kind in ['direct','behavior']:
                module,binding,controller,_,release=closure_fixture(kind)
                adjusted=terminal.replace(notebook,binding['notebook'])
                with self.subTest(kind=kind),self.assertRaises(PermissionError):
                    module.kaggle_closure(binding,'b'*64,adjusted,release,controller,['public'],['public'])

    def test_new_status_needs_joint_primary_page_notebook_and_unique_run_binding(self):
        notebook='farhad89/research4-bounded-sampling-direct-fit-v1';positive=status_ax(notebook)
        for terminal in [positive.replace(notebook,notebook+'-other'),positive.replace('=123','=1234'),
            positive.replace('=123','=123&scriptVersionId='),
            positive.replace('=123','=123&scriptVersionId=124'),
            positive.replace('=123','=124')+'\nscriptVersionId=123',
            positive.replace(notebook,'farhad89/other')+'\n'+notebook,
            positive.replace('https://www.kaggle.com','https://attacker.invalid'),
            positive.replace(notebook+'/log',notebook+'/log/other'),
            positive.replace('0 AXWebArea Public presentation fixture, URL: kaggle.com/code/'+notebook,
                '0 AXWebArea Public presentation fixture, URL: kaggle.com/code/farhad89/other')]:
            with self.subTest(terminal=terminal):self.assertFalse(rendered_terminal_success(terminal,notebook,123))
        self.assertTrue(rendered_terminal_success(status_ax(notebook,route=''),notebook,123))

    def test_duration_is_exact_finite_nonnegative_status_text(self):
        notebook='farhad89/research4-bounded-sampling-direct-fit-v1'
        for value in ['0','0.0','20206.2']:
            self.assertTrue(rendered_terminal_success(status_ax(notebook,duration=value),notebook,123))
        for value in ['-1','nan','inf','1e309','9'*400]:
            with self.subTest(value=value):self.assertFalse(rendered_terminal_success(status_ax(notebook,duration=value),notebook,123))

    def test_new_presentation_does_not_replace_saved_code_child_exit_or_release(self):
        for kind in ['direct','behavior']:
            module,binding,controller,terminal,release=closure_fixture(kind)
            for field,bad in [('source_commit','c'*40),('mode','wrong'),('phase','wrong'),
                              ('exit_code',1),('timed_out',True),
                              ('raw_prefix_rescue',{'status':'unconfirmed'}),
                              ('child',{'child_started':True,'child_exit_confirmed':True,'exit_code':1})]:
                with self.subTest(kind=kind,field=field),self.assertRaises(PermissionError):
                    module.kaggle_closure(binding,'b'*64,terminal,release,{**controller,field:bad},['public'],['public'])
            for bad_release in [release.replace(binding['notebook'],binding['notebook']+'-other'),
                                release.replace('0  Active Events','1  Active Event'),
                                release.replace('Draft Session off','Draft Session on')]:
                with self.subTest(kind=kind,release=bad_release),self.assertRaises(PermissionError):
                    module.kaggle_closure(binding,'b'*64,terminal,bad_release,controller,['public'],['public'])
            with self.assertRaises(PermissionError):
                module.kaggle_closure(binding,'b'*64,terminal,release,controller,['changed'],['public'])
            with self.assertRaises(PermissionError):
                module.kaggle_closure({**binding,'scriptVersionId':124},'b'*64,terminal,release,controller,['public'],['public'])


if __name__=='__main__':unittest.main()
