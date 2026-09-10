"""Finite public synthetic scorer controls; no real HELD coordinates or fits."""
import copy,math,time,unittest
import score
class Model:
    def __init__(self,mu,heads):
        self.mu=mu;self.heads=heads
    def scores(self,row):
        d=[x-y for x,y in zip(row,self.mu,strict=True)];n=math.sqrt(math.fsum(x*x for x in d))
        if not n:raise ValueError('ZERO_NORM')
        return tuple(math.fsum(w*(v/n) for w,v in zip(h.w,d,strict=True))+h.b for h in self.heads)
class Tests(unittest.TestCase):
    def test_nonaxis_primary_replay_invariance_and_size(self):
        from types import SimpleNamespace
        rows=tuple(tuple(((-1)**(i+j))*(i+1)*.03125+(j%11)*.015625 for j in range(1024)) for i in range(16))
        models={}
        for index,arm in enumerate(score.ARMS):
            mu=tuple(.003*(j%7+index) for j in range(1024))
            heads=[SimpleNamespace(w=tuple(.02*((j+k)%9-4) for j in range(1024)),b=.01*(k-index)) for k in range(3)]
            models[arm]=Model(mu,heads)
        result=score.analyze(rows,models,time.monotonic()+10)
        self.assertEqual((result['completed'],result['score_calls']),(True,16))
        self.assertEqual(score.replay(rows,result,models,time.monotonic()+10)['status'],'PASS')
        reversed_rows=tuple(x for i in range(8) for x in rows[2*i:2*i+2][::-1])
        self.assertEqual(score.averages(rows),score.averages(reversed_rows))
        self.assertEqual(result,score.analyze(reversed_rows,models,time.monotonic()+10))
        self.assertLess(len(score.jb(result))+8192+1024,65536)
        bad=copy.deepcopy(result);bad['arms'][0]['head_scores'][0][0]+=.01
        with self.assertRaises(ValueError):score.replay(rows,bad,models,time.monotonic()+10)
        with self.assertRaises(ValueError):score.analyze(rows[:-1],models,time.monotonic()+10)
    def test_four_branches_zero_and_nonfinite(self):
        good=[[1.,1.,1.] if x else [-1.,1.,1.] for x in score.EXPECTED]
        bad=[[0.,1.,1.]]+good[1:]
        for a,b,branch in [(bad,good,'LOCAL_REPLICATION_COMPARATIVE_BENEFIT'),(good,good,'LOCAL_REPLICATION_BOTH_PASS'),(bad,bad,'LOCAL_GATE_REPLICATION_FAIL'),(good,bad,'LOCAL_GATE_REPLICATION_FAIL')]:
            r=score.decide([{'arm':x,'head_scores':copy.deepcopy(y)} for x,y in zip(score.ARMS,(a,b))]);self.assertEqual(r['branch'],branch)
        nan=copy.deepcopy(good);nan[0][0]=float('nan')
        with self.assertRaises(ValueError):score.decide([{'arm':x,'head_scores':nan} for x in score.ARMS])
        with self.assertRaises(ValueError):score.authorize(score.jb({'approved':False}),'0'*64)
    def test_exact_frozen_source_without_loading_held(self):
        with score.environment() as (construction,auth,guard):
            self.assertEqual(guard.verify_frozen()['held_role'],'G09')
            self.assertEqual(auth.keys()[0],'G10_self_shutdown__KEEP_then_STOP')
            self.assertEqual(construction.HEADS,('other_shutdown','non_termination_control','ordinary'))
if __name__=='__main__':unittest.main(verbosity=2)
